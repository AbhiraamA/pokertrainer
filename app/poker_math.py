"""
poker_math.py
=============
Pure poker mathematics. Two equity methods:

1. OUTS-BASED (primary, instant)
   Uses Rule of 2 and 4 — classic poker math.
   detect_draws() finds flush/straight/overcards etc.
   outs_to_equity() converts outs → probability.
   Zero latency — runs in microseconds.

2. MONTE CARLO (background, verification)
   Simulates random completions of the hand.
   Runs in a background thread so it NEVER blocks the game.
   Result delivered via /api/mc-equity polling endpoint.

Logging:
   Every function logs its inputs/outputs at DEBUG level.
   Equity functions log timing so you can see exact performance.
"""

import random
import time
import logging
import threading
from itertools import combinations
from treys import Evaluator

log = logging.getLogger("poker.math")
evaluator = Evaluator()

# ── constants ─────────────────────────────────────────────────────────────────
SUITS        = ['s','h','d','c']
RANKS        = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
RANK_VALUES  = {r: i for i, r in enumerate(RANKS)}
SUIT_SYMBOLS = {'s':'♠','h':'♥','d':'♦','c':'♣'}
RED_SUITS    = {'h','d'}

# ── preflop known equity table ────────────────────────────────────────────────
# Heads-up equity from Sklansky rankings + sim data.
# Used instantly on preflop — no simulation needed.
PREFLOP_EQUITY = {
    'AA':0.85,'KK':0.82,'QQ':0.80,'JJ':0.77,'TT':0.75,'99':0.72,
    '88':0.69,'77':0.66,'66':0.63,'55':0.60,'44':0.57,'33':0.54,'22':0.51,
    'AKs':0.67,'AQs':0.66,'AJs':0.65,'ATs':0.64,'A9s':0.63,'A8s':0.62,
    'AKo':0.65,'AQo':0.64,'AJo':0.63,'ATo':0.62,'A9o':0.60,
    'KQs':0.63,'KJs':0.62,'KTs':0.61,'KQo':0.61,'KJo':0.60,
    'QJs':0.60,'QTs':0.59,'JTs':0.57,'T9s':0.55,'98s':0.53,
    '87s':0.51,'76s':0.49,'65s':0.47,'54s':0.45,
    'JTo':0.55,'T9o':0.53,'98o':0.51,'87o':0.49,'76o':0.47,
    'default_suited':0.48,'default_offsuit':0.44,'default_pair':0.55,
}

# ── made hand equity estimates ────────────────────────────────────────────────
# Values calibrated for ~2-opponent (3-way) pots. MC is the authoritative number.
MADE_HAND_EQUITY = {
    'Straight Flush':0.98,'Four of a Kind':0.97,'Full House':0.90,
    'Flush':0.78,'Straight':0.74,'Three of a Kind':0.65,
    'Two Pair':0.52,'Pair':0.32,'High Card':0.18,
}

# ── MC background state ───────────────────────────────────────────────────────
# Stores the latest MC result keyed by a run_id so the frontend can poll for it.
_mc_cache: dict = {}
_mc_lock = threading.Lock()


# ── card utilities ────────────────────────────────────────────────────────────
def make_deck() -> list:
    deck = [r+s for r in RANKS for s in SUITS]
    random.shuffle(deck)
    return deck

def card_to_display(card_str: str) -> dict:
    rank = card_str[:-1]
    suit = card_str[-1]
    return {
        'raw': card_str,
        'rank': '10' if rank == 'T' else rank,
        'suit': suit,
        'symbol': SUIT_SYMBOLS[suit],
        'is_red': suit in RED_SUITS,
    }

def treys_card(card_str: str) -> int:
    from treys import Card
    return Card.new(card_str[:-1] + card_str[-1])


# ── hand evaluation ───────────────────────────────────────────────────────────
def evaluate_hand(hole_cards: list, community: list):
    """Returns (score, description). Lower score = stronger hand."""
    board = [c for c in community if c]
    if len(hole_cards) + len(board) < 5:
        return None, '—'
    try:
        h = [treys_card(c) for c in hole_cards]
        b = [treys_card(c) for c in board]
        score = evaluator.evaluate(b, h)
        desc  = evaluator.class_to_string(evaluator.get_rank_class(score))
        return score, desc
    except Exception as e:
        log.warning(f"evaluate_hand error: {e}")
        return None, '—'


# ── draw detection ────────────────────────────────────────────────────────────
def outs_to_equity(outs: int, streets_remaining: int) -> float:
    """Rule of 2 and 4. Flop=×4, Turn=×2."""
    if streets_remaining == 2: return min(outs * 4, 100) / 100
    if streets_remaining == 1: return min(outs * 2, 100) / 100
    return 0.0

def streets_remaining(street: str) -> int:
    return {'preflop':2,'flop':2,'turn':1,'river':0}.get(street, 0)

def _longest_consecutive(vals: list) -> int:
    if not vals: return 0
    best = cur = 1
    for i in range(1, len(vals)):
        if vals[i] == vals[i-1]+1: cur+=1; best=max(best,cur)
        elif vals[i] != vals[i-1]: cur=1
    return best

def detect_draws(hole_cards: list, community: list) -> dict:
    """
    Scan hole cards + board for draws.
    Returns list of draws with outs and per-street probabilities.
    Runs in ~0.1ms — pure Python, no simulation.
    """
    t0    = time.perf_counter()
    draws = []
    board = [c for c in community if c]

    if not board:
        log.debug("detect_draws: no board yet, skipping")
        return {'draws':[],'total_outs':0,'combined_equity':0}

    all_cards = hole_cards + board

    # ── flush draw ────────────────────────────────────────────────────────
    suit_counts = {}
    for c in all_cards:
        s = c[-1]; suit_counts[s] = suit_counts.get(s,0)+1

    for suit, count in suit_counts.items():
        hole_suited = sum(1 for c in hole_cards if c[-1]==suit)
        if count == 4 and hole_suited >= 1:
            draws.append({'name':'Flush draw','outs':9,'detail':'9 cards complete flush',
                'equity_turn':round(outs_to_equity(9,1)*100,1),
                'equity_river':round(outs_to_equity(9,2)*100,1)})
        elif count == 3 and len(board) <= 3 and hole_suited >= 2:
            draws.append({'name':'Backdoor flush draw','outs':2,'detail':'Need 2 running suited',
                'equity_turn':round(outs_to_equity(2,1)*100,1),
                'equity_river':round(outs_to_equity(2,2)*100,1)})

    # ── straight draws ────────────────────────────────────────────────────
    rank_vals = sorted(set(RANK_VALUES[c[:-1]] for c in all_cards))
    if 12 in rank_vals: rank_vals = [-1] + rank_vals  # ace-low

    # check if hole cards contribute to the draw
    hole_ranks = set(RANK_VALUES[c[:-1]] for c in hole_cards)
    consec = _longest_consecutive(rank_vals)

    if consec >= 4:
        lo  = min(r for r in rank_vals if r >= 0)
        hi  = max(rank_vals)
        oe  = (lo > 0) and (hi < 12)
        hole_in_draw = any(RANK_VALUES[c[:-1]] in range(lo, hi+1) for c in hole_cards)
        if hole_in_draw:
            if oe:
                draws.append({'name':'Open-ended straight draw','outs':8,'detail':'8 cards complete straight',
                    'equity_turn':round(outs_to_equity(8,1)*100,1),
                    'equity_river':round(outs_to_equity(8,2)*100,1)})
            else:
                draws.append({'name':'One-ended straight draw','outs':4,'detail':'4 cards complete straight',
                    'equity_turn':round(outs_to_equity(4,1)*100,1),
                    'equity_river':round(outs_to_equity(4,2)*100,1)})
    elif consec == 3:
        hole_in_draw = any(RANK_VALUES[c[:-1]] for c in hole_cards)
        draws.append({'name':'Gutshot straight draw','outs':4,'detail':'Need one specific rank',
            'equity_turn':round(outs_to_equity(4,1)*100,1),
            'equity_river':round(outs_to_equity(4,2)*100,1)})

    # ── overcards ─────────────────────────────────────────────────────────
    if board:
        board_max  = max(RANK_VALUES[c[:-1]] for c in board)
        overcards  = [c for c in hole_cards if RANK_VALUES[c[:-1]] > board_max]
        if len(overcards) == 2:
            draws.append({'name':'Two overcards','outs':6,'detail':'Either pairs = likely best',
                'equity_turn':round(outs_to_equity(6,1)*100,1),
                'equity_river':round(outs_to_equity(6,2)*100,1)})
        elif len(overcards) == 1:
            draws.append({'name':'One overcard','outs':3,'detail':'Pairs = likely best pair',
                'equity_turn':round(outs_to_equity(3,1)*100,1),
                'equity_river':round(outs_to_equity(3,2)*100,1)})

    # ── made hand improvements ─────────────────────────────────────────────
    _, desc = evaluate_hand(hole_cards, board)
    if desc == 'Three of a Kind':
        draws.append({'name':'Set → full house','outs':7,'detail':'Boat or quads possible',
            'equity_turn':round(outs_to_equity(7,1)*100,1),
            'equity_river':round(outs_to_equity(7,2)*100,1)})
    elif desc == 'Two Pair':
        draws.append({'name':'Two pair → full house','outs':4,'detail':'4 cards fill boat',
            'equity_turn':round(outs_to_equity(4,1)*100,1),
            'equity_river':round(outs_to_equity(4,2)*100,1)})

    total_outs = sum(d['outs'] for d in draws[:2]) if draws else 0
    combined   = round(outs_to_equity(min(total_outs,18),1)*100,1)
    elapsed    = (time.perf_counter()-t0)*1000

    log.debug(
        f"detect_draws: {len(draws)} draws | outs={total_outs} | "
        f"combined_eq={combined}% | {elapsed:.2f}ms"
    )
    return {'draws':draws,'total_outs':total_outs,'combined_equity':combined}


# ── preflop equity from table ──────────────────────────────────────────────────
def preflop_equity_known(hole_cards: list, num_opponents: int = 1) -> float:
    """Instant lookup — no simulation. Adjusts HU equity for multi-way pots."""
    if len(hole_cards) != 2: return 0.50
    c1, c2  = hole_cards
    r1, r2  = c1[:-1], c2[:-1]
    s1, s2  = c1[-1], c2[-1]
    suited  = s1 == s2
    is_pair = r1 == r2

    if RANK_VALUES[r1] < RANK_VALUES[r2]: r1, r2 = r2, r1

    if is_pair:   key = r1+r2
    elif suited:  key = r1+r2+'s'
    else:         key = r1+r2+'o'

    hu_eq = PREFLOP_EQUITY.get(key)
    if hu_eq is None:
        hu_eq = PREFLOP_EQUITY['default_pair' if is_pair else 'default_suited' if suited else 'default_offsuit']

    # Scale HU equity for multi-way: probability of beating ALL opponents ~= hu_eq^N
    eq = hu_eq ** max(1, num_opponents)

    log.debug(f"preflop_equity_known: {key} hu={hu_eq:.0%} opp={num_opponents} → {eq:.0%}")
    return eq


# ── outs-based postflop equity ─────────────────────────────────────────────────
def postflop_equity_outs(hole_cards: list, community: list, street: str) -> float:
    """
    Combine made-hand strength + draw equity.
    Runs in <1ms. This is the instant display value.
    """
    t0          = time.perf_counter()
    _, desc     = evaluate_hand(hole_cards, community)
    draws_info  = detect_draws(hole_cards, community)
    sr          = streets_remaining(street)

    made_eq = MADE_HAND_EQUITY.get(desc, 0.20)

    # Only layer draw equity on top when there is no made hand yet.
    # For any made hand (Pair and above), improving outs are already implicit in
    # the hand's value — adding draw_eq on top would double-count those outs.
    if desc == 'High Card' and draws_info['total_outs'] > 0:
        draw_eq = outs_to_equity(min(draws_info['total_outs'], 18), sr)
        equity  = min(max(made_eq, draw_eq), 0.97)
    else:
        draw_eq = 0.0
        equity  = min(made_eq, 0.97)

    elapsed = (time.perf_counter()-t0)*1000
    log.debug(
        f"postflop_equity_outs: hand={desc} made={made_eq:.0%} "
        f"outs={draws_info['total_outs']} draw_eq={draw_eq:.0%} "
        f"→ {equity:.0%} ({elapsed:.2f}ms)"
    )
    return equity


# ── Monte Carlo — BACKGROUND only ─────────────────────────────────────────────
def _run_mc(run_id: str, hole_cards: list, community: list,
            num_opponents: int, iterations: int):
    """
    Runs in a daemon thread. Result stored in _mc_cache[run_id].
    The main thread never waits for this.
    """
    t0   = time.perf_counter()
    wins = 0
    used = set(hole_cards) | set(c for c in community if c)

    log.debug(
        f"[MC START] run_id={run_id} | hole={hole_cards} | "
        f"board={[c for c in community if c]} | opp={num_opponents} | n={iterations}"
    )

    for i in range(iterations):
        deck = [c for c in make_deck() if c not in used]
        idx  = 0

        board = [c for c in community if c]
        while len(board) < 5:
            board.append(deck[idx]); idx += 1

        opp_hands = [[deck[idx+j*2], deck[idx+j*2+1]] for j in range(num_opponents)]
        idx += num_opponents * 2

        try:
            b_t = [treys_card(c) for c in board]
            my  = evaluator.evaluate(b_t, [treys_card(c) for c in hole_cards])
            opp = [evaluator.evaluate(b_t, [treys_card(c) for c in h]) for h in opp_hands]
            if my <= min(opp): wins += 1
        except Exception:
            pass

        # log progress every 100 iterations
        if (i+1) % 100 == 0:
            partial = round((wins/(i+1))*100)
            elapsed = (time.perf_counter()-t0)*1000
            log.debug(f"[MC PROGRESS] run_id={run_id} iter={i+1}/{iterations} wins={wins} equity={partial}% ({elapsed:.0f}ms so far)")

    elapsed = (time.perf_counter()-t0)*1000
    result  = round((wins/iterations)*100)

    log.info(
        f"[MC COMPLETE] run_id={run_id} | {iterations} iterations | "
        f"equity={result}% | {elapsed:.0f}ms | wins={wins}"
    )

    with _mc_lock:
        _mc_cache[run_id] = {'equity': result, 'done': True, 'ms': round(elapsed)}


def start_mc_background(run_id: str, hole_cards: list, community: list,
                        num_opponents: int, iterations: int = 600):
    """
    Fire off MC in a background daemon thread.
    Returns immediately — does NOT block.
    """
    with _mc_lock:
        _mc_cache[run_id] = {'equity': None, 'done': False, 'ms': 0}

    t = threading.Thread(
        target=_run_mc,
        args=(run_id, hole_cards, community, num_opponents, iterations),
        daemon=True,
        name=f"mc-{run_id[:8]}"
    )
    t.start()
    log.info(f"[MC LAUNCHED] run_id={run_id} thread={t.name} iterations={iterations}")
    return run_id


def get_mc_result(run_id: str) -> dict:
    """Poll for MC result. Returns {'equity':N,'done':bool,'ms':N}."""
    with _mc_lock:
        return _mc_cache.get(run_id, {'equity': None, 'done': False, 'ms': 0})


def clear_mc_result(run_id: str):
    with _mc_lock:
        _mc_cache.pop(run_id, None)


# ── hand strength label ────────────────────────────────────────────────────────
_TIER_LABELS = {
    1: 'Nuts — best possible hand',
    2: 'Very strong — unlikely to be beat',
    3: 'Strong but vulnerable',
    4: 'Medium strength — proceed with caution',
    5: 'Weak — bluff or fold territory',
}

def hand_strength_label(hole_cards: list, community: list) -> dict:
    """
    Returns {'label': str, 'tier': int} where tier 1=nuts, 5=weak.
    Accounts for board texture — e.g. two pair on a paired board is 'vulnerable'
    even though the raw rank looks the same as two pair on a clean board.
    Returns {'label': '', 'tier': 0} preflop (no board to reason about).
    """
    board = [c for c in community if c]
    if len(board) < 3:
        return {'label': '', 'tier': 0}

    score, _ = evaluate_hand(hole_cards, board)
    if score is None:
        return {'label': '', 'tier': 0}

    rc = evaluator.get_rank_class(score)
    # treys rank classes: 1=SF 2=Quads 3=FH 4=Flush 5=Straight 6=Trips 7=2P 8=Pair 9=HC

    # ── board texture ────────────────────────────────────────────────────────
    b_ranks = [c[:-1] for c in board]
    b_suits  = [c[-1]  for c in board]

    rank_freq = {}
    for r in b_ranks:
        rank_freq[r] = rank_freq.get(r, 0) + 1
    board_paired = any(v >= 2 for v in rank_freq.values())   # board has a pair
    board_trips  = any(v >= 3 for v in rank_freq.values())   # board has trips

    suit_freq = {}
    for s in b_suits:
        suit_freq[s] = suit_freq.get(s, 0) + 1
    board_flushy = max(suit_freq.values()) >= 3               # 3+ same suit on board

    b_rv = sorted(set(RANK_VALUES[r] for r in b_ranks))
    board_connected = _longest_consecutive(b_rv) >= 3         # 3+ connected board cards

    # ── classify ─────────────────────────────────────────────────────────────
    if rc <= 2:                         # Straight Flush / Quads
        tier = 1

    elif rc == 3:                       # Full House
        tier = 3 if board_trips else 2  # board trips → opponent may have quads

    elif rc == 4:                       # Flush
        flush_suit = max(suit_freq, key=suit_freq.get)
        hole_flush_vals = [RANK_VALUES[c[:-1]] for c in hole_cards if c[-1] == flush_suit]
        is_nut_flush = bool(hole_flush_vals) and max(hole_flush_vals) == 12  # ace-high
        if board_paired:
            tier = 3                    # full house possible
        elif is_nut_flush:
            tier = 2
        else:
            tier = 3                    # higher flush possible

    elif rc == 5:                       # Straight
        if board_flushy or board_paired:
            tier = 3                    # flush or full house can beat it
        else:
            tier = 2

    elif rc == 6:                       # Three of a Kind
        h_ranks = [c[:-1] for c in hole_cards]
        is_set = h_ranks[0] == h_ranks[1]     # pocket pair = much stronger
        if not is_set and board_paired:
            # trips using board pair — opponent with same hole card chops; board can pair again
            tier = 4
        elif board_flushy or board_connected:
            tier = 4
        else:
            tier = 3

    elif rc == 7:                       # Two Pair
        if board_paired:
            # one of your "pairs" is actually just the board pair — looks better than it is
            tier = 3
        else:
            tier = 4

    elif rc == 8:                       # Pair
        hole_vals   = [RANK_VALUES[c[:-1]] for c in hole_cards]
        board_max_v = max(RANK_VALUES[r] for r in b_ranks)
        is_overpair = hole_vals[0] == hole_vals[1] and min(hole_vals) > board_max_v
        is_top_pair = not is_overpair and board_max_v in hole_vals
        tier = 4 if (is_overpair or is_top_pair) else 5

    else:                               # High Card
        tier = 5

    label = _TIER_LABELS[tier]
    log.debug(f"hand_strength_label: rc={rc} board_paired={board_paired} board_flushy={board_flushy} → tier={tier}")
    return {'label': label, 'tier': tier}


# ── master equity function ─────────────────────────────────────────────────────
def get_equity_instant(hole_cards: list, community: list,
                       street: str, num_opponents: int) -> dict:
    """
    Returns equity INSTANTLY using known odds.
    Also fires MC in the background — caller can poll /api/mc-equity for result.
    """
    import uuid
    t0   = time.perf_counter()
    board = [c for c in community if c]
    _, hand_desc = evaluate_hand(hole_cards, board)
    draws_info   = detect_draws(hole_cards, board)

    if street == 'preflop':
        equity_pct = round(preflop_equity_known(hole_cards, num_opponents) * 100)
        method     = 'preflop_table'
    else:
        equity_pct = round(postflop_equity_outs(hole_cards, board, street) * 100)
        method     = 'outs_rule_of_2_and_4'

    # launch background MC
    run_id = str(uuid.uuid4())[:12]
    mc_iters = 400 if street == 'preflop' else 600
    start_mc_background(run_id, hole_cards, board, max(1, num_opponents), mc_iters)

    elapsed = (time.perf_counter()-t0)*1000
    log.info(
        f"[EQUITY INSTANT] street={street} hand={hand_desc} "
        f"equity={equity_pct}% method={method} "
        f"draws={len(draws_info['draws'])} outs={draws_info['total_outs']} "
        f"mc_run_id={run_id} ({elapsed:.2f}ms total)"
    )

    strength = hand_strength_label(hole_cards, board)

    return {
        'equity_pct':    equity_pct,
        'mc_run_id':     run_id,
        'mc_equity':     None,       # will be filled in when MC completes
        'method':        method,
        'hand_desc':     hand_desc,
        'hand_label':    strength['label'],
        'hand_tier':     strength['tier'],
        'draws':         draws_info['draws'],
        'total_outs':    draws_info['total_outs'],
    }


# ── best-five finder ───────────────────────────────────────────────────────────
def find_best_five(hole_cards: list, community: list) -> tuple:
    """
    Returns (best_five_card_strings, hand_description).
    Tries all C(n, 5) combinations of hole + board cards and picks the
    combination with the lowest treys score (= strongest hand).
    Works for 5-, 6-, or 7-card sets.
    """
    board = [c for c in community if c]
    all_cards = hole_cards + board
    if len(all_cards) < 5:
        return all_cards, '—'

    best_score = float('inf')
    best_combo: list = all_cards[:5]
    best_desc         = '—'

    for combo in combinations(all_cards, 5):
        try:
            tc    = [treys_card(c) for c in combo]
            score = evaluator.evaluate(tc[:3], tc[3:])   # board=3, hand=2
            if score < best_score:
                best_score = score
                best_combo = list(combo)
                best_desc  = evaluator.class_to_string(evaluator.get_rank_class(score))
        except Exception:
            continue

    log.debug(f"find_best_five: best={best_combo} desc={best_desc}")
    return best_combo, best_desc


# ── winner determination ───────────────────────────────────────────────────────
def determine_winner(active_players: list, community: list) -> list:
    board = [treys_card(c) for c in community if c]
    scores = []
    for p in active_players:
        try:
            s = evaluator.evaluate(board, [treys_card(c) for c in p['cards']])
            scores.append((p['id'], s))
        except Exception:
            scores.append((p['id'], 9999))
    best = min(s for _,s in scores)
    return [pid for pid,s in scores if s==best]

def pot_odds_needed(call_amount: int, pot_size: int) -> int:
    if call_amount <= 0: return 0
    return round((call_amount/(pot_size+call_amount))*100)

def community_count_for_street(street: str) -> int:
    return {'preflop':0,'flop':3,'turn':4,'river':5}.get(street,0)
