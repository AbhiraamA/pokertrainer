"""
ollama_service.py
=================
All Ollama / LLM communication lives here.

Two functions:
  get_bot_action()  — rule-based by default, LLM optional
  get_coaching()    — always LLM, called after human decision

Every LLM call is fully logged:
  - model, prompt sent, raw response, parsed result, latency
"""

import json
import re
import random
import time
import logging
import requests

log = logging.getLogger("poker.ollama")

OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:4b"   # gemma 4 via Ollama

# ─── Personality rule tables ──────────────────────────────────────────────────
RULES = {
    'tight':   {'fold_vs_bet': 0.45, 'raise_freq': 0.15},
    'loose':   {'fold_vs_bet': 0.15, 'raise_freq': 0.35},
    'maniac':  {'fold_vs_bet': 0.05, 'raise_freq': 0.65},
    'calling': {'fold_vs_bet': 0.05, 'raise_freq': 0.05},
}

# Bet sizing as fraction of pot (lo, hi) — randomized each decision
_BET_RANGES = {
    'tight':   (0.50, 0.70),
    'loose':   (0.60, 0.90),
    'maniac':  (0.75, 1.50),
    'calling': (0.40, 0.60),
}

# Preflop hand tier sets (keys match PREFLOP_EQUITY format)
_PREMIUM  = frozenset({'AA','KK','QQ','JJ','AKs','AKo'})
_STRONG   = frozenset({'TT','99','AQs','AJs','AQo','KQs'})
_MARGINAL = frozenset({
    '88','77','66','ATs','A9s','KJs','KTs','QJs','JTs',
    'T9s','98s','87s','76s','65s','54s',
})

_RANK_ORDER = {
    '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
    'T':10,'J':11,'Q':12,'K':13,'A':14,
}

BOT_PERSONALITIES = {
    'tight': {
        'name': 'TAG', 'label': 'Tight-Aggressive',
        'description': 'Plays few hands, bets aggressively when in',
        'system': (
            "You are a tight-aggressive poker bot. You fold most hands preflop. "
            "When you do play, you bet and raise aggressively for value. "
            "Respond ONLY with valid JSON: "
            '{"action":"fold"|"call"|"raise"|"check"|"bet","amount":number_or_null,"thought":"under 8 words"}'
        ),
    },
    'loose': {
        'name': 'LAG', 'label': 'Loose-Aggressive',
        'description': 'Wide range, applies constant pressure',
        'system': (
            "You are a loose-aggressive poker bot. You play many hands and frequently raise. "
            "You bluff often and put pressure on opponents constantly. "
            "Respond ONLY with valid JSON: "
            '{"action":"fold"|"call"|"raise"|"check"|"bet","amount":number_or_null,"thought":"under 8 words"}'
        ),
    },
    'maniac': {
        'name': 'MAN', 'label': 'Maniac',
        'description': 'Hyper-aggressive, raises almost everything',
        'system': (
            "You are a maniac poker bot. You raise and re-raise almost every hand. "
            "You bluff constantly and almost never fold. "
            "Respond ONLY with valid JSON: "
            '{"action":"fold"|"call"|"raise"|"check"|"bet","amount":number_or_null,"thought":"under 8 words"}'
        ),
    },
    'calling': {
        'name': 'STA', 'label': 'Calling Station',
        'description': 'Calls almost everything, rarely raises',
        'system': (
            "You are a calling station poker bot. You call almost every bet. "
            "You rarely raise and almost never fold. "
            "Respond ONLY with valid JSON: "
            '{"action":"fold"|"call"|"raise"|"check"|"bet","amount":number_or_null,"thought":"under 8 words"}'
        ),
    },
}

HINT_SYSTEM = (
    "You are a poker coach giving a Socratic hint. Give exactly one sentence that makes "
    "the player think about the right decision without ever revealing the correct action. "
    "Reference the specific numbers in the hand. "
    "Never say the words fold, call, or raise directly. "
    "Never reveal the correct answer. "
    "Return ONLY the one hint sentence — no JSON, no preamble, no quotes."
)

COACH_SYSTEM = """\
You are a friendly poker coach explaining decisions to a beginner. \
Talk like a knowledgeable friend, not a textbook. \
Monte Carlo equity is more accurate than outs estimates — prefer it in your analysis. \
Never use jargon like "GTO", "balanced ranges", "polarized", "merged", or "solver". \
Instead say things like "you were ahead", "calling was too expensive", "a raise here puts pressure on them". \
Analyze the player's decision and respond in EXACTLY this JSON format:
{
  "verdict": "optimal" | "acceptable" | "mistake",
  "verdictText": "3-5 word plain-english label e.g. 'Good call', 'Fold was right', 'Too expensive to call'",
  "equity": <estimated equity 0-100 as integer>,
  "why": "1-2 short sentences — did the action make sense given your odds and what you were likely up against?",
  "optimal": "1-2 short sentences — what the best play was and why a beginner should remember it",
  "reasoning": "1-2 short sentences — what the board and opponents' likely hands tell you about this spot"
}
Respond ONLY with valid JSON. Reference the EXACT equity and pot-odds percentages provided."""


# ─── Low-level Ollama call ────────────────────────────────────────────────────
def _ollama_call(system_prompt: str, user_prompt: str, model: str) -> tuple[str, float]:
    """
    Send prompt to Ollama. Returns (response_text, latency_ms).
    Logs the full prompt and response at DEBUG level.
    """
    payload = {
        "model":   model,
        "prompt":  f"System: {system_prompt}\n\nUser: {user_prompt}",
        "stream":  False,
        "options": {"temperature": 0.6, "num_predict": 350},
    }

    log.debug(
        f"\n{'─'*60}\n"
        f"[LLM REQUEST] model={model}\n"
        f"--- SYSTEM ---\n{system_prompt}\n"
        f"--- USER ---\n{user_prompt}\n"
        f"{'─'*60}"
    )

    t0 = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        raw      = resp.json().get("response", "")
        latency  = (time.time() - t0) * 1000
        log.debug(
            f"\n{'─'*60}\n"
            f"[LLM RESPONSE] ({latency:.0f}ms)\n{raw}\n"
            f"{'─'*60}"
        )
        return raw, latency
    except Exception as e:
        latency = (time.time() - t0) * 1000
        log.error(f"[LLM ERROR] ({latency:.0f}ms) {e}")
        raise


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a string (LLMs often add preamble)."""
    match = re.search(r'\{[\s\S]*?\}', text)
    if not match:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    return json.loads(match.group(0))


# ─── Hole-card helpers ────────────────────────────────────────────────────────
def _parse_hole(cards: list) -> tuple:
    """Returns (r1, r2, is_pair, is_suited) with r1 = higher rank."""
    if len(cards) != 2:
        return 'A', 'K', False, False
    c1, c2 = cards
    r1, r2 = c1[:-1], c2[:-1]
    s1, s2 = c1[-1], c2[-1]
    if _RANK_ORDER.get(r1, 0) < _RANK_ORDER.get(r2, 0):
        r1, r2, s1, s2 = r2, r1, s2, s1
    return r1, r2, (r1 == r2), (s1 == s2 and r1 != r2)


def preflop_tier(hole_cards: list) -> str:
    r1, r2, is_pair, is_suited = _parse_hole(hole_cards)
    if is_pair:     key = r1 + r2
    elif is_suited: key = r1 + r2 + 's'
    else:           key = r1 + r2 + 'o'
    if key in _PREMIUM:  return 'premium'
    if key in _STRONG:   return 'strong'
    if key in _MARGINAL: return 'marginal'
    return 'trash'


def _bet_size(pot: int, stack: int, personality_key: str,
              human_tendencies: dict = None) -> int:
    """Pot-fraction bet with randomness. Scales up when human calls too much."""
    lo, hi = _BET_RANGES.get(personality_key, (0.50, 0.75))
    if human_tendencies and human_tendencies.get('call_pct', 0) > 70:
        lo = min(lo * 1.25, 1.0)
        hi = min(hi * 1.25, 2.0)
    frac = random.uniform(lo, hi)
    base = max(pot, 10)
    return min(max(int(frac * base), 10), stack)


# ─── Rule-based bot action ────────────────────────────────────────────────────
_THOUGHTS = {
    'tight':   {'fold': 'Not worth it',        'raise': 'Strong hand, betting', 'call': 'Pot odds are fine',  'check': 'Checking my hand', 'bet': 'Value bet'},
    'loose':   {'fold': 'Too much pressure',    'raise': 'Apply pressure',       'call': 'Calling wide',       'check': 'Trapping',          'bet': 'Taking control'},
    'maniac':  {'fold': 'Unbelievable fold',    'raise': 'Always raising',       'call': 'Fine, calling',      'check': 'Slow playing',      'bet': 'Max pressure'},
    'calling': {'fold': 'Too weak to continue', 'raise': 'Raising for once',     'call': 'Calling as usual',   'check': 'Checking along',    'bet': 'Rare bet'},
}


def rule_based_action(
    to_call: int,
    stack: int,
    pot: int,
    personality_key: str,
    *,
    street: str = 'preflop',
    hole_cards: list = None,
    position: str = 'mid',
    equity_pct: float = None,
    req_equity: float = None,
    human_tendencies: dict = None,
) -> dict:
    """
    Improved rule-based decision. No LLM call.

    Features:
    - Preflop hand-strength tiers (premium / strong / marginal / trash)
    - Position awareness (early / mid / late / blind)
    - Pot-odds floor: force fold when equity << required
    - Pot-percentage bet sizing with personality-based randomness
    - Adapts bluff/call behaviour to observed human tendencies
    """
    r = RULES.get(personality_key, RULES['tight'])
    t = _THOUGHTS.get(personality_key, _THOUGHTS['tight'])

    # ── Pot-odds floor ──────────────────────────────────────────────────────
    # Force fold when equity is more than 20 points below break-even
    if to_call > 0 and equity_pct is not None and req_equity is not None:
        if equity_pct < req_equity - 20:
            log.debug(
                f"rule_based: {personality_key} FORCED FOLD "
                f"(eq={equity_pct:.0f}% << req={req_equity:.0f}%)"
            )
            return {'action': 'fold', 'amount': None, 'thought': 'Not enough equity'}

    # ── Bluff-frequency modifier from human tendencies ──────────────────────
    bluff_mod = 0.0
    if human_tendencies and human_tendencies.get('total', 0) >= 5:
        fold_pct = human_tendencies.get('fold_pct', 50)
        call_pct = human_tendencies.get('call_pct', 50)
        if fold_pct > 60:
            bluff_mod = +0.20   # human folds too much → bluff more
        elif call_pct > 70:
            bluff_mod = -0.20   # human calls too much → value-bet only

    raise_freq = min(max(r['raise_freq'] + bluff_mod, 0.0), 1.0)
    fold_freq  = r['fold_vs_bet']

    # ── Preflop tier-based logic ────────────────────────────────────────────
    if street == 'preflop' and hole_cards:
        tier = preflop_tier(hole_cards)

        # Which tiers are playable from this position
        _position_tiers = {
            'early': ('premium', 'strong'),
            'mid':   ('premium', 'strong', 'marginal'),
            'late':  ('premium', 'strong', 'marginal', 'trash'),
            'blind': ('premium', 'strong', 'marginal', 'trash'),
        }
        allowed = _position_tiers.get(position, ('premium', 'strong', 'marginal'))

        if to_call == 0:
            # No bet facing — check or open-raise
            if tier == 'premium' and random.random() < raise_freq + 0.30:
                amt = _bet_size(max(pot, 20), stack, personality_key, human_tendencies)
                log.debug(f"rule_based: {personality_key} pf-premium → bet ${amt}")
                return {'action': 'bet', 'amount': amt, 'thought': t['bet']}
            if tier == 'strong' and random.random() < raise_freq + 0.10:
                amt = _bet_size(max(pot, 20), stack, personality_key, human_tendencies)
                log.debug(f"rule_based: {personality_key} pf-strong → bet ${amt}")
                return {'action': 'bet', 'amount': amt, 'thought': t['bet']}
            if tier == 'marginal' and random.random() < raise_freq:
                amt = _bet_size(max(pot, 20), stack, personality_key, human_tendencies)
                log.debug(f"rule_based: {personality_key} pf-marginal → bet ${amt}")
                return {'action': 'bet', 'amount': amt, 'thought': t['bet']}
            return {'action': 'check', 'amount': None, 'thought': t['check']}

        # Facing a raise preflop
        if tier not in allowed:
            log.debug(f"rule_based: {personality_key} pf-{tier} → fold (pos={position})")
            return {'action': 'fold', 'amount': None, 'thought': t['fold']}

        if tier == 'premium':
            # Almost always continue; 3-bet often
            if random.random() < raise_freq + 0.30 and stack > to_call:
                amt = min(to_call * 3, stack)
                return {'action': 'raise', 'amount': amt, 'thought': t['raise']}
            return {'action': 'call', 'amount': min(to_call, stack), 'thought': t['call']}

        if tier == 'strong':
            if random.random() < raise_freq and stack > to_call:
                amt = min(to_call * 3, stack)
                return {'action': 'raise', 'amount': amt, 'thought': t['raise']}
            return {'action': 'call', 'amount': min(to_call, stack), 'thought': t['call']}

        if tier == 'marginal':
            fp = fold_freq + (0.30 if position == 'early' else 0.0)
            if random.random() < fp:
                return {'action': 'fold', 'amount': None, 'thought': t['fold']}
            return {'action': 'call', 'amount': min(to_call, stack), 'thought': t['call']}

        # trash from allowed position (late/blind speculative)
        if random.random() < 0.85:
            return {'action': 'fold', 'amount': None, 'thought': t['fold']}
        return {'action': 'call', 'amount': min(to_call, stack), 'thought': t['call']}

    # ── Postflop / equity-based logic ───────────────────────────────────────
    if to_call == 0:
        strong_hand = equity_pct is not None and equity_pct > 55
        if strong_hand and random.random() < raise_freq + 0.15:
            amt = _bet_size(max(pot, 10), stack, personality_key, human_tendencies)
            log.debug(f"rule_based: {personality_key} postflop value-bet ${amt} (eq={equity_pct})")
            return {'action': 'bet', 'amount': amt, 'thought': t['bet']}
        if not strong_hand and random.random() < raise_freq:
            amt = _bet_size(max(pot, 10), stack, personality_key, human_tendencies)
            log.debug(f"rule_based: {personality_key} postflop bluff ${amt}")
            return {'action': 'bet', 'amount': amt, 'thought': t['bet']}
        return {'action': 'check', 'amount': None, 'thought': t['check']}

    if random.random() < fold_freq:
        return {'action': 'fold', 'amount': None, 'thought': t['fold']}

    if random.random() < raise_freq and stack > to_call:
        base = max(pot + to_call * 2, to_call * 2)
        amt  = max(_bet_size(base, stack, personality_key, human_tendencies), to_call * 2)
        amt  = min(amt, stack)
        log.debug(f"rule_based: {personality_key} postflop raise ${amt}")
        return {'action': 'raise', 'amount': amt, 'thought': t['raise']}

    call_amt = min(to_call, stack)
    log.debug(f"rule_based: {personality_key} → call ${call_amt}")
    return {'action': 'call', 'amount': call_amt, 'thought': t['call']}


# ─── Bot action (LLM optional) ────────────────────────────────────────────────
def get_bot_action(personality_key: str, game_state: dict, model: str, use_llm: bool = False) -> dict:
    """
    Get bot action. By default uses rule_based_action (instant).
    If use_llm=True, sends to Ollama with rule-based fallback.
    """
    to_call = game_state['to_call']
    stack   = game_state['stack']
    pot     = game_state['pot']

    if not use_llm or stack <= 0:
        return rule_based_action(
            to_call, stack, pot, personality_key,
            street=game_state.get('street', 'preflop'),
            hole_cards=game_state.get('hole_cards'),
            position=game_state.get('position', 'mid'),
            equity_pct=game_state.get('equity_pct'),
            req_equity=game_state.get('req_equity'),
            human_tendencies=game_state.get('human_tendencies'),
        )

    personality = BOT_PERSONALITIES[personality_key]
    user_prompt = (
        f"Your hole cards: {', '.join(game_state['hole_cards'])}\n"
        f"Community: {', '.join(game_state['community']) or 'none'}\n"
        f"Street: {game_state['street']} | Pot: ${pot} | To call: ${to_call} | Stack: ${stack}\n"
        f"Recent: {', '.join(game_state.get('action_history', [])[-4:]) or 'none'}\n"
        f"IMPORTANT: amount must be between ${min(to_call*2, stack)} and ${stack}.\n"
        f"Respond ONLY with valid JSON."
    )

    try:
        raw, latency = _ollama_call(personality['system'], user_prompt, model)
        result = _extract_json(raw)

        action = result.get('action', 'call')
        amount = result.get('amount')

        # sanitize
        if amount is not None:
            amount = max(0, min(int(amount), stack))
            if amount == 0: amount = None

        if action == 'raise' and (amount is None or amount <= to_call):
            amount = min(max(to_call * 2, pot // 2), stack)
        if action == 'call' and to_call == 0:
            action, amount = 'check', None
        if action == 'check' and to_call > 0:
            action, amount = 'call', min(to_call, stack)

        log.info(f"[BOT LLM] {personality_key} → {action} ${amount or ''} ({latency:.0f}ms) | thought: {result.get('thought','')}")
        return {'action': action, 'amount': amount, 'thought': result.get('thought', '')}

    except Exception as e:
        log.warning(f"[BOT LLM FALLBACK] {personality_key}: {e} — using rules")
        return rule_based_action(to_call, stack, pot, personality_key)


# ─── Verdict logic ────────────────────────────────────────────────────────────
def _rule_verdict(action: str, primary_equity, req_equity, to_call: int) -> tuple[str, str]:
    """
    Derive verdict and plain-English label from pot-odds math.
    Returns (verdict, verdictText).

    fold   → mistake iff equity > req_equity (gave up a profitable spot)
    call   → mistake iff equity < req_equity (called without the odds)
    raise/bet → rule-based on equity; LLM detail preserved in 'why'
    check  → always at least acceptable; can't be a pot-odds mistake
    """
    if primary_equity is None:
        return 'acceptable', 'Hard to judge'

    eq = float(primary_equity)

    if to_call == 0:
        # No bet facing — pot-odds math doesn't apply
        if action == 'fold':
            return 'mistake', 'Folded for free'
        if action == 'check':
            return 'optimal', 'Good check'
        if action in ('bet', 'raise'):
            if eq >= 55:
                return 'optimal', 'Good bet'
            if eq >= 35:
                return 'acceptable', 'Reasonable bet'
            return 'mistake', 'Too aggressive here'
        return 'acceptable', 'Reasonable play'

    req = float(req_equity) if req_equity is not None else 0.0
    margin = eq - req  # positive = equity beats the price

    if action == 'fold':
        if margin > 5:
            return 'mistake', 'Should have called'
        if margin >= -3:
            return 'acceptable', 'Close fold'
        return 'optimal', 'Good fold'

    if action == 'call':
        if margin < -5:
            return 'mistake', 'Too expensive to call'
        if margin < 3:
            return 'acceptable', 'Marginal call'
        return 'optimal', 'Good call'

    if action in ('raise', 'bet'):
        if eq < 25:
            return 'mistake', 'Too aggressive here'
        if eq >= 55:
            return 'optimal', 'Well-timed raise'
        return 'acceptable', 'Reasonable raise'

    return 'acceptable', 'Reasonable play'


# ─── Coaching (always LLM) ────────────────────────────────────────────────────
def get_coaching(params: dict) -> dict:
    """
    Analyze the human player's decision and return structured GTO coaching.
    Always uses LLM. Falls back to a helpful error message if Ollama is down.
    """
    draws_str = ', '.join(d['name'] for d in params.get('draws', [])) or 'none detected'

    mc_pct = params.get('mc_equity_pct')
    has_mc = mc_pct is not None and mc_pct != params['equity_pct']
    primary_equity = mc_pct if has_mc else params['equity_pct']
    equity_label   = f"{primary_equity}% (Monte Carlo)" if has_mc else f"{params['equity_pct']}% (outs estimate)"

    user_prompt = (
        f"Hole cards: {', '.join(params['hole_cards'])}\n"
        f"Community: {', '.join(params['community']) or 'none (preflop)'}\n"
        f"Street: {params['street']} | Pot: ${params['pot']} | To call: ${params['to_call']}\n"
        f"Stack: ${params['stack']} | Active opponents: {params['num_opponents']}\n"
        f"Current hand: {params.get('hand_desc', '—')}\n"
        f"Draws detected: {draws_str}\n"
        f"Equity: {equity_label}\n"
        + (f"Outs-based estimate: {params['equity_pct']}%\n" if has_mc else "")
        + f"Equity needed to call profitably: {params['req_equity']}%\n"
        f"Player chose: {params['action']}"
        + (f" to ${params['raise_amount']}" if params.get('raise_amount') else "") + "\n"
        f"Recent actions: {' → '.join(params.get('action_history', [])[-6:]) or 'start of hand'}\n\n"
        f"Analyze this decision. Reference the exact equity numbers above."
    )

    model = params.get('model', DEFAULT_MODEL)

    try:
        raw, latency = _ollama_call(COACH_SYSTEM, user_prompt, model)
        result = _extract_json(raw)

        verdict, verdict_text = _rule_verdict(
            params['action'], primary_equity,
            params.get('req_equity'), params.get('to_call', 0)
        )
        coaching = {
            'verdict':     verdict,
            'verdictText': verdict_text,
            'equity':      result.get('equity', primary_equity),
            'why':         result.get('why', ''),
            'optimal':     result.get('optimal', ''),
            'reasoning':   result.get('reasoning', ''),
        }
        log.info(
            f"[COACHING] verdict={coaching['verdict']} ({verdict_text}) | "
            f"action={params['action']} | equity={primary_equity}% "
            f"req={params.get('req_equity')}% | latency={latency:.0f}ms"
        )
        return coaching

    except Exception as e:
        log.error(f"[COACHING ERROR] {e}")
        verdict, verdict_text = _rule_verdict(
            params['action'], primary_equity,
            params.get('req_equity'), params.get('to_call', 0)
        )
        return {
            'verdict':     verdict,
            'verdictText': verdict_text,
            'equity':      primary_equity,
            'why':         f'Ollama not responding ({e}). Make sure it is running.',
            'optimal':     'Run: ollama serve',
            'reasoning':   f'Then: ollama pull {model}',
        }


# ─── Hand lesson (one-sentence takeaway) ─────────────────────────────────────
HAND_LESSON_SYSTEM = (
    "You are a poker coach reviewing a student's hand. "
    "In exactly one plain English sentence, what is the single most important thing this player "
    "should learn from this hand? Reference specific numbers if relevant. "
    "Never use jargon like GTO, range, polarized, or EV. Be direct and actionable. "
    "Return ONLY the one sentence — no JSON, no preamble, no quotes."
)


def get_hand_lesson(decisions: list, model: str) -> str:
    """Return a one-sentence lesson derived from this hand's decisions."""
    if not decisions:
        return "Look for spots where your equity clearly beats the pot-odds threshold before deciding to continue."

    mistakes = [d for d in decisions if d.get('coaching', {}).get('verdict') == 'mistake']
    lines = []
    for d in decisions[:6]:
        c = d.get('coaching', {})
        eq  = c.get('equity', '?')
        req = d.get('req_equity', '?')
        why = c.get('why', '').strip()
        lines.append(
            f"  {d['street']}: {d['action']} — {c.get('verdict','?')} "
            f"(equity {eq}%, needed {req}%): {why}"
        )

    prompt = (
        f"This player made {len(decisions)} decision(s) this hand, {len(mistakes)} of which "
        f"{'was' if len(mistakes)==1 else 'were'} a mistake.\n\n"
        "Decisions:\n" + "\n".join(lines) + "\n\n"
        "In exactly one plain English sentence, what is the single most important thing "
        "this player should learn from this hand?"
    )
    try:
        raw, _ = _ollama_call(HAND_LESSON_SYSTEM, prompt, model)
        lesson = raw.strip().strip('"').strip("'").strip('`').strip()
        if lesson.startswith('{'):
            m = re.search(r'"[^"]{10,}"', lesson)
            lesson = m.group(0).strip('"') if m else lesson
        return lesson or "Always compare your equity to the pot-odds threshold before continuing."
    except Exception as e:
        log.warning(f"[HAND LESSON ERROR] {e}")
        return "Always compare your equity to the pot-odds threshold before continuing."


# ─── Hint (Socratic, one sentence) ───────────────────────────────────────────
def get_hint(params: dict) -> str:
    """Return one Socratic hint sentence for the current decision."""
    mc_pct = params.get('mc_equity_pct')
    has_mc = mc_pct is not None
    primary_equity = mc_pct if has_mc else params.get('equity_pct')
    equity_label = (f"{primary_equity}% (Monte Carlo)" if has_mc
                    else f"{params.get('equity_pct')}% (outs estimate)")

    user_prompt = (
        f"Hole cards: {', '.join(params['hole_cards'])}\n"
        f"Community: {', '.join(str(c) for c in params['community']) or 'none (preflop)'}\n"
        f"Street: {params['street']} | Pot: ${params['pot']} | To call: ${params['to_call']}\n"
        f"Stack: ${params['stack']} | Active opponents: {params.get('num_opponents', 1)}\n"
        f"Current hand: {params.get('hand_desc', '—')}\n"
        f"Equity: {equity_label}\n"
        f"Equity needed to call profitably: {params.get('req_equity', '—')}%\n"
        f"Recent actions: {' → '.join(params.get('action_history', [])[-6:]) or 'start of hand'}\n\n"
        "Give one Socratic hint sentence to help the player think about this decision."
    )
    model = params.get('model', DEFAULT_MODEL)
    try:
        raw, _ = _ollama_call(HINT_SYSTEM, user_prompt, model)
        hint = raw.strip().strip('"').strip("'").strip('`').strip()
        # strip accidental JSON wrapper
        if hint.startswith('{'):
            m = re.search(r'"[^"]{10,}"', hint)
            hint = m.group(0).strip('"') if m else hint
        return hint or 'Think carefully about whether the numbers favour continuing here.'
    except Exception as e:
        log.warning(f"[HINT ERROR] {e}")
        return 'Ollama is not responding — check that it is running.'
