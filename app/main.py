"""
main.py — FastAPI routes

Endpoints:
  GET  /                → frontend
  POST /api/setup       → configure
  POST /api/deal        → new hand
  GET  /api/state       → current state
  POST /api/action      → human action + coaching
  POST /api/bot-action  → one bot turn
  GET  /api/equity      → instant equity (outs-based) + launches MC background
  GET  /api/mc-equity   → poll for MC result (non-blocking)

MC design:
  /api/equity  fires MC in a daemon thread and returns immediately.
  Frontend polls /api/mc-equity every 300ms until done=True.
  Game never waits for MC.
"""

import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from .game_state import GameState
from .ollama_service import get_bot_action, get_coaching, get_hint, get_hand_lesson, BOT_PERSONALITIES
from .poker_math import (
    community_count_for_street, get_equity_instant,
    get_mc_result, clear_mc_result, pot_odds_needed,
    preflop_equity_known, postflop_equity_outs,
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("poker.api")

app  = FastAPI(title="Poker Trainer v4")
game = GameState()
app.mount("/static", StaticFiles(directory="static"), name="static")


class SetupRequest(BaseModel):
    num_opponents:    int  = 2
    model:            str  = "gemma3:4b"
    use_llm_for_bots: bool = False
    starting_stack:   int  = 1000

class ActionRequest(BaseModel):
    action: str
    amount: Optional[int] = None


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/setup")
async def setup(req: SetupRequest):
    game.num_opponents    = max(1, min(4, req.num_opponents))
    game.model            = req.model
    game.use_llm_for_bots = req.use_llm_for_bots
    game.starting_stack   = max(100, min(10000, req.starting_stack))
    game.session_start_stack = game.starting_stack
    log.info(f"[SETUP] opp={game.num_opponents} model={game.model} stack={game.starting_stack}")
    return {"ok": True}


@app.post("/api/deal")
async def deal():
    log.info("[API] /deal")
    return game.start_hand()


@app.get("/api/state")
async def get_state():
    return game.to_dict()


@app.post("/api/action")
async def player_action(req: ActionRequest):
    if game.phase != 'playing':
        raise HTTPException(400, "Not in playing phase")
    cur = game.players[game.current_player_idx]
    if not cur['is_human']:
        raise HTTPException(400, "Not your turn")

    log.info(f"[API] /action human: {req.action} {req.amount or ''}")

    # capture context BEFORE action
    eq_info      = game.get_equity_info(game.current_player_idx)
    hole_cards   = list(cur['cards'])
    community    = game.community[:community_count_for_street(game.street)]
    street       = game.street
    pot          = game.pot
    to_call      = game.to_call
    stack        = cur['stack']
    history      = list(game.action_history)
    num_opp      = len([p for p in game.players if not p['is_human'] and not p['folded']])

    raise_amount = game.apply_action(req.action, req.amount)
    game.advance()

    # Poll for MC completion before coaching — MC usually finishes in ~30ms.
    # Cap wait at 200ms so a slow simulation never blocks the response.
    mc_run_id = eq_info.get('mc_run_id')
    if mc_run_id:
        deadline = asyncio.get_event_loop().time() + 0.20
        while asyncio.get_event_loop().time() < deadline:
            if get_mc_result(mc_run_id)['done']:
                break
            await asyncio.sleep(0.015)

    mc_result    = get_mc_result(mc_run_id) if mc_run_id else {}
    mc_equity_pct = mc_result['equity'] if mc_result.get('done') and mc_result.get('equity') is not None else None

    coaching = await asyncio.get_event_loop().run_in_executor(
        None, get_coaching, {
            'hole_cards':    hole_cards,
            'community':     community,
            'street':        street,
            'pot':           pot,
            'to_call':       to_call,
            'stack':         stack,
            'action':        req.action,
            'raise_amount':  raise_amount,
            'num_opponents': num_opp,
            'equity_pct':    eq_info['equity_pct'],
            'mc_equity_pct': mc_equity_pct,
            'req_equity':    eq_info['req_equity'],
            'hand_desc':     eq_info.get('hand_desc','—'),
            'draws':         eq_info.get('draws',[]),
            'action_history': history,
            'model':         game.model,
        }
    )

    game.coaching_history.append({
        'street':        street,
        'action':        req.action,
        'coaching':      coaching,
        'equity_pct':    eq_info['equity_pct'],
        'mc_equity_pct': mc_equity_pct,
        'req_equity':    eq_info['req_equity'],
    })
    return {'state': game.to_dict(), 'coaching': coaching, 'equity_info': eq_info}


@app.post("/api/bot-action")
async def bot_action():
    if game.phase != 'playing':
        return {'state': game.to_dict(), 'done': True}

    cur = game.players[game.current_player_idx]
    if cur['is_human'] or cur['folded']:
        game.advance()
        done = game.phase == 'showdown' or (
            game.phase == 'playing' and game.players[game.current_player_idx]['is_human']
        )
        return {'state': game.to_dict(), 'done': done}

    community = game.community[:community_count_for_street(game.street)]
    cur_idx   = game.current_player_idx
    num_players = len(game.players)

    # Position category: distance from dealer (BTN=0, SB=1, BB=2, ...)
    pos_offset = (cur_idx - game.dealer_idx) % num_players
    if pos_offset == 0:
        position_cat = 'late'   # button
    elif pos_offset in (1, 2):
        position_cat = 'blind'  # SB / BB
    elif num_players <= 3 or pos_offset == num_players - 1:
        position_cat = 'late'   # cutoff in larger games
    elif pos_offset <= 2:
        position_cat = 'early'  # UTG
    else:
        position_cat = 'mid'

    # Instant equity for this bot (no MC thread)
    num_opp = max(1, len([p for p in game.players
                          if p['id'] != cur['id'] and not p['folded']]))
    if game.street == 'preflop':
        bot_equity_pct = round(preflop_equity_known(cur['cards'], num_opp) * 100)
    else:
        bot_equity_pct = round(postflop_equity_outs(cur['cards'], community, game.street) * 100)
    req_equity_bot = pot_odds_needed(game.to_call, game.pot)

    human_tendencies = game.get_human_tendencies()

    result = await asyncio.get_event_loop().run_in_executor(
        None, get_bot_action,
        cur['personality'],
        {
            'hole_cards':       cur['cards'],
            'community':        community,
            'street':           game.street,
            'pot':              game.pot,
            'to_call':          game.to_call,
            'stack':            cur['stack'],
            'action_history':   game.action_history[-5:],
            'position':         position_cat,
            'equity_pct':       bot_equity_pct,
            'req_equity':       req_equity_bot,
            'human_tendencies': human_tendencies,
        },
        game.model,
        game.use_llm_for_bots,
    )

    game.players[game.current_player_idx]['last_thought'] = result.get('thought','')
    game.apply_action(result['action'], result.get('amount'))
    game.advance()

    done = game.phase == 'showdown' or (
        game.phase == 'playing' and game.players[game.current_player_idx]['is_human']
    )
    return {'state': game.to_dict(), 'done': done}


@app.get("/api/equity")
async def equity():
    """
    Returns instant outs-based equity AND fires MC in background.
    Response includes mc_run_id for polling.
    """
    human = next((p for p in game.players if p['is_human']), None)
    if not human or game.phase != 'playing':
        return {'equity_pct': None, 'mc_run_id': None}

    board   = game.community[:community_count_for_street(game.street)]
    num_opp = max(1, len([p for p in game.players if not p['is_human'] and not p['folded']]))
    eq      = get_equity_instant(human['cards'], board, game.street, num_opp)
    req     = pot_odds_needed(game.to_call, game.pot)

    return {
        **eq,
        'req_equity':  req,
        'profitable':  eq['equity_pct'] >= req if game.to_call > 0 else True,
    }


@app.get("/api/mc-equity")
async def mc_equity(run_id: str):
    """
    Poll for background MC result.
    Returns {'equity':N, 'done':bool, 'ms':N}
    Frontend calls this every 300ms until done=True.
    """
    result = get_mc_result(run_id)
    log.debug(f"[MC POLL] run_id={run_id} done={result['done']} equity={result['equity']}")
    return result


@app.get("/api/hand-lesson")
async def hand_lesson():
    """Return a one-sentence lesson from this hand's coaching history."""
    decisions = list(game.coaching_history)
    lesson = await asyncio.get_event_loop().run_in_executor(
        None, get_hand_lesson, decisions, game.model
    )
    return {'lesson': lesson}


@app.get("/api/hint")
async def hint():
    """Return one Socratic hint sentence for the current human decision."""
    human = next((p for p in game.players if p['is_human']), None)
    if not human or game.phase != 'playing':
        raise HTTPException(400, "Not in a decision")

    board    = game.community[:community_count_for_street(game.street)]
    num_opp  = max(1, len([p for p in game.players if not p['is_human'] and not p['folded']]))
    eq       = get_equity_instant(human['cards'], board, game.street, num_opp)
    mc_runid = eq.get('mc_run_id', '')
    mc       = get_mc_result(mc_runid) if mc_runid else {}

    hint_text = await asyncio.get_event_loop().run_in_executor(
        None, get_hint, {
            'hole_cards':    human['cards'],
            'community':     board,
            'street':        game.street,
            'pot':           game.pot,
            'to_call':       game.to_call,
            'stack':         human['stack'],
            'num_opponents': num_opp,
            'equity_pct':    eq.get('equity_pct'),
            'mc_equity_pct': mc.get('equity') if mc.get('done') else None,
            'req_equity':    pot_odds_needed(game.to_call, game.pot),
            'hand_desc':     eq.get('hand_desc', '—'),
            'action_history': list(game.action_history),
            'model':         game.model,
        }
    )
    return {'hint': hint_text}


@app.get("/api/session-summary")
async def session_summary():
    """Return a plain-text session summary as a downloadable .txt file."""
    from fastapi.responses import PlainTextResponse

    # Build hand list: all archived hands + current hand if it ended
    hands = list(game.session_hands)
    if game.phase == 'showdown':
        human = next((p for p in game.players if p['is_human']), None)
        if human:
            hands.append({
                'hand_number': game.hand_number,
                'hole_cards':  human['cards'],
                'community':   [c for c in game.community if c],
                'won':         human['id'] in game.winners,
                'chip_delta':  human['stack'] - game._hand_start_stack,
                'decisions':   list(game.coaching_history),
                'log':         list(game.log),
            })

    all_decisions = []
    for h in hands:
        for d in h['decisions']:
            all_decisions.append({**d, 'hand_number': h['hand_number'],
                                   'hole_cards': h['hole_cards'], 'community': h['community']})

    # Generate LLM coaching advice in background thread (falls back if Ollama is down)
    advice = await asyncio.get_event_loop().run_in_executor(
        None, _generate_session_advice, all_decisions, game.model
    )

    text = _build_summary_text(hands, game.session_start_stack, all_decisions, advice)
    return PlainTextResponse(
        content=text,
        headers={'Content-Disposition': 'attachment; filename="poker_session.txt"'},
    )


# ── session summary helpers ────────────────────────────────────────────────────

def _fmt_cards(cards: list) -> str:
    sym = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    out = []
    for c in cards:
        if c:
            rank = '10' if c[:-1] == 'T' else c[:-1]
            out.append(f"{rank}{sym.get(c[-1], c[-1])}")
    return ' '.join(out) if out else '—'


def _generate_session_advice(all_decisions: list, model: str) -> str:
    """Ask the LLM for personalized end-of-session coaching. Falls back to rule-based."""
    from .ollama_service import _ollama_call

    total   = len(all_decisions)
    optimal = sum(1 for d in all_decisions if d['coaching']['verdict'] == 'optimal')
    accept  = sum(1 for d in all_decisions if d['coaching']['verdict'] == 'acceptable')
    mistake = sum(1 for d in all_decisions if d['coaching']['verdict'] == 'mistake')
    mistakes = [d for d in all_decisions if d['coaching']['verdict'] == 'mistake']

    if not total:
        return "No decisions were recorded this session — play a few hands to get personalized advice."

    mistake_lines = []
    for i, d in enumerate(mistakes[:5], 1):
        board = [c for c in d['community'] if c]
        board_str = _fmt_cards(board) if board else "no board (preflop)"
        eq  = d['coaching'].get('equity', '?')
        req = d.get('req_equity', '?')
        why = d['coaching'].get('why', '').strip()
        mistake_lines.append(
            f"  {i}. Hand #{d['hand_number']}, {d['street']}, you chose to {d['action']}\n"
            f"     Cards: {_fmt_cards(d['hole_cards'])}  Board: {board_str}\n"
            f"     Equity {eq}%  (needed {req}% to break even)\n"
            f"     Coach note: {why}"
        )

    prompt = (
        f"Session stats: {len(all_decisions)} decisions — "
        f"{optimal} optimal, {accept} acceptable, {mistake} mistake(s).\n\n"
        + (f"Mistakes this session:\n" + "\n\n".join(mistake_lines) + "\n\n" if mistake_lines
           else "No mistakes — clean session.\n\n")
        + "Write 2-3 short paragraphs of honest, specific, actionable coaching for this player. "
        "Be direct and encouraging. Reference the actual hands above where relevant. "
        "Use plain English — no jargon like 'GTO', 'range', or 'polarized'. "
        "End with one concrete thing to focus on next session. "
        "Return ONLY the coaching text — no headers, no JSON, no preamble."
    )

    system = (
        "You are a friendly poker coach writing a short end-of-session review for a beginner. "
        "Be warm, honest, and specific. No jargon."
    )

    try:
        raw, _ = _ollama_call(system, prompt, model)
        # Strip any accidental JSON wrappers
        text = raw.strip().strip('`').strip()
        if text.startswith('{'):
            import re
            m = re.search(r'"[^"]{30,}"', text)
            text = m.group(0).strip('"') if m else text
        return text
    except Exception as e:
        log.warning(f"[SESSION ADVICE LLM FAILED] {e} — using rule-based fallback")
        return _fallback_advice(all_decisions)


def _fallback_advice(all_decisions: list) -> str:
    mistakes = [d for d in all_decisions if d['coaching']['verdict'] == 'mistake']
    n = len(mistakes)
    if not all_decisions:
        return "Play some hands to get advice."
    if n == 0:
        return "No major leaks this session — you made solid decisions throughout. Keep applying the same discipline next time."

    preflop = sum(1 for d in mistakes if d['street'] == 'preflop')
    calls   = sum(1 for d in mistakes if d['action'] == 'call')
    folds   = sum(1 for d in mistakes if d['action'] == 'fold')
    raises  = sum(1 for d in mistakes if d['action'] in ('raise', 'bet'))
    t       = n * 0.55

    if preflop >= t:
        return (
            f"{preflop} of your {n} mistakes happened before the flop. "
            "The fix is straightforward: fold more weak hands preflop, especially offsuit low cards. "
            "For next session, try folding any hand where both cards are below a 7 unless you're in the big blind."
        )
    if calls >= t:
        return (
            f"{calls} of your {n} mistakes were calls when the numbers didn't add up. "
            "Before calling, check whether your equity beats the pot-odds threshold shown on screen. "
            "When it doesn't, folding is almost always right — chasing costs more than it wins."
        )
    if folds >= t:
        return (
            f"{folds} of your {n} mistakes were folds when you still held a meaningful edge. "
            "Trust the equity number — if you're well above the break-even threshold, a fold gives up expected profit. "
            "Next session, call (or raise) more confidently when your equity is clearly in your favour."
        )
    if raises >= t:
        return (
            f"{raises} of your {n} mistakes came from raising or betting in weak spots. "
            "Save aggression for hands where your equity is strong. "
            "Next session, ask yourself before each raise: am I likely ahead here, or am I just applying pressure?"
        )
    return (
        f"Your {n} mistakes were spread across different spots — no single glaring leak, but room to tighten up everywhere. "
        "The most reliable habit to build: always compare your equity to the pot-odds threshold before calling or raising. "
        "Next session, slow down on that one decision and the rest tends to follow."
    )


def _fmt_log(hand_log: list) -> list:
    """Format a hand's action log into indented plain-text lines."""
    out = []
    for entry in hand_log:
        t    = entry.get('type', '')
        text = entry.get('text', '')
        thought = entry.get('thought', '')
        if t == 'street':
            # section divider — skip the very first "Hand #N" banner (redundant in this context)
            if '━━━ Hand #' in text:
                continue
            out.append(f"\n  {text}")
        elif t == 'player':
            out.append(f"  > {text}")
        elif t == 'bot':
            line = f"  . {text}"
            if thought:
                line += f'  ("{thought}")'
            out.append(line)
        elif t == 'winner':
            out.append(f"  {text}")
        elif t == 'system':
            out.append(f"  [{text}]")
    return out


def _build_summary_text(hands: list, session_start_stack: int,
                         all_decisions: list, advice: str) -> str:
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

    total   = len(all_decisions)
    optimal = sum(1 for d in all_decisions if d['coaching']['verdict'] == 'optimal')
    accept  = sum(1 for d in all_decisions if d['coaching']['verdict'] == 'acceptable')
    mistake = sum(1 for d in all_decisions if d['coaching']['verdict'] == 'mistake')
    net       = sum(h['chip_delta'] for h in hands)
    end_stack = session_start_stack + net
    net_str   = f"+${net}" if net >= 0 else f"-${abs(net)}"

    W = 48
    lines = [
        "POKER TRAINER — SESSION SUMMARY",
        "=" * W,
        f"Generated: {now}",
        "",
        "OVERVIEW",
        "-" * W,
        f"Hands played:    {len(hands)}",
        f"Decisions made:  {total}",
        f"Net chips:       {net_str}  (started ${session_start_stack}, ended ${end_stack})",
        "",
    ]

    if total > 0:
        pct = lambda x: f"{round(x / total * 100)}%"
        lines += [
            "DECISION BREAKDOWN",
            "-" * W,
            f"  Optimal      {optimal:>3}   ({pct(optimal)})",
            f"  Acceptable   {accept:>3}   ({pct(accept)})",
            f"  Mistakes     {mistake:>3}   ({pct(mistake)})",
            "",
        ]

    # Biggest mistakes
    mistake_list = [d for d in all_decisions if d['coaching']['verdict'] == 'mistake']
    if mistake_list:
        lines += ["BIGGEST MISTAKES", "-" * W]
        for d in mistake_list[:3]:
            board     = [c for c in d['community'] if c]
            board_str = _fmt_cards(board) if board else "none (preflop)"
            eq        = d['coaching'].get('equity', '?')
            req       = d.get('req_equity', '?')
            why       = d['coaching'].get('why', '').strip()
            verdict   = d['coaching'].get('verdictText', 'Mistake')
            lines += [
                f"Hand #{d['hand_number']} · {d['street'].capitalize()} · You {d['action']}  [{verdict}]",
                f"  Cards: {_fmt_cards(d['hole_cards'])}   Board: {board_str}",
                f"  Equity: {eq}%   Needed to call: {req}%",
                f"  {why}",
                "",
            ]

    # Quick recap table
    lines += ["HAND RECAP", "-" * W]
    for h in hands:
        delta_str = f"+${h['chip_delta']}" if h['chip_delta'] >= 0 else f"-${abs(h['chip_delta'])}"
        result    = "Won " if h['won'] else "Lost"
        dec       = h['decisions']
        opt_h = sum(1 for d in dec if d['coaching']['verdict'] == 'optimal')
        acc_h = sum(1 for d in dec if d['coaching']['verdict'] == 'acceptable')
        mis_h = sum(1 for d in dec if d['coaching']['verdict'] == 'mistake')
        parts = []
        if opt_h: parts.append(f"{opt_h} optimal")
        if acc_h: parts.append(f"{acc_h} ok")
        if mis_h: parts.append(f"{mis_h} mistake{'s' if mis_h > 1 else ''}")
        dec_str = ', '.join(parts) if parts else "no decisions"
        lines.append(
            f"  Hand #{h['hand_number']:>2}  {_fmt_cards(h['hole_cards']):<12}  "
            f"{result} {delta_str:<8}  {dec_str}"
        )
    lines.append("")

    # Full hand histories
    lines += ["=" * W, "HAND HISTORIES", "=" * W, ""]
    for h in hands:
        delta_str = f"+${h['chip_delta']}" if h['chip_delta'] >= 0 else f"-${abs(h['chip_delta'])}"
        result    = "Won" if h['won'] else "Lost"
        board     = _fmt_cards([c for c in h['community'] if c]) or "—"

        lines += [
            f"HAND #{h['hand_number']}  |  Your cards: {_fmt_cards(h['hole_cards'])}  "
            f"|  Board: {board}  |  {result} {delta_str}",
            "-" * W,
        ]

        # Action log
        if h.get('log'):
            lines += _fmt_log(h['log'])
        else:
            lines.append("  (no action log recorded)")

        # Decisions with coaching
        if h['decisions']:
            lines += ["", "  YOUR DECISIONS:"]
            for d in h['decisions']:
                v       = d['coaching']['verdict']
                vtext   = d['coaching'].get('verdictText', v)
                marker  = '✓' if v == 'optimal' else '~' if v == 'acceptable' else '✗'
                eq      = d['coaching'].get('equity', '?')
                req     = d.get('req_equity', '?')
                why     = d['coaching'].get('why', '').strip()
                optimal_play = d['coaching'].get('optimal', '').strip()
                lines += [
                    f"  {marker} {d['street'].capitalize()} → {d['action']}   "
                    f"[{vtext}]  equity {eq}%  (needed {req}%)",
                    f"    Why: {why}",
                ]
                if optimal_play and v != 'optimal':
                    lines.append(f"    Better: {optimal_play}")
        lines += ["", ""]

    # LLM coaching assessment
    lines += [
        "=" * W,
        "COACH'S ASSESSMENT",
        "=" * W,
        "",
        advice,
        "",
    ]

    return "\n".join(lines)
