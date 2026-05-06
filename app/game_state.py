"""
game_state.py
=============
Owns the entire hand lifecycle. Handles:
  - dealing, blinds, betting rounds
  - action validation and pot math
  - street advancement with proper acted/bet checks
  - infinite loop protection (raise cap, safety counter)
  - showdown and winner determination
"""

import random
import uuid
import logging
from .poker_math import (
    make_deck, card_to_display, evaluate_hand, get_equity_instant,
    pot_odds_needed, determine_winner, find_best_five, community_count_for_street,
)
from .ollama_service import BOT_PERSONALITIES

log = logging.getLogger("poker.game")

STARTING_STACK = 500
BIG_BLIND      = 10
SMALL_BLIND    = 5
STREETS        = ['preflop', 'flop', 'turn', 'river']
MAX_RAISES     = 4  # cap per street to prevent infinite loops


def make_player(idx: int, is_human: bool, pkey: str, stack: int = STARTING_STACK) -> dict:
    p = BOT_PERSONALITIES[pkey]
    return {
        'id':                   idx,
        'name':                 'You' if is_human else f"{p['label'].split('-')[0].strip()} {idx}",
        'is_human':             is_human,
        'personality':          pkey,
        'cards':                [],
        'stack':                stack,
        'current_bet':          0,
        'total_bet_this_hand':  0,
        'folded':               False,
        'all_in':               False,
        'last_action':          None,
        'last_thought':         None,
        'has_acted_this_street': False,
    }


def next_active_idx(players: list, from_idx: int) -> int:
    """Find next non-folded, non-all-in player after from_idx."""
    n = len(players)
    for i in range(1, n + 1):
        idx = (from_idx + i) % n
        if not players[idx]['folded'] and not players[idx]['all_in']:
            return idx
    return -1


def street_is_over(players: list, to_call: int) -> bool:
    """
    Street ends when ALL active players have:
      1. acted at least once this street
      2. matched the current to_call amount
    Special case: if only 0 or 1 active players remain, street is also over.
    """
    active = [p for p in players if not p['folded'] and not p['all_in']]
    if len(active) <= 1:
        log.debug(f"street_is_over: only {len(active)} active players → True")
        return True
    result = all(p['has_acted_this_street'] and p['current_bet'] >= to_call for p in active)
    if result:
        log.debug(f"street_is_over: all {len(active)} players acted and bets matched → True")
    return result


def cap_raise(requested: int, stack: int, to_call: int, current_bet: int) -> int:
    """Clamp a raise to what the player can actually put in."""
    max_total = current_bet + stack
    return max(min(requested, max_total), to_call + 1)


class GameState:
    def __init__(self):
        self.game_id            = str(uuid.uuid4())
        self.phase              = 'setup'
        self.street             = 'preflop'
        self.players            = []
        self.community          = []
        self.deck               = []
        self.pot                = 0
        self.to_call            = BIG_BLIND
        self.min_raise          = BIG_BLIND * 2
        self.current_player_idx = 0
        self.dealer_idx         = 0
        self.action_history     = []
        self.log                = []
        self.hand_number        = 0
        self.winners            = []
        self.coaching_history   = []
        self.num_opponents      = 2
        self.model              = 'gemma3:4b'
        self.use_llm_for_bots   = False
        self._raises_this_street = 0
        # configurable starting stack (set via /api/setup)
        self.starting_stack      = STARTING_STACK
        # session-level tracking (persists across hands)
        self.session_start_stack = STARTING_STACK
        self.session_hands       = []   # one record per completed hand
        self._hand_start_stack   = STARTING_STACK
        # per-bot behavioural stats keyed by player_id (seat)
        self.bot_stats: dict     = {}   # {id: {raises,calls,checks,folds,total,hands}}
        self.win_reason: str     = ''

    # ── start hand ──────────────────────────────────────────────────────────
    def start_hand(self) -> dict:
        total = self.num_opponents + 1
        pool  = random.sample(list(BOT_PERSONALITIES.keys()) * 4, total)

        # carry stacks from previous hand
        old_stacks = {p['id']: p['stack'] for p in self.players}

        # archive completed hand into session history before resetting
        if self.hand_number > 0:
            human_stack_end = old_stacks.get(0, self._hand_start_stack)
            human_cards     = next((p['cards'] for p in self.players if p['is_human']), [])
            self.session_hands.append({
                'hand_number': self.hand_number,
                'hole_cards':  human_cards,
                'community':   [c for c in self.community if c],
                'won':         0 in self.winners,
                'chip_delta':  human_stack_end - self._hand_start_stack,
                'decisions':   list(self.coaching_history),
                'log':         list(self.log),
            })

        # record human stack at start of this new hand
        self._hand_start_stack = old_stacks.get(0, self.starting_stack)
        self.players = []
        for i in range(total):
            is_human = (i == 0)
            pkey     = 'tight' if is_human else pool[i]
            stack    = old_stacks.get(i, self.starting_stack)
            if stack <= 0: stack = self.starting_stack
            self.players.append(make_player(i, is_human, pkey, stack))

        # initialise / increment bot stat slots for this hand
        for p in self.players:
            if not p['is_human']:
                s = self.bot_stats.setdefault(
                    p['id'], {'raises': 0, 'calls': 0, 'checks': 0, 'folds': 0, 'total': 0, 'hands': 0}
                )
                s['hands'] += 1

        # deal cards
        self.deck = make_deck()
        idx = 0
        for p in self.players:
            p['cards'] = [self.deck[idx], self.deck[idx+1]]
            idx += 2
        self.community = self.deck[idx:idx+5]

        # post blinds
        sb_idx, bb_idx = 1 % total, 2 % total
        for i, amt in [(sb_idx, SMALL_BLIND), (bb_idx, BIG_BLIND)]:
            self.players[i]['current_bet']         = amt
            self.players[i]['stack']               -= amt
            self.players[i]['total_bet_this_hand']  = amt

        self.pot                  = SMALL_BLIND + BIG_BLIND
        self.to_call              = BIG_BLIND
        self.min_raise            = BIG_BLIND * 2
        self.street               = 'preflop'
        self.current_player_idx   = (bb_idx + 1) % total
        self.dealer_idx           = (self.dealer_idx + 1) % total
        self.action_history       = []
        self.hand_number         += 1
        self.winners              = []
        self.coaching_history     = []
        self.phase                = 'playing'
        self._raises_this_street  = 0

        human_cards = self.players[0]['cards']
        bot_names   = [f"{p['name']}({BOT_PERSONALITIES[p['personality']]['name']})" for p in self.players if not p['is_human']]
        self.log = [
            {'text': f'━━━ Hand #{self.hand_number} ━━━', 'type': 'street', 'thought': ''},
            {'text': f"Dealing to {total} players. Bots: {', '.join(bot_names)}", 'type': 'system', 'thought': ''},
            {'text': f'SB posts ${SMALL_BLIND}, BB posts ${BIG_BLIND} | Pot: ${self.pot}', 'type': 'system', 'thought': ''},
        ]

        log.info(
            f"\n{'═'*60}\n"
            f"HAND #{self.hand_number} | {total} players\n"
            f"Human cards: {human_cards}\n"
            f"Bots: {bot_names}\n"
            f"First to act: player idx {self.current_player_idx}\n"
            f"{'═'*60}"
        )
        return self.to_dict()

    # ── apply action ────────────────────────────────────────────────────────
    def apply_action(self, action: str, amount: int = None):
        """
        Apply an action for the current player.
        Returns raise_amount or None.
        Validates and adjusts invalid actions rather than raising exceptions.
        """
        if self.phase != 'playing':
            log.warning("apply_action called outside playing phase")
            return None

        p            = self.players[self.current_player_idx]
        raise_amount = None
        label        = 'HUMAN' if p['is_human'] else f'BOT({p["personality"]})'

        log.info(
            f"[ACTION] {label} {p['name']} → {action}"
            + (f" ${amount}" if amount else "")
            + f" | stack=${p['stack']} pot=${self.pot} to_call=${self.to_call}"
        )

        # ── fold ──────────────────────────────────────────────────────────
        if action == 'fold':
            p.update({'folded': True, 'last_action': 'fold', 'has_acted_this_street': True})
            self._log_action(p, f"{p['name']} folds")

        # ── check ─────────────────────────────────────────────────────────
        elif action == 'check':
            if self.to_call > p['current_bet']:
                log.debug(f"check → redirecting to call (to_call=${self.to_call} > current_bet=${p['current_bet']})")
                return self.apply_action('call', None)
            p.update({'last_action': 'check', 'has_acted_this_street': True})
            self._log_action(p, f"{p['name']} checks")

        # ── call ──────────────────────────────────────────────────────────
        elif action == 'call':
            needed   = self.to_call - p['current_bet']
            call_amt = min(needed, p['stack'])
            if call_amt <= 0:
                log.debug("call with 0 amount → redirecting to check")
                return self.apply_action('check', None)
            p['stack']               -= call_amt
            p['current_bet']         += call_amt
            p['total_bet_this_hand'] += call_amt
            p['last_action']          = 'call'
            p['has_acted_this_street'] = True
            if p['stack'] == 0:
                p['all_in'] = True
                log.info(f"[ALL-IN] {p['name']} is all in")
            self.pot += call_amt
            self._log_action(p, f"{p['name']} calls ${call_amt} | pot now ${self.pot}")

        # ── raise / bet ───────────────────────────────────────────────────
        elif action in ('raise', 'bet'):
            if self._raises_this_street >= MAX_RAISES:
                log.warning(f"[RAISE CAP] {p['name']} raise cap reached ({MAX_RAISES}), converting to call")
                return self.apply_action('call', None)

            raw_total        = cap_raise(amount or self.min_raise, p['stack'], self.to_call, p['current_bet'])
            chips_in         = min(raw_total - p['current_bet'], p['stack'])
            if chips_in <= 0:
                return self.apply_action('check', None)

            p['stack']               -= chips_in
            p['current_bet']         += chips_in
            p['total_bet_this_hand'] += chips_in
            p['last_action']          = action
            p['has_acted_this_street'] = True
            if p['stack'] == 0:
                p['all_in'] = True

            self.pot         += chips_in
            raise_size        = p['current_bet'] - self.to_call
            self.to_call      = p['current_bet']
            self.min_raise    = max(BIG_BLIND, raise_size)
            self._raises_this_street += 1
            raise_amount = chips_in

            # reopen action for everyone else
            for i, op in enumerate(self.players):
                if i != self.current_player_idx and not op['folded'] and not op['all_in']:
                    op['has_acted_this_street'] = False

            self._log_action(p, f"{p['name']} {action}s to ${self.to_call} | pot now ${self.pot}")
            log.info(f"[RAISE #{self._raises_this_street}] new to_call=${self.to_call} min_raise=${self.min_raise}")

        # ── bot stat tracking (observation only — no effect on game logic) ──
        if not p['is_human']:
            s = self.bot_stats.setdefault(
                p['id'], {'raises': 0, 'calls': 0, 'checks': 0, 'folds': 0, 'total': 0, 'hands': 0}
            )
            key = {'fold': 'folds', 'call': 'calls', 'check': 'checks',
                   'raise': 'raises', 'bet': 'raises'}.get(action)
            if key:
                s[key] += 1
                s['total'] += 1

        self.action_history.append(self.log[-1]['text'])
        return raise_amount

    # ── advance ─────────────────────────────────────────────────────────────
    def advance(self) -> dict:
        """
        After an action, determine what happens next:
          1. Only one active player → end hand (others folded)
          2. Street is over → advance to next street or showdown
          3. Otherwise → move to next active player
        """
        active = [p for p in self.players if not p['folded']]

        if len(active) <= 1:
            log.info(f"[ADVANCE] Only {len(active)} player(s) remain → ending hand")
            self._end_hand()
            return self.to_dict()

        if street_is_over(self.players, self.to_call):
            next_idx = STREETS.index(self.street) + 1
            if next_idx >= len(STREETS):
                log.info("[ADVANCE] River complete → showdown")
                self._end_hand()
                return self.to_dict()

            next_street = STREETS[next_idx]
            log.info(f"[ADVANCE] Street {self.street} → {next_street}")

            for p in self.players:
                if not p['folded']:
                    p['current_bet']          = 0
                    p['has_acted_this_street'] = False
                    p['last_action']           = None

            self.street               = next_street
            self.to_call              = 0
            self.min_raise            = BIG_BLIND
            self._raises_this_street  = 0

            board_cards = self.community[:community_count_for_street(next_street)]
            self.log.append({'text': f'━━━ {next_street.upper()} ━━━ | Board: {" ".join(board_cards)}', 'type': 'street', 'thought': ''})
            self.action_history.append(f'[{next_street}]')

            first = next((i for i, p in enumerate(self.players) if not p['folded'] and not p['all_in']), None)
            if first is None:
                self._end_hand()
                return self.to_dict()

            self.current_player_idx = first
            log.info(f"[ADVANCE] First to act on {next_street}: {self.players[first]['name']}")
            return self.to_dict()

        # next player
        next_idx = next_active_idx(self.players, self.current_player_idx)

        if next_idx == -1 or next_idx == self.current_player_idx:
            # safety: if stuck, check if everyone has actually acted
            unacted = [p for p in self.players if not p['folded'] and not p['all_in'] and not p['has_acted_this_street']]
            if not unacted:
                log.warning("[ADVANCE] All players acted but street_is_over returned False — forcing advance")
                return self.advance()
            log.error(f"[ADVANCE] Stuck at player {self.current_player_idx} — ending hand as safety")
            self._end_hand()
            return self.to_dict()

        self.current_player_idx = next_idx
        log.debug(f"[ADVANCE] Next player: {self.players[next_idx]['name']} (idx={next_idx})")
        return self.to_dict()

    # ── end hand ────────────────────────────────────────────────────────────
    def _end_hand(self):
        active  = [p for p in self.players if not p['folded']]
        human   = next((p for p in self.players if p['is_human']), None)
        full_board = self.community[:5]

        if len(active) == 1:
            # Hand ended by fold — no card evaluation needed
            winner_ids = [active[0]['id']]
            winner = active[0]
            if human and winner['is_human']:
                self.win_reason = "You win — everyone folded!"
            elif human:
                self.win_reason = f"{winner['name']} wins — you folded"
            else:
                self.win_reason = f"{winner['name']} wins"
        else:
            try:
                winner_ids = determine_winner(active, full_board)
            except Exception as e:
                log.error(f"[SHOWDOWN ERROR] {e} — awarding to first active player")
                winner_ids = [active[0]['id']]

            # Find best 5 cards + hand description for every non-folded player
            for p in active:
                try:
                    p['best_five'], p['hand_desc'] = find_best_five(p['cards'], full_board)
                except Exception as e:
                    log.warning(f"find_best_five failed for {p['name']}: {e}")
                    p['best_five'], p['hand_desc'] = [], '—'

            # Build win_reason line
            winner = next((p for p in self.players if p['id'] in winner_ids), None)
            if len(winner_ids) > 1:
                desc = winner.get('hand_desc', 'best hand') if winner else 'best hand'
                self.win_reason = f"Split pot — both have {desc}"
            elif winner and human:
                if winner['is_human']:
                    self.win_reason = f"You win with {winner.get('hand_desc', 'the best hand')}!"
                else:
                    w_desc = winner.get('hand_desc', '?')
                    h_desc = human.get('hand_desc', '?') if not human.get('folded') else 'folded'
                    self.win_reason = f"{winner['name']} wins — {w_desc} beats your {h_desc}"
            else:
                self.win_reason = ''

        share = self.pot // max(len(winner_ids), 1)
        self.winners = winner_ids
        for p in self.players:
            if p['id'] in winner_ids:
                p['stack'] += share

        winner_names = ' & '.join(p['name'] for p in self.players if p['id'] in winner_ids)
        self.log.append({'text': f'🏆 {winner_names} wins ${self.pot}', 'type': 'winner', 'thought': ''})
        self.phase = 'showdown'

        log.info(
            f"\n{'═'*60}\n"
            f"HAND #{self.hand_number} COMPLETE\n"
            f"Winner(s): {winner_names} | Pot: ${self.pot}\n"
            f"Win reason: {self.win_reason}\n"
            f"Final stacks: {[(p['name'], p['stack']) for p in self.players]}\n"
            f"{'═'*60}"
        )

    # ── human tendency tracker ──────────────────────────────────────────────
    def get_human_tendencies(self) -> dict:
        """Compute human fold/call/raise frequency from archived session hands."""
        decisions = []
        for h in self.session_hands:
            decisions.extend(h.get('decisions', []))

        total  = len(decisions)
        if total == 0:
            return {'fold_pct': 50, 'call_pct': 50, 'raise_pct': 0, 'total': 0}

        folds  = sum(1 for d in decisions if d.get('action') == 'fold')
        calls  = sum(1 for d in decisions if d.get('action') == 'call')
        raises = sum(1 for d in decisions if d.get('action') in ('raise', 'bet'))
        return {
            'fold_pct':  round(folds  / total * 100),
            'call_pct':  round(calls  / total * 100),
            'raise_pct': round(raises / total * 100),
            'total':     total,
        }

    # ── equity info ─────────────────────────────────────────────────────────
    def get_equity_info(self, player_idx: int) -> dict:
        p         = self.players[player_idx]
        board     = self.community[:community_count_for_street(self.street)]
        num_opp   = max(1, len([x for x in self.players if not x['is_human'] and not x['folded']]))
        eq_data   = get_equity_instant(p['cards'], board, self.street, num_opp)
        req       = pot_odds_needed(self.to_call, self.pot)

        log.info(
            f"[EQUITY] {p['name']} | street={self.street} | "
            f"hand={eq_data['hand_desc']} | "
            f"outs_eq={eq_data['equity_pct']}% | mc_run={eq_data.get('mc_run_id','?')} | "
            f"needed={req}% | profitable={eq_data['equity_pct'] >= req}"
        )

        return {
            **eq_data,
            'req_equity':  req,
            'profitable':  eq_data['equity_pct'] >= req if self.to_call > 0 else True,
        }

    # ── log helper ──────────────────────────────────────────────────────────
    def _log_action(self, player: dict, text: str):
        self.log.append({
            'text':    text,
            'type':    'player' if player['is_human'] else 'bot',
            'thought': player.get('last_thought', '') or '',
        })

    # ── serialise ───────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        cc = community_count_for_street(self.street)
        community_display = [card_to_display(c) for c in self.community[:cc]]
        while len(community_display) < 5:
            community_display.append(None)

        return {
            'game_id':            self.game_id,
            'phase':              self.phase,
            'street':             self.street,
            'players':            [{**p, 'cards_display': [card_to_display(c) for c in p['cards']],
                                    'bot_stats': self.bot_stats.get(p['id']) if not p['is_human'] else None,
                                    'best_five': p.get('best_five', []),
                                    'hand_desc': p.get('hand_desc', '')}
                                   for p in self.players],
            'community':          community_display,
            'pot':                self.pot,
            'to_call':            self.to_call,
            'min_raise':          self.min_raise,
            'current_player_idx': self.current_player_idx,
            'action_history':     self.action_history,
            'log':                self.log[-50:],
            'hand_number':        self.hand_number,
            'dealer_idx':         self.dealer_idx,
            'winners':            self.winners,
            'coaching_history':   self.coaching_history,
            'num_opponents':      self.num_opponents,
            'model':              self.model,
            'win_reason':         self.win_reason,
        }
