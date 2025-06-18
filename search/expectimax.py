# search/expectimax.py
"""
Expectimax + B.E.P.P. (Bounded Expectation & Probability Pruning)

Nouveautés :
• set_bepp_params(prob_cutoff, beam_k) ⇒ modifie les bornes à chaud
"""

import time, random
from typing import Callable, Optional, Dict, Tuple, List

from board import Board
from eval.heuristics import bounded_eval

# ─────────────────────── paramètres B.E.P.P. (modifiables) ─────────────────
PROB_CUTOFF = 0.02   # θ : probabilité minimale développée à un nœud chance
BEAM_K      = 4      # nombre max de directions MAX gardées après tri
V_MIN, V_MAX = 0.0, 1.0   # domaine de bounded_eval (toujours 0-1)

def set_bepp_params(*, prob_cutoff: float | None = None,
                    beam_k: int | None = None) -> None:
    """Permet de changer θ et/ou k depuis un autre module (argparse)."""
    global PROB_CUTOFF, BEAM_K
    if prob_cutoff is not None:
        PROB_CUTOFF = max(0.0, min(1.0, float(prob_cutoff)))
    if beam_k is not None and beam_k >= 1:
        BEAM_K = int(beam_k)

# ────────────────────────────────────────────────────────────────────────────
DIRECTIONS = ["up", "down", "left", "right"]

def best_move(board: Board,
              depth: int,
              time_limit_ms: int,
              eval_fn: Optional[Callable[[Board], float]] = None
              ) -> str:
    """Choisit la meilleure direction avec BEPP + approfondissement itératif."""
    eval_fn  = eval_fn or bounded_eval
    deadline = time.time() + time_limit_ms / 1000.0
    tt: Dict[int, Tuple[int, float]] = {}

    best_dir, best_val = None, float("-inf")

    for d in range(1, depth + 1):
        if time.time() >= deadline:
            break

        # ---- BEAM tri rapide ---------------------------------------------
        moves: List[Tuple[float, str, Board]] = []
        for dir_ in DIRECTIONS:
            tmp = board.clone()
            if not tmp.move(dir_, add_random=False)[0]:
                continue
            moves.append((eval_fn(tmp), dir_, tmp))
        moves.sort(reverse=True, key=lambda t: t[0])
        moves = moves[:BEAM_K]

        for _, dir_, child in moves:
            val = _expectimax(child, d - 1, False,
                              -float("inf"), float("inf"),
                              eval_fn, tt, deadline)
            if val > best_val:
                best_val, best_dir = val, dir_

        if time.time() >= deadline:
            break

    return best_dir or "up"

# ───────────────────────────────── algorithme récursif ──────────────────────
def _expectimax(board: Board,
                depth: int,
                maximizing: bool,
                alpha: float,
                beta: float,
                eval_fn: Callable[[Board], float],
                tt: Dict[int, Tuple[int, float]],
                deadline: float) -> float:

    if time.time() >= deadline:
        return eval_fn(board)

    key = hash(board)
    if key in tt:
        saved_d, val = tt[key]
        if saved_d >= depth:
            return val

    # feuille ?
    if depth == 0 or not board.can_move():
        val = eval_fn(board)
        tt[key] = (depth, val)
        return val

    # ─────────── Max ───────────
    if maximizing:
        best = float("-inf")
        for dir_ in DIRECTIONS:
            tmp = board.clone()
            if not tmp.move(dir_, add_random=False)[0]:
                continue
            val = _expectimax(tmp, depth - 1, False,
                              alpha, beta,
                              eval_fn, tt, deadline)
            best = max(best, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        tt[key] = (depth, best)
        return best

    # ─────────── Chance ─────────
    running, p_seen = 0.0, 0.0
    empties = board.get_empty_cells()

    for (r, c) in empties:         # ordre séquentiel - reproductible
        for exp, prob in ((1, 0.9), (2, 0.1)):  # 2 avant 4
            if prob < PROB_CUTOFF:
                running += prob * eval_fn(board)
                p_seen  += prob
                continue

            tmp = board.clone()
            tmp.set_tile(r, c, exp)
            val = _expectimax(tmp, depth - 1, True,
                              alpha, beta,
                              eval_fn, tt, deadline)
            running += prob * val
            p_seen  += prob

            upper = running + (1 - p_seen) * V_MAX
            if upper < alpha:
                break   # on ne battra jamais α
        if upper < alpha:
            break

    expected = running / p_seen if p_seen else eval_fn(board)
    tt[key] = (depth, expected)
    return expected
