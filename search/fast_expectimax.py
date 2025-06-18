# search/fast_expectimax.py
"""
Moteur « roll-out » très rapide (Numba parallèle).

Idée :
    – pour chaque direction possible, on simule k roll-outs aléatoires sur
      depth d et on garde la moyenne d’un score simple (nombre de cases vides).
    – pas de table de transpo, pas de nœuds CHANCE explicites.
En pratique k = 64, depth = 3 → < 1 ms sur un CPU mobile.
"""

from typing import Tuple
import numpy as np
import numba as nb
from board import move_board, can_move

# ──────────────────────────────────────────────────────────────────────────
@nb.njit(inline="always")
def _add_random_tile(b: nb.uint64) -> nb.uint64:
    """Ajoute 2 (90 %) ou 4 (10 %) sur une case vide aléatoire (Numba)."""
    empties = []
    for pos in range(16):
        if ((b >> (pos * 4)) & 0xF) == 0:
            empties.append(pos)
    if not empties:
        return b
    sel = empties[np.random.randint(len(empties))]
    val = 1 if np.random.random() < 0.9 else 2
    return b | (nb.uint64(val) << (sel * 4))

# ──────────────────────────────────────────────────────────────────────────
@nb.njit(parallel=True, fastmath=True)
def _rollout_value(b_raw: nb.uint64, depth: int, k: int) -> float:
    """Valeur moyenne de k roll-outs aléatoires (score = cases vides)."""
    total = 0.0
    for r in nb.prange(k):
        b = b_raw
        for d in range(depth):
            dir_id = np.random.randint(4)         # 0←1→2↑3↓
            b2, _, moved = move_board(b, dir_id)
            if not moved and not can_move(b):
                break
            if moved:
                b = _add_random_tile(b2)
        # score : nombre de cases vides
        empties = 0
        for pos in range(16):
            if ((b >> (pos * 4)) & 0xF) == 0:
                empties += 1
        total += empties
    return total / k

# ──────────────────────────────────────────────────────────────────────────
def fast_best_move(board, depth: int = 3, k: int = 64) -> str:
    """
    Choisit la direction via roll-out moyen (très rapide, qualité correcte).
    """
    DIRECTIONS = ["left", "right", "up", "down"]
    best_dir, best_val = "up", -1e9
    for dir_id, dir_str in enumerate(DIRECTIONS):
        tmp = board.clone()
        moved, _ = tmp.move(dir_str, add_random=False)
        if not moved:
            continue
        val = _rollout_value(np.uint64(tmp.raw), depth - 1, k)
        if val > best_val:
            best_val, best_dir = val, dir_str
    return best_dir
