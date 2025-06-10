"""
random_play.py – moteur de benchmark 2048 « random policy »
Joue N parties indépendantes avec des directions aléatoires et
ajout d'une tuile (2 ou 4) après chaque coup valide.

Fonction publique : random_benchmark(n_games: int, max_steps: int = 10_000)
Retourne un tableau NumPy int32 des scores finaux.
"""
import numpy as np
import numba as nb
from board import move_board, can_move   # fonctions JIT déjà compilées

# ────────────────────────────────────────────────────────────────────────────
@nb.njit(parallel=True, fastmath=True, cache=True)
def random_benchmark(n_games: int, max_steps: int = 10_000) -> np.ndarray:
    scores = np.zeros(n_games, dtype=np.int32)

    for g in nb.prange(n_games):
        # --- initialisation : deux tuiles « 2 » aléatoires ---------------
        b = np.uint64(0)
        p1 = np.random.randint(16)
        p2 = (p1 + np.random.randint(1, 15)) % 16   # index distinct
        b |= np.uint64(1) << (p1 * 4)               # val=1 -> tuile 2
        b |= np.uint64(1) << (p2 * 4)

        score = 0

        for _ in range(max_steps):
            dir_id = np.random.randint(4)           # 0:← 1:→ 2:↑ 3:↓
            b2, gain, moved = move_board(b, dir_id)

            if not moved:
                if not can_move(b):
                    break          # plateau bloqué
                continue           # coup invalide mais partie pas finie

            b = b2
            score += gain

            # --- ajout d'une nouvelle tuile (90 % 2, 10 % 4) -------------
            # 1. décompte des cases vides
            empty_cnt = 0
            for pos in range(16):
                if ((b >> (pos * 4)) & 0xF) == 0:
                    empty_cnt += 1

            if empty_cnt == 0:
                break              # grille pleine

            # 2. choix aléatoire de la case à remplir
            pick = np.random.randint(empty_cnt)

            # 3. placement effectif
            idx = 0
            for pos in range(16):
                if ((b >> (pos * 4)) & 0xF) == 0:
                    if idx == pick:
                        val = 1 if np.random.random() < 0.9 else 2
                        b |= np.uint64(val) << (pos * 4)
                        break
                    idx += 1

        scores[g] = score

    return scores
