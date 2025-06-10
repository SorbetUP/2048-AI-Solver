# simulate.py
import numpy as np, numba as nb
from board import move_board, can_move

@nb.njit(parallel=True, fastmath=True, cache=True)
def run_simulation(n_games: int, max_steps: int = 10_000) -> np.ndarray:
    scores = np.zeros(n_games, dtype=np.int32)
    boards = np.zeros(n_games, dtype=np.uint64)

    # init : deux tuiles par plateau
    for i in nb.prange(n_games):
        b = np.uint64(0)
        # on place 2× la tuile « 2 » (val=1) sur deux positions aléatoires distinctes
        p1 = np.random.randint(16)
        p2 = (p1 + np.random.randint(1, 15)) % 16
        b |= np.uint64(1) << (p1 * 4)
        b |= np.uint64(1) << (p2 * 4)
        boards[i] = b

    # partie
    for i in nb.prange(n_games):
        b = boards[i]
        score = 0
        for _ in range(max_steps):
            dir_id = np.random.randint(4)
            b2, gain, moved = move_board(b, dir_id)
            if not moved:
                if not can_move(b):
                    break
                continue
            score += gain
            b = b2
        scores[i] = score
    return scores
