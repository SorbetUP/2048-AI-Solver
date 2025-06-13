import time
import random
from typing import Callable, Optional
from board import Board
from eval.heuristics import basic_eval  # à adapter selon le nom réel

DIRECTIONS = ["up", "down", "left", "right"]

def best_move(board: Board, depth: int, time_limit_ms: int, eval_fn: Optional[Callable] = None) -> str:
    start_time = time.time()
    time_limit = start_time + (time_limit_ms / 1000.0)
    eval_fn = eval_fn or basic_eval
    transpo_table = {}

    best = None
    best_score = float('-inf')

    max_depth = 1
    while time.time() < time_limit and max_depth <= depth:
        for direction in DIRECTIONS:
            temp = board.clone()
            if not temp.move(direction):
                continue

            score = expectimax(temp, depth=max_depth - 1, maximizing=False,
                               eval_fn=eval_fn, transpo_table=transpo_table,
                               time_limit=time_limit)

            if score > best_score:
                best_score = score
                best = direction

        max_depth += 1

    return best or "up"  # fallback


def expectimax(board: Board, depth: int, maximizing: bool, eval_fn: Callable,
               transpo_table: dict, time_limit: float) -> float:
    
    if time.time() > time_limit:
        return eval_fn(board)
    
    key = hash(board)
    if key in transpo_table:
        saved_depth, value = transpo_table[key]
        if saved_depth >= depth:
            return value

    if depth == 0 or board.is_game_over():
        val = eval_fn(board)
        transpo_table[key] = (depth, val)
        return val

    if maximizing:
        max_val = float('-inf')
        for direction in DIRECTIONS:
            temp = board.clone()
            if not temp.move(direction):
                continue
            val = expectimax(temp, depth - 1, False, eval_fn, transpo_table, time_limit)
            max_val = max(max_val, val)
        transpo_table[key] = (depth, max_val)
        return max_val

    else:  # chance node
        cells = board.get_empty_cells()
        if not cells:
            return eval_fn(board)

        total = 0
        count = 0
        sampled = random.sample(cells, min(4, len(cells)))

        for x, y in sampled:
            for value, prob in [(2, 0.9), (4, 0.1)]:
                if prob < 0.05:
                    continue
                temp = board.clone()
                temp.grid[x][y] = value
                val = expectimax(temp, depth - 1, True, eval_fn, transpo_table, time_limit)
                total += prob * val
            count += 1

        expected = total / count if count > 0 else 0
        transpo_table[key] = (depth, expected)
        return expected
