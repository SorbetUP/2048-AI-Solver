import time
import random
from typing import Callable, Optional
from board import Board
from eval.heuristics import basic_eval  # Heuristique par défaut si aucune n'est fournie

# Liste des directions possibles (les coups dans 2048)
DIRECTIONS = ["up", "down", "left", "right"]

def best_move(board: Board, depth: int, time_limit_ms: int, eval_fn: Optional[Callable] = None) -> str:
    """
    Fonction principale : retourne le meilleur coup à jouer sur une grille donnée
    en utilisant l'algorithme Expectimax avec approfondissement itératif.

    - board : état courant du jeu (classe Board)
    - depth : profondeur maximale d'exploration (sera augmenté progressivement)
    - time_limit_ms : temps alloué pour prendre une décision (en millisecondes)
    - eval_fn : fonction d’évaluation optionnelle (Victor ou heuristique simple)
    """
    start_time = time.time()
    time_limit = start_time + (time_limit_ms / 1000.0)
    eval_fn = eval_fn or basic_eval
    transpo_table = {}  # Table de transposition pour éviter les re-calculs

    best = None
    best_score = float('-inf')
    max_depth = 1

    # Approfondissement itératif : on tente des profondeurs croissantes jusqu'à épuiser le temps
    while time.time() < time_limit and max_depth <= depth:
        for direction in DIRECTIONS:
            temp = board.clone()
            if not temp.move(direction)[0]:  # ignore les coups illégaux
                continue

            score = expectimax(temp, depth=max_depth - 1, maximizing=False,
                               eval_fn=eval_fn, transpo_table=transpo_table,
                               time_limit=time_limit)

            if score > best_score:
                best_score = score
                best = direction

        max_depth += 1

    return best or "up"  # fallback si rien trouvé


def expectimax(board: Board, depth: int, maximizing: bool, eval_fn: Callable,
               transpo_table: dict, time_limit: float) -> float:
    """
    Fonction récursive de l’algorithme Expectimax :
    - Noeud joueur : max des valeurs des coups possibles
    - Noeud chance : somme pondérée des possibilités d’apparition (2 ou 4)
    """
    if time.time() > time_limit:
        return eval_fn(board)

    key = hash(board)
    if key in transpo_table:
        saved_depth, value = transpo_table[key]
        if saved_depth >= depth:
            return value

    if depth == 0 or not board.can_move():
        val = eval_fn(board)
        transpo_table[key] = (depth, val)
        return val

    if maximizing:
        # Le joueur choisit le meilleur coup
        max_val = float('-inf')
        for direction in DIRECTIONS:
            temp = board.clone()
            if not temp.move(direction)[0]:
                continue
            val = expectimax(temp, depth - 1, False, eval_fn, transpo_table, time_limit)
            max_val = max(max_val, val)
        transpo_table[key] = (depth, max_val)
        return max_val

    else:
        # Noeud chance : le jeu insère une tuile 2 (90%) ou 4 (10%) dans une case vide
        cells = board.get_empty_cells()
        if not cells:
            return eval_fn(board)

        total = 0
        count = 0
        sampled = random.sample(cells, min(4, len(cells)))  # on limite à 4 échantillons pour rester rapide

        for x, y in sampled:
            for value, prob in [(2, 0.9), (4, 0.1)]:
                if prob < 0.05:
                    continue
                temp = board.clone()
                temp.set_tile(x, y, value)
                val = expectimax(temp, depth - 1, True, eval_fn, transpo_table, time_limit)
                total += prob * val
            count += 1

        expected = total / count if count > 0 else 0
        transpo_table[key] = (depth, expected)
        return expected
