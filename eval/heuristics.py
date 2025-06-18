# eval/heuristics.py
from board import Board
import math

# ────────────────────────────────────────────────────────────────
def basic_eval(board: Board) -> float:
    """
    Heuristique d’origine : cases vides + bonus max-tile / 2048.
    Utilisée encore par certains scripts.
    """
    raw = board.raw
    empty = sum(((raw >> (i * 4)) & 0xF) == 0 for i in range(16))
    max_tile = board.max_tile()
    return empty + (max_tile / 2048)


# ────────────────────────────────────────────────────────────────
def bounded_eval(board: Board) -> float:
    """
    Heuristique **bornée dans [0 ; 1]** pour permettre les
    bornes α/β dans BEPP.

    • Ratio de cases vides (poids 0.6)  
    • Exposant (log2) de la plus grosse tuile / 16 (poids 0.4)

    Valeur 0 : grille pleine avec tuile 2  
    Valeur 1 : grille vide ou tuile 65 536 atteinte.
    """
    raw = board.raw
    empty_ratio = sum(((raw >> (i * 4)) & 0xF) == 0 for i in range(16)) / 16.0

    max_tile_exp = int(math.log2(board.max_tile()))
    max_ratio = max_tile_exp / 16.0          # 16 → tuile 65 536

    return 0.6 * empty_ratio + 0.4 * max_ratio
