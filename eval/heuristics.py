from board import Board

def basic_eval(board: Board) -> float:
    """
    Heuristique de base pour évaluer une grille :
    - Favorise les grilles avec beaucoup de cases vides
    - Donne un petit bonus si la plus grosse tuile est élevée
    (plus la tuile max est grande, plus la grille est potentiellement prometteuse)
    """
    raw = board.raw
    empty_cells = sum(((raw >> (i * 4)) & 0xF) == 0 for i in range(16))
    max_tile = board.max_tile()
    return empty_cells + (max_tile / 2048)  # Bonus si tuile max > 2048