from board import Board

def basic_eval(board: Board) -> float:
    """Évaluation basique : favorise les grilles avec + de cases vides et une grosse tuile"""
    raw = board.raw
    empty_cells = sum(((raw >> (i * 4)) & 0xF) == 0 for i in range(16))
    max_tile = board.max_tile()
    return empty_cells + (max_tile / 2048)  # bonus pour des grosses tuiles
