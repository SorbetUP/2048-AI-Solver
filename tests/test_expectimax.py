import unittest
from board import Board
from search.expectimax import best_move

class DummyVictor:
    """
    Évaluation simple imitant une IA : plus il y a de cases vides, mieux c'est.
    Sert à tester l'intégration de fonctions personnalisées (ex : Victor).
    """
    def __call__(self, board: Board) -> float:
        raw = board.raw
        return sum(((raw >> (i * 4)) & 0xF) == 0 for i in range(16))


class TestExpectimax(unittest.TestCase):

    def test_deterministic_board_best_move(self):
        """
        Grille simple avec deux tuiles fusionnables à gauche.
        On s'attend à ce que l'algorithme choisisse 'left'.
        """
        board = Board()
        board._b = 0  # grille vide

        # Place manuellement des tuiles 2 côte à côte
        tiles = [1, 1, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0]
        b = 0
        for i, val in enumerate(tiles):
            b |= (val << (i * 4))
        board._b = b

        move = best_move(board, depth=2, time_limit_ms=1000)
        self.assertEqual(move, "left", f"Expected 'left' but got {move}")

    def test_eval_fn_victor_like(self):
        """
        Teste l'intégration avec une fonction d'évaluation personnalisée.
        Vérifie que l’algorithme retourne bien un coup valide.
        """
        board = Board()
        board._b = 0
        tiles = [1, 2, 0, 0,
                 0, 0, 0, 0,
                 0, 3, 0, 0,
                 0, 0, 0, 0]
        b = 0
        for i, val in enumerate(tiles):
            b |= (val << (i * 4))
        board._b = b

        move = best_move(board, depth=3, time_limit_ms=1000, eval_fn=DummyVictor())
        self.assertIn(move, ["up", "left", "down", "right"], f"Unexpected move: {move}")


if __name__ == '__main__':
    unittest.main()