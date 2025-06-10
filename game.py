# game.py
from board import Board

class Game:
    WIN_TILE = 2048

    def __init__(self):
        self.board = Board()
        self.score = 0
        self.over  = False
        self.won   = False

    def move(self, dir_str: str) -> tuple[bool, int]:
        moved, gained = self.board.move(dir_str)
        if moved:
            self.score += gained
            if self.board.max_tile() >= self.WIN_TILE:
                self.won  = True
                self.over = True
            elif not self.board.can_move():
                self.over = True
        return moved, gained

    # raccourcis
    def is_over(self) -> bool: return self.over
    def is_won (self) -> bool: return self.won
