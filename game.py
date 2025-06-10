from board import Board

class Game:
    WIN_TILE = 2048

    def __init__(self):
        self.board = Board()
        self.over = False
        self.won = False
        self.score = 0

    def move(self, direction):
        if direction == 'left':
            moved, gained = self.board.move_left()
        elif direction == 'right':
            moved, gained = self.board.move_right()
        elif direction == 'up':
            moved, gained = self.board.move_up()
        elif direction == 'down':
            moved, gained = self.board.move_down()
        else:
            return False, 0

        if moved:
            self.score += gained
            if self.board.max_tile() >= self.WIN_TILE:
                self.won = True
                self.over = True
            elif not self.board.can_move():
                self.over = True
        return moved, gained

    def is_over(self):
        return self.over

    def is_won(self):
        return self.won
