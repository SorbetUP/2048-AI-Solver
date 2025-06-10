import random

SIZE = 4

class Board:
    def __init__(self):
        self.grid = [[0]*SIZE for _ in range(SIZE)]
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if self.grid[r][c] == 0]
        if not empty:
            return
        r, c = random.choice(empty)
        self.grid[r][c] = 4 if random.random() < 0.1 else 2

    def compress(self, row):
        new_row = [v for v in row if v != 0]
        new_row += [0] * (SIZE - len(new_row))
        return new_row

    def merge(self, row):
        score = 0
        for i in range(SIZE - 1):
            if row[i] != 0 and row[i] == row[i+1]:
                row[i] *= 2
                score += row[i]  # ajout au score
                row[i+1] = 0
        return row, score

    def move_left(self):
        moved = False
        total_score = 0
        for i in range(SIZE):
            original = self.grid[i][:]
            compressed = self.compress(original)
            merged, score = self.merge(compressed)
            total_score += score
            new_row = self.compress(merged)
            self.grid[i] = new_row
            if new_row != original:
                moved = True
        if moved:
            self.add_random_tile()
        return moved, total_score

    def move_right(self):
        self.reflect_horizontal()
        moved, score = self.move_left()
        self.reflect_horizontal()
        return moved, score

    def move_up(self):
        self.transpose()
        moved, score = self.move_left()
        self.transpose()
        return moved, score

    def move_down(self):
        self.transpose()
        moved, score = self.move_right()
        self.transpose()
        return moved, score

    def transpose(self):
        self.grid = [list(row) for row in zip(*self.grid)]

    def reflect_horizontal(self):
        for i in range(SIZE):
            self.grid[i].reverse()

    def can_move(self):
        for row in self.grid:
            if 0 in row:
                return True
        for r in range(SIZE):
            for c in range(SIZE - 1):
                if self.grid[r][c] == self.grid[r][c+1]:
                    return True
        for c in range(SIZE):
            for r in range(SIZE - 1):
                if self.grid[r][c] == self.grid[r+1][c]:
                    return True
        return False

    def max_tile(self):
        return max(max(row) for row in self.grid)
