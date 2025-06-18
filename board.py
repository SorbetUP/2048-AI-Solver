"""
board.py – moteur 2048 « bit-board » ultrarapide
© 2025 – libre de droits
"""
from __future__ import annotations
import random
import numpy as np
import numba as nb

# ───────────────────────────────────────── LUT lignes ───────────────────────
ROW_LEFT  = np.empty(1 << 16, dtype=np.uint16)
ROW_RIGHT = np.empty_like(ROW_LEFT)
SCORE_LUT = np.empty(1 << 16, dtype=np.uint32)


def _left_row(row16: int) -> tuple[int, int]:
    tiles = [(row16 >> (4 * i)) & 0xF for i in range(4)]
    new   = [t for t in tiles if t]
    score = 0
    i = 0
    while i < len(new) - 1:
        if new[i] == new[i + 1]:
            new[i] += 1
            score  += 1 << new[i]
            del new[i + 1]
        i += 1
    new += [0] * (4 - len(new))
    res = new[0] | (new[1] << 4) | (new[2] << 8) | (new[3] << 12)
    return res, score


for r in range(1 << 16):
    left, sc      = _left_row(r)
    ROW_LEFT[r]   = left
    SCORE_LUT[r]  = sc
    rev           = ((r >> 12) & 0xF) | ((r >> 4) & 0xF0) \
                  | ((r << 4) & 0xF00) | ((r << 12) & 0xF000)
    rev_left, _   = _left_row(rev)
    ROW_RIGHT[r]  = ((rev_left >> 12) & 0xF) | ((rev_left >> 4) & 0xF0) \
                  | ((rev_left << 4) & 0xF00) | ((rev_left << 12) & 0xF000)

# ─────────────────────────────────── kernels JIT ────────────────────────────
@nb.njit(inline="always")
def _get(b: nb.uint64, r: nb.int8, c: nb.int8) -> nb.uint8:
    return nb.uint8((b >> ((r * 4 + c) * 4)) & 0xF)


@nb.njit(inline="always")
def _set(b: nb.uint64, r: nb.int8, c: nb.int8, v: nb.uint8) -> nb.uint64:
    mask = nb.uint64(0xF) << ((r * 4 + c) * 4)
    return (b & ~mask) | (nb.uint64(v) << ((r * 4 + c) * 4))


@nb.njit(cache=True, fastmath=True)
def _transpose(b: nb.uint64) -> nb.uint64:
    res = nb.uint64(0)
    for r in range(4):
        for c in range(4):
            res = _set(res, c, r, _get(b, r, c))
    return res


@nb.njit(cache=True, fastmath=True)
def move_board(b: nb.uint64, d: nb.int8) -> tuple[nb.uint64, nb.int32, nb.bool_]:
    """0← 1→ 2↑ 3↓  – renvoie (board, gain, bougé ?)."""
    if d > 1:                    # coups verticaux → transpose
        b = _transpose(b)

    new_b  = nb.uint64(0)
    score  = nb.int32(0)
    moved  = False
    lut    = ROW_LEFT if d % 2 == 0 else ROW_RIGHT

    for row in range(4):
        r16   = nb.uint16((b >> (row * 16)) & 0xFFFF)
        new16 = lut[r16]
        if new16 != r16:
            moved = True
        new_b |= nb.uint64(new16) << (row * 16)
        score += nb.int32(SCORE_LUT[r16])

    if d > 1:
        new_b = _transpose(new_b)

    return new_b, score, moved


@nb.njit(cache=True)
def can_move(b: nb.uint64) -> nb.bool_:
    for pos in range(16):
        if ((b >> (pos * 4)) & 0xF) == 0:
            return True
    for r in range(4):
        for c in range(3):
            if _get(b, r, c) == _get(b, r, c + 1):
                return True
    for c in range(4):
        for r in range(3):
            if _get(b, r, c) == _get(b, r + 1, c):
                return True
    return False

# ──────────────────────────────── classe Board ─────────────────────────────
class Board:
    __slots__ = ("_b",)

    def __init__(self):
        self._b = np.uint64(0)
        self._add_random_tile()
        self._add_random_tile()

    # ---------------------------------------------------------------- clone
    def clone(self) -> "Board":
        obj = Board.__new__(Board)
        obj._b = np.uint64(self._b)      # assure le type
        return obj

    # ----------------------------------------------------------- hash / eq
    def __hash__(self): return int(self._b)
    def __eq__(self, o): return isinstance(o, Board) and int(self._b) == int(o._b)

    # ------------------------------------------------------------- helpers
    @property
    def raw(self) -> int:
        return int(self._b)

    def max_tile(self) -> int:
        return 1 << max((_get(np.uint64(self._b), r, c)  # cast de sûreté
                         for r in range(4) for c in range(4)))

    def get_empty_cells(self) -> list[tuple[int, int]]:
        return [(p // 4, p % 4) for p in range(16)
                if ((np.uint64(self._b) >> (p * 4)) & 0xF) == 0]

    def set_tile(self, r: int, c: int, exp: int):
        self._b = np.uint64(_set(np.uint64(self._b), r, c, np.uint8(exp)))

    # ---------------------------------------------------------------- move
    def move(self, direction: str, *, add_random: bool = True) -> tuple[bool, int]:
        dir_id = {"left": 0, "right": 1, "up": 2, "down": 3}[direction]
        # 🔑 on force le type ⇒ jamais (float64, int64)
        new_b, gain, moved = move_board(np.uint64(self._b), np.int8(dir_id))
        if moved:
            self._b = np.uint64(new_b)
            if add_random:
                self._add_random_tile()
        return bool(moved), int(gain)

    def can_move(self) -> bool:
        return bool(can_move(np.uint64(self._b)))

    # ----------------------------------------------------------- internals
    def _add_random_tile(self):
        empties = self.get_empty_cells()
        if not empties:
            return
        r, c = random.choice(empties)
        self.set_tile(r, c, 1 if random.random() < 0.9 else 2)

    # ------------------------------------------------------------ debug str
    def __str__(self):
        sep = "+-----" * 4 + "+\n"
        s   = sep
        for r in range(4):
            s += "|"
            for c in range(4):
                v = _get(np.uint64(self._b), r, c)
                s += f"{(1 << v) if v else '.':>5}|"
            s += "\n" + sep
        return s
