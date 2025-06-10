"""
board.py – moteur 2048 « bit-board » ultrarapide
© 2025 – libre de droits, sans garantie
"""
from __future__ import annotations
import random
import numpy as np
import numba as nb

# ────────────────────────────────────────────────────────────────
# 1. Tables de transition (65 536 états de ligne) + score
# ────────────────────────────────────────────────────────────────
ROW_LEFT  = np.empty(1 << 16, dtype=np.uint16)
ROW_RIGHT = np.empty_like(ROW_LEFT)
# Le score peut dépasser 65 535 → on passe en uint32
SCORE_LUT = np.empty(1 << 16, dtype=np.uint32)

def _left_row(row16: int) -> tuple[int, int]:
    """Applique le coup Gauche à une ligne codée sur 16 bits, renvoie (nouvelle_ligne, gain)."""
    tiles = [(row16 >> (4 * i)) & 0xF for i in range(4)]
    new = [t for t in tiles if t]           # compression
    score = 0
    i = 0
    while i < len(new) - 1:                 # fusion éventuelle
        if new[i] == new[i + 1]:
            new[i] += 1
            score += 1 << new[i]            # 2**exposant ajouté au score
            del new[i + 1]
        i += 1
    new += [0] * (4 - len(new))             # complétion à gauche
    res = new[0] | (new[1] << 4) | (new[2] << 8) | (new[3] << 12)
    return res, score

for r in range(1 << 16):
    left, sc     = _left_row(r)
    ROW_LEFT[r]  = left
    SCORE_LUT[r] = sc
    # miroir horizontal pour générer le coup « droite »
    rev = ((r >> 12) & 0xF) | ((r >> 4) & 0xF0) | ((r << 4) & 0xF00) | ((r << 12) & 0xF000)
    rev_left, _  = _left_row(rev)
    ROW_RIGHT[r] = ((rev_left >> 12) & 0xF) | ((rev_left >> 4) & 0xF0) | \
                   ((rev_left << 4) & 0xF00) | ((rev_left << 12) & 0xF000)

# ────────────────────────────────────────────────────────────────
# 2. Fonctions bas-niveau (JIT numba)
# ────────────────────────────────────────────────────────────────
@nb.njit(inline="always")
def _get(b: nb.uint64, r: nb.int8, c: nb.int8) -> nb.uint8:
    """Renvoie la valeur-exposant (0…15) en (r,c)."""
    return nb.uint8((b >> ((r * 4 + c) * 4)) & 0xF)

@nb.njit(inline="always")
def _set(b: nb.uint64, r: nb.int8, c: nb.int8, val: nb.uint8) -> nb.uint64:
    """Place val (exposant) en (r,c) et renvoie la nouvelle grille."""
    mask = nb.uint64(0xF) << ((r * 4 + c) * 4)
    return (b & ~mask) | (nb.uint64(val) << ((r * 4 + c) * 4))

@nb.njit(cache=True, fastmath=True)
def _transpose(b: nb.uint64) -> nb.uint64:
    """Transpose la grille 4×4 stockée sur 64 bits."""
    res = nb.uint64(0)
    for r in range(4):
        for c in range(4):
            res = _set(res, c, r, _get(b, r, c))
    return res

@nb.njit(cache=True, fastmath=True)
def move_board(b: nb.uint64, direction: nb.int8) -> tuple[nb.uint64, nb.int32, nb.bool_]:
    """
    Applique le coup : 0=left 1=right 2=up 3=down
    Retourne (nouveau_board, gain, a_bouge?).
    """
    # Les coups verticaux utilisent une transposition
    if direction > 1:          # up / down
        b = _transpose(b)

    new_b  = nb.uint64(0)
    score  = nb.int32(0)
    moved  = False
    lut    = ROW_LEFT if direction % 2 == 0 else ROW_RIGHT

    for row in range(4):
        r16   = nb.uint16((b >> (row * 16)) & 0xFFFF)
        new16 = lut[r16]
        if new16 != r16:
            moved = True
        new_b |= nb.uint64(new16) << (row * 16)
        score += nb.int32(SCORE_LUT[r16])   # cast explicite → int32

    if direction > 1:          # re-transpose
        new_b = _transpose(new_b)

    return new_b, score, moved

@nb.njit(cache=True)
def can_move(b: nb.uint64) -> nb.bool_:
    """Teste s’il reste au moins un coup légal."""
    # case vide ?
    for pos in range(16):
        if ((b >> (pos * 4)) & 0xF) == 0:
            return True
    # égalité horizontale
    for r in range(4):
        for c in range(3):
            if _get(b, r, c) == _get(b, r, c + 1):
                return True
    # égalité verticale
    for c in range(4):
        for r in range(3):
            if _get(b, r, c) == _get(b, r + 1, c):
                return True
    return False

# ────────────────────────────────────────────────────────────────
# 3. Classe Board (interface Python conviviale)
# ────────────────────────────────────────────────────────────────
class Board:
    __slots__ = ("_b",)

    def __init__(self):
        self._b: np.uint64 = np.uint64(0)
        self._add_random_tile()
        self._add_random_tile()

    # — propriétés utiles ————————————————————————————
    @property
    def raw(self) -> int:
        """Retourne l’entier 64 bits brut (debug/serialisation)."""
        return int(self._b)

    def max_tile(self) -> int:
        """Plus grande tuile présente (en valeur 2ⁿ)."""
        return 1 << max((_get(self._b, r, c) for r in range(4) for c in range(4)))

    # — logique de jeu ——————————————————————————————
    def move(self, direction: str) -> tuple[bool, int]:
        """direction ∈ {'left','right','up','down'}, renvoie (moved?, gain)."""
        dir_id = {"left": 0, "right": 1, "up": 2, "down": 3}[direction]
        new_b, gain, moved = move_board(self._b, dir_id)
        if moved:
            self._b = new_b
            self._add_random_tile()
        return bool(moved), int(gain)

    def can_move(self) -> bool:
        return bool(can_move(self._b))

    # — utilitaires internes —————————————————————————
    def _add_random_tile(self):
        empties = [pos for pos in range(16) if ((self._b >> (pos * 4)) & 0xF) == 0]
        if not empties:
            return
        pos = random.choice(empties)
        val = 1 if random.random() < 0.9 else 2   # 2 (90 %) ou 4 (10 %)
        self._b |= np.uint64(val) << (pos * 4)

    # — rendu texte minimal (debug) ——————————————————
    def __str__(self) -> str:
        line = "+-----" * 4 + "+\n"
        out  = line
        for r in range(4):
            out += "|"
            for c in range(4):
                v = _get(self._b, r, c)
                out += f"{(1 << v) if v else '.':>5}|"
            out += "\n" + line
        return out
