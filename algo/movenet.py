"""
algo/movenet.py – wrapper pour HistGradientBoostingClassifier
──────────────────────────────────────────────────────────────
Entrée  : 17 entiers uint16 → [empty_cnt, c0…c15]
Sortie  : str  ∈ {"up","down","left","right"}
"""

from pathlib import Path
import joblib, numpy as np

DIRS = ["up", "down", "left", "right"]          # id → label


class MoveNet:
    """Petit wrapper pour appeler le modèle en 1 ligne :  mv = movenet(board)"""

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        self.clf = joblib.load(model_path)      # HistGradientBoostingClassifier

    # ────────────────────────── helpers ──────────────────────────
    @staticmethod
    def _features(board) -> np.ndarray:
        """Encode la grille (uint16) + nombre de cases vides (uint16)."""
        raw = board.raw
        exp = np.array([(raw >> (i * 4)) & 0xF for i in range(16)], dtype=np.uint8)

        empty_cnt = np.uint16((exp == 0).sum())
        tiles     = np.where(exp == 0, 0, 1 << exp).astype(np.uint16)

        return np.concatenate(([empty_cnt], tiles)).reshape(1, -1)  # shape (1, 17)

    # ────────────────────────── prédiction ───────────────────────
    def __call__(self, board, *_) -> str:
        pred = self.clf.predict(self._features(board))[0]
        # le modèle peut renvoyer int ou str :
        return DIRS[int(pred)] if isinstance(pred, (int, np.integer)) else str(pred)
