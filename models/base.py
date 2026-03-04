from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    def __init__(self, name: str) -> None:
        self.name      = name
        self.is_fitted = False

    # ── Interface ─────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:

    # ── Utilities ─────────────────────────────────────────────────────────

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name}: call fit() before predict()."
            )

    def _check_input(self, X: np.ndarray, expected_cols: int | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if expected_cols is not None and X.shape[1] != expected_cols:
            raise ValueError(
                f"{self.name}: expected {expected_cols} features, "
                f"got {X.shape[1]}."
            )
        return X

    def coef_dict(self) -> dict:
        return {}

    def __repr__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted})"
