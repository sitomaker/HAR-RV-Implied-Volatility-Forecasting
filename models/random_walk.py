from __future__ import annotations

import numpy as np

from .base import BaseModel


class RandomWalk(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="M0-RandomWalk")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomWalk":
        self._check_input(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._check_input(X)
        return np.zeros(X.shape[0])
