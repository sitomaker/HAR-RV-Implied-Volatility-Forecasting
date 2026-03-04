"""
models/random_walk.py — M0: Random Walk Benchmark

LaTeX eq. (1):
    y_hat_t^(h) = 0,  for all t, h.

The random walk embodies the null of no predictability: the best
forecast of any future IV change is zero. Every competing model must
produce statistically significant improvements over M0 to be considered
useful. Out-of-sample R² relative to M0 is the primary evaluation metric.
"""
from __future__ import annotations

import numpy as np

from .base import BaseModel


class RandomWalk(BaseModel):
    """
    M0: Random Walk — predicts zero change in IV at every period.

    fit() is a no-op; predict() returns a zero vector. Inherits input
    validation and is_fitted guard from BaseModel.
    """

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