from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from .base import BaseModel

_N_FEATURES = 3
_FEATURE_NAMES = ["RV_d", "RV_w", "RV_m"]


class HAROLS(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="M2-HAR-OLS")
        self._model     = LinearRegression(fit_intercept=True)
        self.coef_      : np.ndarray | None = None
        self.intercept_ : float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HAROLS":
        X = self._check_input(X, expected_cols=_N_FEATURES)
        self._model.fit(X, y)
        self.coef_      = self._model.coef_
        self.intercept_ = float(self._model.intercept_)
        self.is_fitted  = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._check_input(X, expected_cols=_N_FEATURES)
        return self._model.predict(X)

    def coef_dict(self) -> dict:
        if self.coef_ is None:
            return {}
        return dict(zip(_FEATURE_NAMES, self.coef_))
