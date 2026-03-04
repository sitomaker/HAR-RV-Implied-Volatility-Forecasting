from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .base import BaseModel

_N_FEATURES = 11
_FEATURE_NAMES = [
    "RV_d", "RV_w", "RV_m",
    "VRP",
    "slope", "curv",
    "dy10", "MOVE", "spread",
    "r_lag1", "r_lag5",
]


class HARExtended(BaseModel):
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Parameters
        ----------
        alpha : Ridge penalty lambda (selected by inner CV externally).
        """
        super().__init__(name="M4-HAR-Extended")
        self.alpha      = alpha
        self._model     = Ridge(alpha=alpha, fit_intercept=True)
        self.coef_      : np.ndarray | None = None
        self.intercept_ : float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HARExtended":
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

    @property
    def beta_vrp(self) -> float | None:
        if self.coef_ is None:
            return None
        idx = _FEATURE_NAMES.index("VRP")
        return float(self.coef_[idx])
