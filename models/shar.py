from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from .base import BaseModel

_N_FEATURES    = 4
_FEATURE_NAMES = ["RV_d_plus", "RV_d_minus", "RV_w", "RV_m"]


class SHAR(BaseModel):

    def __init__(self) -> None:
        super().__init__(name="M3-SHAR")
        self._model     = LinearRegression(fit_intercept=True)
        self.coef_      : np.ndarray | None = None
        self.intercept_ : float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SHAR":
        
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

    # ── Asymmetry diagnostics ─────────────────────────────────────────────

    @property
    def beta_d_plus(self) -> float | None:
        return float(self.coef_[0]) if self.coef_ is not None else None

    @property
    def beta_d_minus(self) -> float | None:
        return float(self.coef_[1]) if self.coef_ is not None else None

    def asymmetry_holds(self) -> bool:
        
        self._check_is_fitted()
        return abs(self.beta_d_minus) > abs(self.beta_d_plus)

    def asymmetry_ratio(self) -> float | None:
        
        self._check_is_fitted()
        denom = abs(self.beta_d_plus)
        if denom < 1e-12:
            return None
        return abs(self.beta_d_minus) / denom
