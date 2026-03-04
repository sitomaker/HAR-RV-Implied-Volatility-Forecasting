"""
models/har_ols.py — M2: HAR-OLS (Academic Baseline)

LaTeX eq. (6):
    y_t^(h) = alpha + beta_d * RV_t^(d)
                    + beta_w * RV_t^(w)
                    + beta_m * RV_t^(m) + epsilon_t

Estimated by OLS. Newey-West HAC standard errors with Andrews (1991)
data-dependent bandwidth are applied in statistical tests (tests.py),
not during fit():

    q* = floor( 4 * (T/100)^(2/9) )

For T = 3,773 this gives q* ≈ 8 > h-1 = 4, satisfying the minimum
lag requirement for the h=5 overlapping target (MA(4) structure).

Feature standardisation is handled externally by walk_forward.py
using training-set statistics only.

M2 is nested within M4 under gamma_1 = ... = gamma_8 = 0, and
nests M0 (testable via Clark-West).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from .base import BaseModel

_N_FEATURES = 3
_FEATURE_NAMES = ["RV_d", "RV_w", "RV_m"]


class HAROLS(BaseModel):
    """
    M2: HAR-OLS — three HAR components estimated by OLS.

    Features expected in order (standardised externally):
        [RV_d, RV_w, RV_m]
    """

    def __init__(self) -> None:
        super().__init__(name="M2-HAR-OLS")
        self._model     = LinearRegression(fit_intercept=True)
        self.coef_      : np.ndarray | None = None
        self.intercept_ : float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HAROLS":
        """
        OLS estimation.

        Parameters
        ----------
        X : (T, 3) — [RV_d, RV_w, RV_m], standardised.
        y : (T,)   — target y^(h).
        """
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