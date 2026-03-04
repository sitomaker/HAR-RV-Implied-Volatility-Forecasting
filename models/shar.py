"""
models/shar.py — M3: Semi-HAR (SHAR)

LaTeX eq. (7):
    y_t^(h) = alpha + beta_d^+ * RV_t^(d,+)
                    + beta_d^- * RV_t^(d,-)
                    + beta_w   * RV_t^(w)
                    + beta_m   * RV_t^(m) + epsilon_t

M3 nests M2 under H_0: beta_d^+ = beta_d^-, testable via a standard
Wald test on OLS residuals (tests.py). The asymmetry hypothesis predicts
|beta_d^-| > |beta_d^+|: downside semi-variance carries more predictive
content for future implied volatility than upside semi-variance.

Proxy caveat (LaTeX remark, §features): the daily sign-classification
proxy for semi-variance loses within-day asymmetry information. Weak
results in M3 may reflect proxy coarseness rather than the absence of
asymmetric volatility dynamics.

Feature standardisation is handled externally by walk_forward.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from .base import BaseModel

_N_FEATURES    = 4
_FEATURE_NAMES = ["RV_d_plus", "RV_d_minus", "RV_w", "RV_m"]


class SHAR(BaseModel):
    """
    M3: Semi-HAR — separates daily RV into signed upside/downside
    semi-variance components following Patton & Sheppard (2015).

    Features expected in order (standardised externally):
        [RV_d_plus, RV_d_minus, RV_w, RV_m]
    """

    def __init__(self) -> None:
        super().__init__(name="M3-SHAR")
        self._model     = LinearRegression(fit_intercept=True)
        self.coef_      : np.ndarray | None = None
        self.intercept_ : float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SHAR":
        """
        OLS estimation.

        Parameters
        ----------
        X : (T, 4) — [RV_d_plus, RV_d_minus, RV_w, RV_m], standardised.
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

    # ── Asymmetry diagnostics ─────────────────────────────────────────────

    @property
    def beta_d_plus(self) -> float | None:
        return float(self.coef_[0]) if self.coef_ is not None else None

    @property
    def beta_d_minus(self) -> float | None:
        return float(self.coef_[1]) if self.coef_ is not None else None

    def asymmetry_holds(self) -> bool:
        """
        Returns True if |beta_d^-| > |beta_d^+|, as the asymmetry
        hypothesis predicts. Requires fit() to have been called.
        """
        self._check_is_fitted()
        return abs(self.beta_d_minus) > abs(self.beta_d_plus)

    def asymmetry_ratio(self) -> float | None:
        """
        Returns |beta_d^-| / |beta_d^+|. Values > 1 support the
        asymmetry hypothesis. Returns None if beta_d_plus is zero.
        """
        self._check_is_fitted()
        denom = abs(self.beta_d_plus)
        if denom < 1e-12:
            return None
        return abs(self.beta_d_minus) / denom