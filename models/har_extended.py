"""
models/har_extended.py — M4: HAR-Extended (Primary Specification)

LaTeX eq. (8):
    y_t^(h) = alpha
              + beta_d  * RV_t^(d)   + beta_w  * RV_t^(w)  + beta_m * RV_t^(m)
              + gamma_1 * VRP_t
              + gamma_2 * slope_t    + gamma_3 * curv_t
              + gamma_4 * dy10_t     + gamma_5 * MOVE_t     + gamma_6 * spread_t
              + gamma_7 * r_{t-1}    + gamma_8 * r_{t-1:t-5}
              + epsilon_t

Estimated via Ridge (l2) regularisation (LaTeX eq. 9):

    theta_hat = argmin_theta { sum_t (y_t - x_t' theta)^2 + lambda ||theta_{-0}||^2 }

where theta_{-0} excludes the intercept from penalisation and lambda is
selected by expanding-window inner CV (walk_forward.py).

All features are standardised to zero mean and unit variance using
training-set statistics only before Ridge estimation. The intercept is
fitted to the unstandardised target.

Ridge is preferred over Lasso because:
  (i)  HAR regressors are mechanically collinear; Lasso would arbitrarily
       zero one of a correlated group rather than shrink both jointly.
  (ii) With 11 regressors against ~500+ observations, variance reduction
       takes priority over sparsity.

M4 nests M2 under gamma_1 = ... = gamma_8 = 0. The key coefficient is
gamma_1 = beta_VRP, predicted negative by the VRP mean-reversion
proposition: a high variance risk premium today forecasts a decline in
implied volatility.
"""
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
    """
    M4: HAR-Extended with Ridge (l2) regularisation.

    Features expected in order (standardised externally):
        [RV_d, RV_w, RV_m, VRP, slope, curv,
         dy10, MOVE, spread, r_lag1, r_lag5]   — 11 features.

    The Ridge penalty lambda is passed at construction; walk_forward.py
    selects it via inner CV before instantiating this class.
    """

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
        """
        Ridge regression on standardised features.

        Parameters
        ----------
        X : (T, 11) — standardised features in _FEATURE_NAMES order.
        y : (T,)    — target y^(h).
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

    @property
    def beta_vrp(self) -> float | None:
        """
        gamma_1 = beta_VRP. Predicted negative under VRP mean-reversion
        (LaTeX Proposition: VRP mean-reversion implies beta_VRP < 0).
        """
        if self.coef_ is None:
            return None
        idx = _FEATURE_NAMES.index("VRP")
        return float(self.coef_[idx])