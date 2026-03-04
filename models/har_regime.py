"""
models/har_regime.py — M5: HAR with Selective Regime Interactions

LaTeX eq. (10) — mixed specification:

    y_t^(h) = β_d · RV_t^(d) + β_w · RV_t^(w) + β_m · RV_t^(m)
              + Σ_{k ∈ {L,M,H}} 1(regime_t = k) · [α_k + γ_k · VRP_t]
              + ε_t

Shared slopes (β_d, β_w, β_m) capture physical volatility dynamics
that are approximately regime-invariant. Regime-specific slopes (γ_k)
are applied only to VRP, whose relationship with ΔVIX is known to
change across volatility states (Bekaert & Hoerova, 2014).

Parameter count: 3 shared + 1×3 interaction + 3 intercepts = 9.
(Original fully-interacted version: 4×3 + 3 = 15.)

Design matrix (9 columns):
    [RV_d  RV_w  RV_m | VRP×Low  VRP×Med  VRP×High | 1_Low  1_Med  1_High]
     ─── shared ───    ──── regime-specific ─────    ──── intercepts ─────

Ridge regularisation with λ selected by inner CV.
fit_intercept=False because regime-specific intercepts are explicit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import BaseModel
from config import FEATURES_M5_BASE, FEATURES_M5_INTERACT

_REGIMES = ["Low", "Medium", "High"]

# Derived lists
_SHARED_FEATURES  = [f for f in FEATURES_M5_BASE
                     if f not in FEATURES_M5_INTERACT]
_INTERACT_FEATURES = [f for f in FEATURES_M5_BASE
                      if f in FEATURES_M5_INTERACT]

# Column count: shared + interactions×regimes + intercepts
_N_SHARED    = len(_SHARED_FEATURES)                          # 3
_N_INTERACT  = len(_INTERACT_FEATURES) * len(_REGIMES)        # 1×3 = 3
_N_INTERCEPT = len(_REGIMES)                                  # 3
_N_COLS      = _N_SHARED + _N_INTERACT + _N_INTERCEPT         # 9


def _build_design_matrix(X_df: pd.DataFrame,
                         regime: pd.Series) -> np.ndarray:
    """
    Build the mixed shared/interaction design matrix.

    Layout (9 columns for default config):
        [0:3]  shared:      RV_d, RV_w, RV_m         (same across regimes)
        [3:6]  interaction:  VRP×Low, VRP×Med, VRP×High
        [6:9]  intercepts:  1_Low, 1_Med, 1_High

    Parameters
    ----------
    X_df   : DataFrame with columns ⊇ FEATURES_M5_BASE, standardised.
    regime : Series of str ('Low', 'Medium', 'High'), aligned with X_df.

    Returns
    -------
    matrix : (T, _N_COLS) float64 array.
    """
    T = len(X_df)
    matrix = np.zeros((T, _N_COLS), dtype=np.float64)
    col = 0

    # ── 1. Shared slopes (regime-invariant) ───────────────────
    for feat in _SHARED_FEATURES:
        matrix[:, col] = X_df[feat].to_numpy(dtype=np.float64)
        col += 1

    # ── 2. Regime-specific slopes (interactions) ──────────────
    for feat in _INTERACT_FEATURES:
        feat_vals = X_df[feat].to_numpy(dtype=np.float64)
        for k in _REGIMES:
            mask = (regime == k).to_numpy(dtype=np.float64)
            matrix[:, col] = feat_vals * mask
            col += 1

    # ── 3. Regime-specific intercepts ─────────────────────────
    for k in _REGIMES:
        matrix[:, col] = (regime == k).to_numpy(dtype=np.float64)
        col += 1

    assert col == _N_COLS, f"Expected {_N_COLS} columns, built {col}."
    return matrix


def _coef_names() -> list[str]:
    """Human-readable coefficient names matching design matrix layout."""
    names = []
    # Shared
    for feat in _SHARED_FEATURES:
        names.append(feat)
    # Interactions
    for feat in _INTERACT_FEATURES:
        for k in _REGIMES:
            names.append(f"{feat}_x_{k}")
    # Intercepts
    for k in _REGIMES:
        names.append(f"intercept_{k}")
    return names


class HARRegime(BaseModel):
    """
    M5: HAR with Selective Regime Interactions (Ridge).

    Usage
    -----
    model = HARRegime(alpha=lam)
    model.fit(X_train_df, y_train, regime_train)
    preds = model.predict(X_test_df, regime_test)

    X : DataFrame with at least FEATURES_M5_BASE columns (standardised).
    regime : Series of str ('Low', 'Medium', 'High').
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(name="M5-HAR-Regime")
        self.alpha  = alpha
        self._model = Ridge(alpha=alpha, fit_intercept=False)
        self.coef_: np.ndarray | None = None

    def fit(self,
            X: pd.DataFrame,
            y: np.ndarray,
            regime: pd.Series) -> "HARRegime":
        """
        Fit Ridge on the mixed design matrix.

        Parameters
        ----------
        X      : DataFrame (T, ≥4) with FEATURES_M5_BASE.
        y      : (T,) target array.
        regime : (T,) Series of regime labels.
        """
        Z = _build_design_matrix(X, regime)
        self._model.fit(Z, y)
        self.coef_     = self._model.coef_
        self.is_fitted = True
        return self

    def predict(self,
                X: pd.DataFrame,
                regime: pd.Series) -> np.ndarray:
        """Predict using the mixed design matrix."""
        self._check_is_fitted()
        Z = _build_design_matrix(X, regime)
        return self._model.predict(Z)

    def coef_dict(self) -> dict:
        """Return {name: coef} for all 9 coefficients."""
        if self.coef_ is None:
            return {}
        return dict(zip(_coef_names(), self.coef_))

    def shared_coefs(self) -> dict:
        """Return the regime-invariant (shared) coefficients."""
        self._check_is_fitted()
        cd = self.coef_dict()
        return {f: cd[f] for f in _SHARED_FEATURES}

    def regime_coefs(self, regime: str) -> dict:
        """
        Return intercept + interaction coefficients for one regime.

        Example for regime='High':
            {'VRP_x_High': ..., 'intercept_High': ...}
        """
        self._check_is_fitted()
        if regime not in _REGIMES:
            raise ValueError(
                f"regime must be one of {_REGIMES}, got {regime!r}."
            )
        cd = self.coef_dict()
        keys = [f"{f}_x_{regime}" for f in _INTERACT_FEATURES]
        keys.append(f"intercept_{regime}")
        return {k: cd[k] for k in keys}

    def summary(self) -> str:
        """Print a formatted coefficient summary."""
        self._check_is_fitted()
        lines = ["M5 HAR-Regime Coefficients",
                 "=" * 40,
                 "",
                 "Shared (regime-invariant):"]
        for k, v in self.shared_coefs().items():
            lines.append(f"  {k:20s}: {v:+.6f}")
        lines.append("")
        for reg in _REGIMES:
            lines.append(f"Regime = {reg}:")
            for k, v in self.regime_coefs(reg).items():
                lines.append(f"  {k:20s}: {v:+.6f}")
            lines.append("")
        return "\n".join(lines)