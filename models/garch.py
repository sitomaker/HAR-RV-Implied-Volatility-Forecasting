"""
models/garch.py — M1: GARCH(1,1) Benchmark (v3 — change-based forecast)

Key insight: mapping GARCH *levels* to VIX fails because the P-Q variance
risk premium is time-varying. Instead, we map GARCH *changes* to ΔVIX
via a Mincer-Zarnowitz calibration estimated on training data.

    raw_signal_t = sqrt(252·σ²_{t+h|t}) - sqrt(252·σ²_{t|t-1})
    ŷ_t = a + b · raw_signal_t

This avoids the P-Q level mismatch while preserving GARCH's information
about conditional variance dynamics.
"""
from __future__ import annotations

import numpy as np
from .base import BaseModel

try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False


class GARCHModel(BaseModel):

    def __init__(self, horizon: int = 1) -> None:
        super().__init__(name=f"M1-GARCH(h={horizon})")
        self.horizon = horizon

        # GARCH parameters
        self.mu_: float | None = None
        self.omega_: float | None = None
        self.alpha_: float | None = None
        self.beta_: float | None = None

        # Conditional variance state
        self.sigma2_T_: float | None = None
        self.last_eps2_: float | None = None

        # Calibration coefficients (MZ regression)
        self.calib_a_: float = 0.0
        self.calib_b_: float = 1.0

        # Full conditional variance series (for calibration)
        self._train_sigma2_series: np.ndarray | None = None

        # Test-period data
        self._vix_current: np.ndarray | None = None
        self._test_returns_pct: np.ndarray | None = None

    # ══════════════════════════════════════════════════════════════
    #  FIT
    # ══════════════════════════════════════════════════════════════

    def fit(self, X: np.ndarray, y: np.ndarray,
            returns: np.ndarray | None = None) -> "GARCHModel":
        if returns is None:
            raise ValueError(
                "GARCHModel.fit() requires 'returns' in decimal form."
            )

        ret_pct = np.asarray(returns, dtype=float) * 100.0

        if _ARCH_AVAILABLE:
            fitted = self._fit_qmle(ret_pct)
        else:
            fitted = False
        if not fitted:
            self._fit_fallback(ret_pct)

        # Save last innovation for initial condition
        self.last_eps2_ = (ret_pct[-1] - self.mu_) ** 2

        # Calibrate raw GARCH signal → ΔVIX mapping
        self._calibrate_on_training(ret_pct, y)

        self.is_fitted = True
        return self

    def _fit_qmle(self, ret_pct: np.ndarray) -> bool:
        try:
            import pandas as pd
            ret_series = pd.Series(ret_pct,
                                   index=pd.RangeIndex(len(ret_pct)),
                                   name="ret_pct")
            res = arch_model(
                ret_series,
                vol="Garch", p=1, q=1,
                mean="Constant", dist="Normal",
            ).fit(disp="off", show_warning=False)

            self.mu_ = float(res.params.get(
                "mu", res.params.get("Const", 0.0)))
            self.omega_ = float(res.params["omega"])
            self.alpha_ = float(res.params["alpha[1]"])
            self.beta_ = float(res.params["beta[1]"])

            cond_vol = res.conditional_volatility
            if hasattr(cond_vol, "iloc"):
                self.sigma2_T_ = float(cond_vol.iloc[-1] ** 2)
            else:
                self.sigma2_T_ = float(cond_vol[-1] ** 2)

            # Store full conditional variance series for calibration
            if hasattr(cond_vol, "values"):
                self._train_sigma2_series = cond_vol.values ** 2
            else:
                self._train_sigma2_series = np.array(cond_vol) ** 2

            return True
        except Exception as exc:
            print(f"  GARCH QMLE failed ({exc}). Falling back.")
            return False

    def _fit_fallback(self, ret_pct: np.ndarray) -> None:
        var_ret = float(np.var(ret_pct))
        self.mu_ = float(np.mean(ret_pct))
        self.alpha_ = 0.10
        self.beta_ = 0.85
        self.omega_ = var_ret * (1 - self.alpha_ - self.beta_)
        self.sigma2_T_ = var_ret

        # Build conditional variance series manually
        n = len(ret_pct)
        sigma2 = np.empty(n)
        sigma2[0] = var_ret
        for t in range(1, n):
            eps2 = (ret_pct[t - 1] - self.mu_) ** 2
            sigma2[t] = (self.omega_
                         + self.alpha_ * eps2
                         + self.beta_ * sigma2[t - 1])
        self._train_sigma2_series = sigma2

    def _calibrate_on_training(self, ret_pct: np.ndarray,
                                y: np.ndarray) -> None:
        """
        Mincer-Zarnowitz calibration: regress actual ΔVIX on
        raw GARCH signal (change in annualized vol) using training data.

        raw_signal_t = sqrt(252·σ²_{t+1|t}) - sqrt(252·σ²_{t|t-1})
        y_t = a + b · raw_signal_t + ε_t
        """
        sigma2 = self._train_sigma2_series
        if sigma2 is None or len(sigma2) < 50:
            self.calib_a_ = 0.0
            self.calib_b_ = 1.0
            return

        h = self.horizon
        ab = min(self.alpha_ + self.beta_, 0.9999)
        omega = self.omega_
        sigma2_bar = omega / max(1.0 - ab, 1e-8)

        n = len(sigma2)
        raw_signals = np.empty(n)

        for t in range(n):
            # Current annualized vol
            vol_current = np.sqrt(252.0 * max(sigma2[t], 1e-8))

            # h-step ahead forecast from sigma2[t]
            if t < n - 1:
                # Use next-step sigma2 as the forecast base
                eps2_t = (ret_pct[t] - self.mu_) ** 2
                sigma2_next = omega + self.alpha_ * eps2_t + self.beta_ * sigma2[t]
            else:
                sigma2_next = omega + ab * sigma2[t]

            if h == 1:
                avg_sigma2 = sigma2_next
            else:
                geom = (1.0 - ab ** h) / (1.0 - ab)
                avg_sigma2 = (sigma2_bar
                              + (sigma2_next - sigma2_bar) * geom / h)

            vol_forecast = np.sqrt(252.0 * max(avg_sigma2, 1e-8))
            raw_signals[t] = vol_forecast - vol_current

        # Align with targets (y may be shorter or have NaN)
        n_y = min(len(y), n)
        sig = raw_signals[:n_y]
        tgt = np.asarray(y[:n_y], dtype=float)

        mask = np.isfinite(sig) & np.isfinite(tgt)
        if mask.sum() < 30:
            self.calib_a_ = 0.0
            self.calib_b_ = 1.0
            return

        sig_valid = sig[mask]
        tgt_valid = tgt[mask]

        # OLS: y = a + b·signal
        X_ols = np.column_stack([np.ones(mask.sum()), sig_valid])
        try:
            beta = np.linalg.lstsq(X_ols, tgt_valid, rcond=None)[0]
            self.calib_a_ = float(beta[0])
            self.calib_b_ = float(beta[1])
        except np.linalg.LinAlgError:
            self.calib_a_ = float(np.mean(tgt_valid))
            self.calib_b_ = 1.0

    # ══════════════════════════════════════════════════════════════
    #  PREDICT
    # ══════════════════════════════════════════════════════════════

    def set_test_info(self, vix_current: np.ndarray,
                      returns_test: np.ndarray | None = None,
                      **kwargs) -> None:
        self._vix_current = np.asarray(vix_current, dtype=float)
        if returns_test is not None:
            self._test_returns_pct = (
                np.asarray(returns_test, dtype=float) * 100.0
            )
        else:
            self._test_returns_pct = None

    def set_vix(self, vix_current: np.ndarray) -> None:
        """Legacy interface."""
        self._vix_current = np.asarray(vix_current, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        if self._vix_current is None:
            raise RuntimeError("Call set_test_info() before predict().")
        return self._predict_core()

    def predict_with_vix(self, X: np.ndarray,
                         vix_current: np.ndarray) -> np.ndarray:
        """Legacy interface — wraps set_test_info + predict."""
        self.set_test_info(vix_current=vix_current)
        return self.predict(X)

    def _predict_core(self) -> np.ndarray:
        """
        Change-based forecast with Mincer-Zarnowitz calibration.

        For each test day i:
          1. σ²_current is the filtered conditional variance at day i
          2. σ²_forecast is the h-step ahead conditional variance
          3. raw_signal = sqrt(252·σ²_forecast) - sqrt(252·σ²_current)
          4. pred = a + b · raw_signal  (calibrated on training)
          5. Update σ² using actual return (no look-ahead)
        """
        n = len(self._vix_current)
        omega = self.omega_
        alpha = self.alpha_
        beta = self.beta_
        ab = min(alpha + beta, 0.9999)
        h = self.horizon
        mu = self.mu_

        sigma2_bar = omega / max(1.0 - ab, 1e-8)
        preds = np.empty(n)

        # Initial σ² from training
        if self.last_eps2_ is not None:
            sigma2_current = (omega + alpha * self.last_eps2_
                              + beta * self.sigma2_T_)
        else:
            sigma2_current = omega + ab * self.sigma2_T_

        for i in range(n):
            # ── Current annualized vol ──
            vol_current = np.sqrt(
                252.0 * max(sigma2_current, 1e-8)
            )

            # ── h-step ahead forecast ──
            # 1-step: σ²_{t+1|t} = σ²_current (already computed)
            sigma2_next = sigma2_current

            if h == 1:
                avg_sigma2 = sigma2_next
            elif abs(1.0 - ab) < 1e-8:
                avg_sigma2 = sigma2_next
            else:
                geom = (1.0 - ab ** h) / (1.0 - ab)
                avg_sigma2 = (sigma2_bar
                              + (sigma2_next - sigma2_bar) * geom / h)

            vol_forecast = np.sqrt(252.0 * max(avg_sigma2, 1e-8))

            # ── Raw signal: change in GARCH vol ──
            raw_signal = vol_forecast - vol_current

            # ── Calibrated prediction ──
            preds[i] = self.calib_a_ + self.calib_b_ * raw_signal

            # ── Filter σ² with actual test return (FIX 1) ──
            if (self._test_returns_pct is not None
                    and i < len(self._test_returns_pct)):
                eps2 = (self._test_returns_pct[i] - mu) ** 2
                sigma2_current = omega + alpha * eps2 + beta * sigma2_current
            else:
                sigma2_current = omega + ab * sigma2_current

            sigma2_current = max(sigma2_current, 1e-8)

        return preds