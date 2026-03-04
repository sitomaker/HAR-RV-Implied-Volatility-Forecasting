"""
evaluation/losses.py — Loss functions for forecast evaluation.

Primary loss functions (LaTeX eqs. 11-13):
    MSE   = (1/T) * sum (y - y_hat)²
    MAE   = (1/T) * sum |y - y_hat|
    QLIKE = (1/T) * sum [sigma²/sigma_hat² - log(sigma²/sigma_hat²) - 1]

QLIKE (Patton 2011) is robust to noise in the realized variance proxy,
making it appropriate for GARCH evaluation where the proxy is OHLC-based RV.
The floor epsilon = 1e-8 on predicted variance follows LaTeX §losses.
"""
from __future__ import annotations

import numpy as np

_QLIKE_FLOOR = 1e-8   # LaTeX §losses: clip predicted_var at this floor


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Squared Error."""
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean((actual - predicted) ** 2))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(actual, predicted)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def qlike(actual_var: np.ndarray, predicted_var: np.ndarray) -> float:
    """
    QLIKE loss for variance-level forecasts (Patton 2011, LaTeX eq. 13).

        QLIKE = (1/T) * sum [ sigma² / sigma_hat² - log(sigma² / sigma_hat²) - 1 ]

    Robust to noise in the volatility proxy under mild conditions.
    Any predicted_var <= 0 is clipped to _QLIKE_FLOOR = 1e-8 for
    numerical stability (LaTeX §losses).

    Parameters
    ----------
    actual_var    : realized variance proxy sigma²_{t+h}  (e.g. RV_d²)
    predicted_var : GARCH conditional variance forecast sigma_hat²_{t+h}
    """
    actual_var    = np.asarray(actual_var,    dtype=float)
    predicted_var = np.maximum(
        np.asarray(predicted_var, dtype=float), _QLIKE_FLOOR
    )
    ratio = actual_var / predicted_var
    return float(np.mean(ratio - np.log(ratio) - 1))


def loss_differential(e1: np.ndarray,
                      e2: np.ndarray,
                      loss: str = "mse") -> np.ndarray:
    """
    Element-wise loss differential d_t = L(e1_t) - L(e2_t).

    Used by the DM test (LaTeX eq. 16). Positive values indicate
    e1 has higher loss than e2 (i.e., model 2 is better than model 1).

    Parameters
    ----------
    e1   : forecast errors from benchmark model
    e2   : forecast errors from competing model
    loss : 'mse' or 'mae'
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    if loss == "mse":
        return e1 ** 2 - e2 ** 2
    elif loss == "mae":
        return np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss {loss!r}. Use 'mse' or 'mae'.")