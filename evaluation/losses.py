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
    actual_var    = np.asarray(actual_var,    dtype=float)
    predicted_var = np.maximum(
        np.asarray(predicted_var, dtype=float), _QLIKE_FLOOR
    )
    ratio = actual_var / predicted_var
    return float(np.mean(ratio - np.log(ratio) - 1))


def loss_differential(e1: np.ndarray,
                      e2: np.ndarray,
                      loss: str = "mse") -> np.ndarray:
   
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    if loss == "mse":
        return e1 ** 2 - e2 ** 2
    elif loss == "mae":
        return np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss {loss!r}. Use 'mse' or 'mae'.")
