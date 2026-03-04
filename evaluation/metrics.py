from __future__ import annotations

import numpy as np
from scipy import stats


#  OUT-OF-SAMPLE R²  (LaTeX eq. 14)
def oos_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    ss_res   = float(np.sum((actual - predicted) ** 2))
    ss_bench = float(np.sum(actual ** 2))

    if ss_bench == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_bench


def oos_r2_bootstrap_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    horizon: int = 1,
    n_boot: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n         = len(actual)
    b         = max(2 * horizon, 10)
    rng       = np.random.default_rng(seed)

    r2_point  = oos_r2(actual, predicted)
    r2_boot   = np.empty(n_boot)

    # Build list of block start indices
    n_blocks = int(np.ceil(n / b))

    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx    = np.concatenate([
            np.arange(s, min(s + b, n)) for s in starts
        ])[:n]
        r2_boot[i] = oos_r2(actual[idx], predicted[idx])

    alpha     = 1.0 - confidence
    ci_lower  = float(np.percentile(r2_boot, 100 * alpha / 2))
    ci_upper  = float(np.percentile(r2_boot, 100 * (1 - alpha / 2)))

    return {
        "r2":       r2_point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se_boot":  float(np.std(r2_boot, ddof=1)),
    }


#  DIRECTIONAL ACCURACY
def hit_rate(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n         = len(actual)

    if n == 0:
        return float("nan")

    # M0 special case: all predictions are zero
    if np.all(predicted == 0.0):
        p_up = float(np.mean(actual > 0))
        return max(p_up, 1.0 - p_up)

    # General case: correct sign matches, ties in predicted excluded from numerator but not denominator (conservative)
    correct = np.sum(
        (np.sign(actual) == np.sign(predicted)) & (predicted != 0.0)
    )
    return float(correct / n)


def pesaran_timmermann(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n         = len(actual)

    if n < 10:
        return {"pt_stat": float("nan"), "p_value": float("nan")}

    p_y    = float(np.mean(actual    > 0))
    p_yhat = float(np.mean(predicted > 0))

    # Expected hit rate under independence
    p_star = p_y * p_yhat + (1.0 - p_y) * (1.0 - p_yhat)
    p_hat  = hit_rate(actual, predicted)

    # Asymptotic variance of p_star
    v1 = p_star * (1.0 - p_star) / n
    v2 = (2.0 * p_y    - 1.0) ** 2 * p_yhat * (1.0 - p_yhat) / n
    v3 = (2.0 * p_yhat - 1.0) ** 2 * p_y    * (1.0 - p_y)    / n
    var_p_star = v1 + v2 + v3

    if var_p_star <= 0.0:
        return {"pt_stat": float("nan"), "p_value": float("nan")}

    pt_stat = (p_hat - p_star) / np.sqrt(var_p_star)
    p_value = float(1.0 - stats.norm.cdf(pt_stat))

    return {"pt_stat": float(pt_stat), "p_value": p_value}


#  REGIME-CONDITIONAL METRICS 
def regime_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    regime: np.ndarray,
) -> dict:
    from .losses import mse as mse_fn, mae as mae_fn

    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    regime    = np.asarray(regime)

    results = {}
    for k in ("Low", "Medium", "High"):
        mask = regime == k
        n_k  = int(mask.sum())
        if n_k == 0:
            results[k] = {"n": 0, "r2_oos": float("nan"),
                          "mse": float("nan"), "mae": float("nan"),
                          "hit_rate": float("nan")}
            continue

        a_k = actual[mask]
        p_k = predicted[mask]
        results[k] = {
            "n":        n_k,
            "r2_oos":   oos_r2(a_k, p_k),
            "mse":      mse_fn(a_k, p_k),
            "mae":      mae_fn(a_k, p_k),
            "hit_rate": hit_rate(a_k, p_k),
        }

    return results
