from __future__ import annotations

import numpy as np
from scipy import stats


def _newey_west_variance(x: np.ndarray, n_lags: int) -> float:
    """
    Newey-West HAC variance of the sample mean of x.
    """
    T = len(x)
    x_dm = x - x.mean()
    gamma_0 = float(np.dot(x_dm, x_dm) / T)
    lrv = gamma_0

    for lag in range(1, n_lags + 1):
        if lag >= T:
            break
        weight = 1.0 - lag / (n_lags + 1)
        gamma_j = float(np.dot(x_dm[lag:], x_dm[:-lag]) / T)
        lrv += 2.0 * weight * gamma_j

    lrv = max(lrv, 1e-12)
    return lrv / T


def diebold_mariano_hln(
    e1: np.ndarray,
    e2: np.ndarray,
    horizon: int = 1,
    loss: str = "mse",
) -> dict:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold correction.
    H0: equal predictive accuracy. Two-sided.
    """
    from .losses import loss_differential

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    T = len(e1)

    if T < 10:
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "d_bar": float("nan"),
            "T": T,
        }

    d = loss_differential(e1, e2, loss=loss)
    d_bar = float(d.mean())

    nw_lags = max(horizon - 1, 1)
    var_dbar = max(_newey_west_variance(d, nw_lags), 1e-12)

    dm_raw = d_bar / np.sqrt(var_dbar)

    hln_num = T + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / T
    hln_factor = np.sqrt(max(hln_num, 1.0) / T)
    dm_star = dm_raw * hln_factor

    p_value = float(2.0 * stats.t.sf(abs(dm_star), df=T - 1))

    return {
        "dm_stat": float(dm_star),
        "p_value": p_value,
        "d_bar": d_bar,
        "T": T,
        "nw_lags": nw_lags,
    }


def clark_west(
    e1: np.ndarray,
    e2: np.ndarray,
    yhat1: np.ndarray,
    yhat2: np.ndarray,
    horizon: int = 1,
) -> dict:
    """
    Clark-West test for nested model comparison.
    H0: restricted model is correctly specified. One-sided.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    yhat1 = np.asarray(yhat1, dtype=float)
    yhat2 = np.asarray(yhat2, dtype=float)
    T = len(e1)

    if T < 10:
        return {
            "cw_stat": float("nan"),
            "p_value": float("nan"),
            "f_bar": float("nan"),
        }

    f_t = e1**2 - (e2**2 - (yhat2 - yhat1) ** 2)
    f_bar = float(f_t.mean())

    nw_lags = max(horizon - 1, 1)
    var_fbar = max(_newey_west_variance(f_t, nw_lags), 1e-12)

    cw_stat = f_bar / np.sqrt(var_fbar)
    p_value = float(stats.norm.sf(cw_stat))

    return {
        "cw_stat": float(cw_stat),
        "p_value": p_value,
        "f_bar": f_bar,
    }


def mincer_zarnowitz(
    actual: np.ndarray,
    forecast: np.ndarray,
    hac_lags: int = 8,
) -> dict:
    """
    Mincer-Zarnowitz calibration regression.
    H0: (a, b) = (0, 1).
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    try:
        import statsmodels.api as sm

        X = sm.add_constant(forecast, prepend=True)
        results = sm.OLS(actual, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lags},
        )

        a_hat = float(results.params[0])
        b_hat = float(results.params[1])

        R = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        r = np.array([0.0, 1.0])
        wald = results.wald_test((R, r), scalar=True)

        return {
            "a_hat": a_hat,
            "b_hat": b_hat,
            "a_se": float(results.bse[0]),
            "b_se": float(results.bse[1]),
            "wald_stat": float(wald.statistic),
            "wald_p": float(wald.pvalue),
            "r2": float(results.rsquared),
        }

    except Exception:
        return {
            "a_hat": float("nan"),
            "b_hat": float("nan"),
            "a_se": float("nan"),
            "b_se": float("nan"),
            "wald_stat": float("nan"),
            "wald_p": float("nan"),
            "r2": float("nan"),
        }


def benjamini_hochberg(
    p_values: list | np.ndarray,
    q: float = 0.10,
) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    """
    p_arr = np.asarray(p_values, dtype=float)
    M = len(p_arr)

    if M == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_arr)
    p_sorted = p_arr[order]
    ranks = np.arange(1, M + 1)

    adj_sorted = p_sorted * M / ranks
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)

    adjusted = np.empty(M)
    adjusted[order] = adj_sorted

    return adjusted


def andrews_bandwidth(T: int) -> int:
    """
    Andrews (1991) data-dependent HAC bandwidth.
    """
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
