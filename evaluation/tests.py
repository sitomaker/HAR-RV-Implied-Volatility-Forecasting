"""
evaluation/tests.py — Statistical hypothesis tests for forecast comparison.

DM-HLN  (LaTeX eq. 16):
    d_t = L(e_{1,t}) - L(e_{2,t})
    HAC variance with Newey-West, max(h-1, 1) lags
    HLN correction: sqrt((T+1-2h+h(h-1)/T) / T)
    Statistic: DM* ~ t_{T-1}  (two-sided; sign: positive = model 2 better)

CW      (LaTeX eqs. 17-18):
    f_t = e1² - [e2² - (y_hat_2 - y_hat_1)²]
    SE via HAC (Newey-West, max(h-1,1) lags) for consistency with DM
    CW ~ N(0,1), one-sided (reject when CW > z_{1-alpha})

MZ      (LaTeX eq. 19):
    y_{t+h} = a + b * y_hat_{t+h} + u
    HAC Wald test, q* lags (Andrews 1991, eq. hac_lags)
    H0: (a, b) = (0, 1)

BH      (LaTeX §mht):
    Benjamini-Hochberg (1995) FDR correction at q_FDR = 0.10
    Applied simultaneously to all 32 primary p-values
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITY — NEWEY-WEST HAC VARIANCE
# ══════════════════════════════════════════════════════════════════════════════

def _newey_west_variance(x: np.ndarray, n_lags: int) -> float:
    """
    Newey-West HAC variance of the mean of x.

        Var_NW(x_bar) = (gamma_0 + 2 * sum_{j=1}^{q} w_j * gamma_j) / T²

    where w_j = 1 - j/(q+1) (Bartlett kernel) and gamma_j is the
    j-th sample autocovariance of x.

    Returns the HAC estimate of Var(x_bar) = V / T, where V is the
    long-run variance.
    """
    T        = len(x)
    x_dm     = x - x.mean()
    gamma_0  = float(np.dot(x_dm, x_dm) / T)
    lrv      = gamma_0

    for lag in range(1, n_lags + 1):
        if lag >= T:
            break
        weight  = 1.0 - lag / (n_lags + 1)         # Bartlett kernel
        gamma_j = float(np.dot(x_dm[lag:], x_dm[:-lag]) / T)
        lrv    += 2.0 * weight * gamma_j

    lrv = max(lrv, 1e-12)
    return lrv / T   # Var(x_bar) = long-run variance / T


# ══════════════════════════════════════════════════════════════════════════════
#  DIEBOLD-MARIANO WITH HLN CORRECTION  (LaTeX eq. 16)
# ══════════════════════════════════════════════════════════════════════════════

def diebold_mariano_hln(
    e1: np.ndarray,
    e2: np.ndarray,
    horizon: int = 1,
    loss: str = "mse",
) -> dict:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) correction.

        d_t = L(e1_t) - L(e2_t)
        DM* = d_bar / sqrt(Var_NW(d_bar)) * HLN_factor

    HLN finite-sample correction (LaTeX eq. 16):
        HLN = sqrt( (T+1-2h+h(h-1)/T) / T )

    HAC lags: max(h-1, 1).  Under H0: DM* ~ t_{T-1}.

    H0: E[d_t] = 0  (equal predictive accuracy).
    Two-sided test; positive DM* indicates model 2 beats model 1.

    Parameters
    ----------
    e1      : forecast errors from benchmark model
    e2      : forecast errors from competing model
    horizon : forecast horizon h
    loss    : 'mse' or 'mae'
    """
    from .losses import loss_differential

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    T  = len(e1)

    if T < 10:
        return {"dm_stat": float("nan"), "p_value": float("nan"),
                "d_bar": float("nan"), "T": T}

    d     = loss_differential(e1, e2, loss=loss)
    d_bar = float(d.mean())

    nw_lags  = max(horizon - 1, 1)
    var_dbar = _newey_west_variance(d, nw_lags)
    var_dbar = max(var_dbar, 1e-12)

    dm_raw = d_bar / np.sqrt(var_dbar)

    # HLN finite-sample correction
    hln_num    = T + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / T
    hln_factor = np.sqrt(max(hln_num, 1.0) / T)
    dm_star    = dm_raw * hln_factor

    # Two-sided p-value against t_{T-1}
    p_value = float(2.0 * stats.t.sf(abs(dm_star), df=T - 1))

    return {
        "dm_stat":  float(dm_star),
        "p_value":  p_value,
        "d_bar":    d_bar,
        "T":        T,
        "nw_lags":  nw_lags,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLARK-WEST TEST  (LaTeX eqs. 17-18)
# ══════════════════════════════════════════════════════════════════════════════

def clark_west(
    e1: np.ndarray,
    e2: np.ndarray,
    yhat1: np.ndarray,
    yhat2: np.ndarray,
    horizon: int = 1,
) -> dict:
    """
    Clark-West (2007) test for nested model comparison.

    Adjusted loss differential (LaTeX eq. 17):
        f_t = e1_t² - [ e2_t² - (y_hat_{2,t} - y_hat_{1,t})² ]

    The correction term removes the upward bias in the unrestricted
    model's MSE under H0.

    CW statistic (LaTeX eq. 18):
        CW = f_bar / SE_NW(f)  ~ N(0,1)

    SE uses Newey-West HAC with max(h-1, 1) lags for consistency with
    the DM implementation. One-sided test: reject H0 when CW > z_{1-alpha}.

    H0: restricted model is the correct specification.
    H1: unrestricted model has genuine predictive content.

    Parameters
    ----------
    e1      : forecast errors from restricted (nested) model
    e2      : forecast errors from unrestricted model
    yhat1   : forecasts from restricted model
    yhat2   : forecasts from unrestricted model
    horizon : forecast horizon h (for HAC lag selection)
    """
    e1    = np.asarray(e1,    dtype=float)
    e2    = np.asarray(e2,    dtype=float)
    yhat1 = np.asarray(yhat1, dtype=float)
    yhat2 = np.asarray(yhat2, dtype=float)
    T     = len(e1)

    if T < 10:
        return {"cw_stat": float("nan"), "p_value": float("nan"),
                "f_bar": float("nan")}

    f_t   = e1 ** 2 - (e2 ** 2 - (yhat2 - yhat1) ** 2)
    f_bar = float(f_t.mean())

    nw_lags  = max(horizon - 1, 1)
    var_fbar = _newey_west_variance(f_t, nw_lags)
    var_fbar = max(var_fbar, 1e-12)

    cw_stat = f_bar / np.sqrt(var_fbar)

    # One-sided p-value: upper tail of N(0,1)
    p_value = float(stats.norm.sf(cw_stat))

    return {
        "cw_stat": float(cw_stat),
        "p_value": p_value,
        "f_bar":   f_bar,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MINCER-ZARNOWITZ CALIBRATION  (LaTeX eq. 19)
# ══════════════════════════════════════════════════════════════════════════════

def mincer_zarnowitz(
    actual: np.ndarray,
    forecast: np.ndarray,
    hac_lags: int = 8,
) -> dict:
    """
    Mincer-Zarnowitz calibration regression (LaTeX eq. 19):

        y_{t+h} = a + b * y_hat_{t+h} + u_t

    H0: (a, b) = (0, 1)  — well-calibrated forecast.
    Wald statistic with Newey-West HAC standard errors.
    Default hac_lags = q* ≈ 8 (Andrews 1991, LaTeX eq. hac_lags, T≈3,773).

    Interpretation of deviations from H0:
        b_hat < 1 : forecast too dispersed (overconfident)
        b_hat > 1 : forecast too conservative (underconfident)
        a_hat ≠ 0 : unconditional bias

    Returns
    -------
    dict with a_hat, b_hat, their HAC SEs, Wald statistic and p-value,
    and regression R².
    """
    actual   = np.asarray(actual,   dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    try:
        import statsmodels.api as sm

        X       = sm.add_constant(forecast, prepend=True)
        results = sm.OLS(actual, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lags},
        )

        a_hat = float(results.params[0])
        b_hat = float(results.params[1])

        # Wald test H0: (a, b) = (0, 1)
        # R @ beta = r  →  [[1,0],[0,1]] @ [a,b] = [0,1]
        R = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        r = np.array([0.0, 1.0])
        wald = results.wald_test((R, r), scalar=True)

        return {
            "a_hat":     a_hat,
            "b_hat":     b_hat,
            "a_se":      float(results.bse[0]),
            "b_se":      float(results.bse[1]),
            "wald_stat": float(wald.statistic),
            "wald_p":    float(wald.pvalue),
            "r2":        float(results.rsquared),
        }

    except Exception as exc:
        print(f"  MZ regression failed: {exc}")
        return {k: float("nan") for k in
                ("a_hat", "b_hat", "a_se", "b_se", "wald_stat", "wald_p", "r2")}


# ══════════════════════════════════════════════════════════════════════════════
#  BENJAMINI-HOCHBERG FDR CORRECTION  (LaTeX §mht)
# ══════════════════════════════════════════════════════════════════════════════

def benjamini_hochberg(
    p_values: list | np.ndarray,
    q: float = 0.10,
) -> np.ndarray:
    """
    Benjamini-Hochberg (1995) step-up FDR correction at level q.

    Applied simultaneously to all M primary p-values (LaTeX §mht:
    M = 32 tests). Under positive dependence (PRDS), BH controls FDR
    at exactly q (Benjamini & Yekutieli 2001).

    Algorithm:
        1. Sort p-values: p_(1) ≤ ... ≤ p_(M).
        2. Find k* = max{ i : p_(i) ≤ i * q / M }.
        3. Reject H_(1), ..., H_(k*).
        4. Adjusted p-value: p_adj_(i) = min_{j≥i} p_(j) * M / j,
           clipped to [0, 1].

    Parameters
    ----------
    p_values : raw p-values (length M)
    q        : FDR level (default 0.10)

    Returns
    -------
    adjusted : array of BH-adjusted p-values in the original order.
               Reject H_i if adjusted[i] ≤ q.
    """
    p_arr = np.asarray(p_values, dtype=float)
    M     = len(p_arr)

    if M == 0:
        return np.array([], dtype=float)

    # Sort indices by p-value (ascending)
    order     = np.argsort(p_arr)          # original indices in sorted order
    p_sorted  = p_arr[order]               # p_(1) ≤ p_(2) ≤ ... ≤ p_(M)
    ranks     = np.arange(1, M + 1)        # 1, 2, ..., M

    # Adjusted p-values in sorted order: p_(i) * M / i
    adj_sorted = p_sorted * M / ranks

    # Enforce monotonicity: step-up means take cumulative min from the right
    # adj_(i) = min(adj_(i), adj_(i+1), ..., adj_(M))
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)

    # Map back to original order
    adjusted             = np.empty(M)
    adjusted[order]      = adj_sorted

    return adjusted


# ══════════════════════════════════════════════════════════════════════════════
#  HAC BANDWIDTH  (Andrews 1991, LaTeX eq. hac_lags)
# ══════════════════════════════════════════════════════════════════════════════

def andrews_bandwidth(T: int) -> int:
    """
    Andrews (1991) data-dependent HAC bandwidth.

        q* = floor( 4 * (T/100)^(2/9) )

    For T = 3,773: q* ≈ 8.
    For T = 500:   q* ≈ 6.
    """
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))