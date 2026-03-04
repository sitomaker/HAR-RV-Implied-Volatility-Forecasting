"""
evaluation/walk_forward.py — Expanding-window walk-forward evaluator.

Implements the strict protocol (LaTeX eq. wf):

    Fold l: [1, T_min + (l-1)*tau - h]  →purge h days→  [T_min + (l-1)*tau,
                                                           T_min + l*tau - 1]

    T_min = 500,  tau = 63.

Anti-leak discipline (LaTeX §walkforward checklist):
    1. Feature standardisation: mu, sigma from training rows only;
       features with sigma < 1e-8 left unstandardised.
    2. Ridge lambda: inner expanding-window CV within training set only.
       Each inner fold re-standardises independently.
    3. GARCH parameters: re-estimated on training returns at every fold.
    4. Regime thresholds: fixed ex ante, never updated.
    5. No feature selection or transformation uses test-period info.
    6. Training targets truncated h days before the test period (purge gap).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    WF_T_MIN, WF_TAU, RIDGE_LAMBDAS, INNER_CV_FOLDS,
    FEATURES_M2, FEATURES_M3, FEATURES_M4, FEATURES_M5_BASE,
    HORIZONS, DATA_PROC,
)
from models.random_walk  import RandomWalk
from models.garch        import GARCHModel
from models.har_ols      import HAROLS
from models.shar         import SHAR
from models.har_extended import HARExtended
from models.har_regime   import HARRegime

from evaluation.losses  import mse as mse_fn, rmse as rmse_fn, mae as mae_fn
from evaluation.metrics import (
    oos_r2, oos_r2_bootstrap_ci,
    hit_rate, pesaran_timmermann,
    regime_metrics,
)
from evaluation.tests import (
    diebold_mariano_hln, clark_west, mincer_zarnowitz,
    benjamini_hochberg, andrews_bandwidth,
)


# ── Constants ──────────────────────────────────────────────────────────────

_STD_FLOOR = 1e-8   # features with std < this are left unstandardised


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE STANDARDISATION
# ══════════════════════════════════════════════════════════════════════════════

def _standardise(X_train: np.ndarray,
                 X_test:  np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardise X_test using mean and std computed on X_train only.
    Features with std < _STD_FLOOR are passed through unchanged.
    """
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0, ddof=1)
    sig = np.where(sig < _STD_FLOOR, 1.0, sig)   # avoid division by zero

    return (X_train - mu) / sig, (X_test - mu) / sig


# ══════════════════════════════════════════════════════════════════════════════
#  INNER CV FOR RIDGE λ  (LaTeX eq. innercv)
# ══════════════════════════════════════════════════════════════════════════════

def inner_cv_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambdas: list = RIDGE_LAMBDAS,
    n_inner: int  = INNER_CV_FOLDS,
    horizon: int  = 1,
) -> tuple[float, list[float]]:
    """
    Inner expanding-window CV to select Ridge λ (LaTeX eq. innercv).

        λ*_l = argmin_{λ ∈ Λ} (1/K) * sum_{k=1}^K MSE_val(λ, inner fold k)

    Each inner fold re-standardises its own train/val split independently
    (no leakage from outer fold statistics into inner folds).
    The purge gap of h days is respected within inner folds.

    Parameters
    ----------
    X_train : outer training features (NOT pre-standardised)
    y_train : outer training targets
    lambdas : grid Λ (default from config)
    n_inner : K inner folds (default from config)
    horizon : h, used for the inner purge gap

    Returns
    -------
    best_lambda  : selected λ*
    all_cv_mses  : list of mean CV-MSE per lambda (for diagnostics)
    """
    n          = len(X_train)
    T_inner    = max(100, n // (n_inner + 1))
    tau_inner  = max(1, (n - T_inner) // n_inner)

    if tau_inner < 5:
        return float(lambdas[len(lambdas) // 2]), []

    best_lambda = lambdas[0]
    best_mse    = np.inf
    all_cv_mses = []

    for lam in lambdas:
        fold_mses = []
        for k in range(n_inner):
            train_end = T_inner + k * tau_inner
            val_start = train_end + horizon          # purge gap = h
            val_end   = min(val_start + tau_inner, n)

            if val_start >= n or val_end > n or val_start >= val_end:
                break

            X_tr_raw = X_train[:train_end]
            y_tr     = y_train[:train_end]
            X_va_raw = X_train[val_start:val_end]
            y_va     = y_train[val_start:val_end]

            # Independent standardisation per inner fold
            X_tr_sc, X_va_sc = _standardise(X_tr_raw, X_va_raw)

            mdl  = Ridge(alpha=lam, fit_intercept=True)
            mdl.fit(X_tr_sc, y_tr)
            pred = mdl.predict(X_va_sc)
            fold_mses.append(float(np.mean((y_va - pred) ** 2)))

        avg_mse = float(np.mean(fold_mses)) if fold_mses else np.inf
        all_cv_mses.append(avg_mse)

        if avg_mse < best_mse:
            best_mse    = avg_mse
            best_lambda = lam

    return best_lambda, all_cv_mses


# ══════════════════════════════════════════════════════════════════════════════
#  GENERIC WALK-FORWARD  (M0, M2–M5)
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(
    features:     pd.DataFrame,
    target_col:   str,
    feature_cols: list[str],
    model_name:   str,
    horizon:      int,
    model_class,
    model_kwargs: dict | None = None,
    use_ridge:    bool = False,
    is_regime:    bool = False,
    T_min:        int  = WF_T_MIN,
    tau:          int  = WF_TAU,
) -> dict:
    """
    Expanding-window walk-forward for linear models (M0, M2–M5).

    Purge gap: the last `horizon` rows of each training fold are excluded
    from model fitting because their targets are not yet realised
    (Assumption realtime, LaTeX eq. wf). This prevents target leakage
    from overlapping h=5 windows.

    For each fold:
        1. Strict temporal split with purge gap of h days.
        2. Standardise features on training rows only (_STD_FLOOR guard).
        3. If use_ridge: select λ via inner CV (re-standardising per fold).
        4. Fit model on purged, standardised training set.
        5. Predict on standardised test set.

    Parameters
    ----------
    features     : full feature DataFrame (DatetimeIndex)
    target_col   : 'y1' or 'y5'
    feature_cols : list of feature column names
    model_name   : string label for results dict
    horizon      : h ∈ {1, 5}
    model_class  : one of {RandomWalk, HAROLS, SHAR, HARExtended, HARRegime}
    model_kwargs : extra kwargs passed to model_class constructor
    use_ridge    : if True, select λ via inner CV (M4, M5)
    is_regime    : if True, pass regime column to fit/predict (M5)
    T_min        : minimum training observations
    tau          : test block size
    """
    if model_kwargs is None:
        model_kwargs = {}

    needed = feature_cols + [target_col, "VIX_level"]
    if is_regime:
        needed.append("regime")
    missing = [c for c in needed if c not in features.columns]
    if missing:
        raise ValueError(
            f"walk_forward [{model_name}]: missing columns {missing}."
        )

    valid = features[needed].dropna().copy()
    n     = len(valid)

    predictions  : list[float] = []
    actuals      : list[float] = []
    vix_levels   : list[float] = []
    regimes_oos  : list[str]   = []
    fold_info    : list[dict]  = []
    lambda_trace : list[float] = []

    n_folds = max(1, (n - T_min) // tau)
    pbar    = tqdm(total=n_folds, desc=f"  {model_name} h={horizon}", leave=True)
    cutoff  = T_min

    while cutoff < n:
        test_end = min(cutoff + tau, n)
        test     = valid.iloc[cutoff:test_end]
        if len(test) == 0:
            break

        # ── Purge gap: exclude last h rows from training targets ──────
        # Rows valid.iloc[cutoff-h : cutoff] have unobserved targets.
        train_end_purged = max(cutoff - horizon, 0)
        train = valid.iloc[:train_end_purged]

        if len(train) < 10:
            cutoff = test_end
            pbar.update(1)
            continue

        y_train = train[target_col].values
        y_test  = test[target_col].values

        X_train_raw = train[feature_cols].values
        X_test_raw  = test[feature_cols].values

        # ── Standardise on training set only ─────────────────────────
        X_train_sc, X_test_sc = _standardise(X_train_raw, X_test_raw)

        # ── Select λ via inner CV (Ridge models) ─────────────────────
        if use_ridge:
            lam, cv_mses = inner_cv_lambda(
                X_train_raw, y_train,
                horizon=horizon,
            )
            kwargs = dict(model_kwargs, alpha=lam)
            lambda_trace.append(lam)
        else:
            kwargs = model_kwargs
            lam    = None

        # ── Instantiate, fit, predict ─────────────────────────────────
        model = model_class(**kwargs)

        if is_regime:
            X_tr_df = pd.DataFrame(X_train_sc, columns=feature_cols,
                                   index=train.index)
            X_te_df = pd.DataFrame(X_test_sc,  columns=feature_cols,
                                   index=test.index)
            model.fit(X_tr_df, y_train, train["regime"])
            preds = model.predict(X_te_df, test["regime"])
        else:
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)

        predictions.extend(preds.tolist())
        actuals.extend(y_test.tolist())
        vix_levels.extend(test["VIX_level"].values.tolist())

        if is_regime or "regime" in valid.columns:
            regimes_oos.extend(test["regime"].values.tolist())

        train_preds = model.predict(X_train_sc) if not is_regime else \
                      model.predict(
                          pd.DataFrame(X_train_sc, columns=feature_cols,
                                       index=train.index),
                          train["regime"]
                      )

        fold_info.append({
            "fold_start":         str(test.index[0].date()),
            "fold_end":           str(test.index[-1].date()),
            "train_size":         train_end_purged,
            "test_size":          len(test),
            "lambda_selected":    lam,
            "train_predictions":  train_preds.tolist(),
        })

        cutoff = test_end
        pbar.update(1)

    pbar.close()

    preds_arr  = np.array(predictions)
    actual_arr = np.array(actuals)

    result: dict = {
        "model":        model_name,
        "horizon":      horizon,
        "predictions":  preds_arr,
        "actuals":      actual_arr,
        "errors":       actual_arr - preds_arr,
        "vix_levels":   np.array(vix_levels),
        "folds":        fold_info,
        "n_oos":        len(preds_arr),
    }

    if regimes_oos:
        result["regimes"] = np.array(regimes_oos)

    if lambda_trace:
        result["lambda_median"] = float(np.median(lambda_trace))
        result["lambda_iqr"]    = float(
            np.percentile(lambda_trace, 75) - np.percentile(lambda_trace, 25)
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  GARCH WALK-FORWARD  (M1 — uses returns, not features)
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_garch(
    features:   pd.DataFrame,
    target_col: str,
    horizon:    int,
    T_min:      int = WF_T_MIN,
    tau:        int = WF_TAU,
) -> dict:
    """Walk-forward for GARCH(1,1) — change-based forecast with MZ calibration."""

    if "VIX_level" not in features.columns:
        raise ValueError("walk_forward_garch: 'VIX_level' column missing.")

    master       = pd.read_parquet(DATA_PROC / "master.parquet")
    returns_full = master["SPY_ret"].dropna()

    needed     = [target_col, "VIX_level"]
    valid      = features[needed].dropna().copy()
    common_idx = valid.index.intersection(returns_full.index)
    valid      = valid.loc[common_idx]
    ret_series = returns_full.loc[common_idx]

    n           = len(valid)
    predictions : list[float] = []
    actuals     : list[float] = []
    vix_oos     : list[float] = []
    fold_info   : list[dict]  = []

    n_folds = max(1, (n - T_min) // tau)
    pbar    = tqdm(total=n_folds,
                   desc=f"  M1_GARCH h={horizon}", leave=True)
    cutoff  = T_min

    while cutoff < n:
        test_end = min(cutoff + tau, n)
        test     = valid.iloc[cutoff:test_end]
        if len(test) == 0:
            break

        train_end_purged = max(cutoff - horizon, 0)

        # Training data
        train_returns = ret_series.iloc[:train_end_purged].values
        train_y       = valid[target_col].iloc[:train_end_purged].values

        # Purge gap returns (for σ² propagation)
        purge_rets_pct = (
            ret_series.iloc[train_end_purged:cutoff].values * 100.0
        )

        # Test data
        test_returns = ret_series.iloc[cutoff:test_end].values
        vix_current  = test["VIX_level"].values

        if len(train_returns) < 50:
            cutoff = test_end
            pbar.update(1)
            continue

        # ── 1. Fit + calibrate ────────────────────────────────
        garch = GARCHModel(horizon=horizon)
        garch.fit(
            X       = np.zeros((len(train_returns), 1)),
            y       = train_y,
            returns = train_returns,
        )

        # ── 2. Propagate σ² through purge gap ─────────────────
        sigma2 = garch.sigma2_T_
        for r_pct in purge_rets_pct:
            eps2   = (r_pct - garch.mu_) ** 2
            sigma2 = (garch.omega_
                      + garch.alpha_ * eps2
                      + garch.beta_  * sigma2)
            sigma2 = max(sigma2, 1e-8)

        garch.sigma2_T_ = sigma2
        if len(purge_rets_pct) > 0:
            garch.last_eps2_ = (purge_rets_pct[-1] - garch.mu_) ** 2

        # ── 3. Set test info + predict ────────────────────────
        garch.set_test_info(
            vix_current  = vix_current,
            returns_test = test_returns,
        )
        preds = garch.predict(np.zeros((len(test), 1)))

        predictions.extend(preds.tolist())
        actuals.extend(test[target_col].values.tolist())
        vix_oos.extend(vix_current.tolist())

        fold_info.append({
            "fold_start":  str(test.index[0].date()),
            "fold_end":    str(test.index[-1].date()),
            "train_size":  train_end_purged,
            "test_size":   len(test),
            "calib_a":     float(garch.calib_a_),
            "calib_b":     float(garch.calib_b_),
        })

        cutoff = test_end
        pbar.update(1)

    pbar.close()

    preds_arr  = np.array(predictions)
    actual_arr = np.array(actuals)

    return {
        "model":       "M1_GARCH",
        "horizon":     horizon,
        "predictions": preds_arr,
        "actuals":     actual_arr,
        "errors":      actual_arr - preds_arr,
        "vix_levels":  np.array(vix_oos),
        "folds":       fold_info,
        "n_oos":       len(preds_arr),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER RUNNER — ALL MODELS × ALL HORIZONS
# ══════════════════════════════════════════════════════════════════════════════

def run_all_models(features_path: str | Path | None = None) -> dict:
    """Run walk-forward for all six models at both horizons."""
    if features_path is None:
        features_path = DATA_PROC / "features_gk.parquet"

    features = pd.read_parquet(features_path)

    print("=" * 60)
    print("WALK-FORWARD EVALUATION  —  HAR-RV IV Forecasting")
    print(f"T_min={WF_T_MIN},  tau={WF_TAU},  horizons={HORIZONS}")
    print("=" * 60)

    all_results: dict = {}

    for horizon in HORIZONS:
        target_col = f"y{horizon}"
        print(f"\n{'='*50}\nHORIZON h = {horizon}\n{'='*50}")
        h_results: dict = {}

        # M0 — RandomWalk
        print("\nM0: Random Walk")
        h_results["M0_RandomWalk"] = walk_forward(
            features, target_col, ["VIX_level"],
            "M0_RandomWalk", horizon, RandomWalk,
        )

        # M1 — GARCH(1,1)
        print("\nM1: GARCH(1,1)")
        try:
            h_results["M1_GARCH"] = walk_forward_garch(
                features, target_col, horizon
            )
        except Exception as exc:
            print(f"  GARCH failed: {exc}")

        # M2 — HAR-OLS
        print("\nM2: HAR-OLS")
        h_results["M2_HAROLS"] = walk_forward(
            features, target_col, FEATURES_M2,
            "M2_HAROLS", horizon, HAROLS,
        )

        # M3 — SHAR
        print("\nM3: SHAR")
        h_results["M3_SHAR"] = walk_forward(
            features, target_col, FEATURES_M3,
            "M3_SHAR", horizon, SHAR,
        )

        # M4 — HAR-Extended (Ridge)
        print("\nM4: HAR-Extended (Ridge)")
        h_results["M4_HARExtended"] = walk_forward(
            features, target_col, FEATURES_M4,
            "M4_HARExtended", horizon, HARExtended,
            use_ridge=True,
        )

        # M5 — HAR-Regime (Ridge + interactions)
        print("\nM5: HAR-Regime (Ridge)")
        h_results["M5_HARRegime"] = walk_forward(
            features, target_col, FEATURES_M5_BASE,
            "M5_HARRegime", horizon, HARRegime,
            use_ridge=True, is_regime=True,
        )

        all_results[horizon] = h_results

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS + FULL TEST BATTERY
# ══════════════════════════════════════════════════════════════════════════════

# Full test battery (LaTeX tab:test_battery): 8 pairs × 2 horizons × 2 losses
# = 32 p-values subjected simultaneously to BH correction.
_CW_PAIRS = [
    # (restricted, unrestricted)
    ("M0_RandomWalk",  "M2_HAROLS"),
    ("M0_RandomWalk",  "M3_SHAR"),
    ("M0_RandomWalk",  "M4_HARExtended"),
    ("M2_HAROLS",      "M3_SHAR"),
    ("M2_HAROLS",      "M4_HARExtended"),
]
_DM_PAIRS = [
    # (benchmark, model)
    ("M0_RandomWalk",  "M1_GARCH"),
    ("M1_GARCH",       "M4_HARExtended"),
    ("M4_HARExtended", "M5_HARRegime"),
]
_MZ_MODELS = ["M2_HAROLS", "M4_HARExtended"]


def compute_metrics(result: dict) -> dict:
    """Compute scalar performance metrics for one model-horizon result."""
    actual = result["actuals"]
    preds  = result["predictions"]
    h      = result["horizon"]

    pt = pesaran_timmermann(actual, preds)
    r2 = oos_r2_bootstrap_ci(actual, preds, horizon=h)

    m: dict = {
        "model":       result["model"],
        "horizon":     h,
        "n_oos":       result["n_oos"],
        "mse":         mse_fn(actual, preds),
        "rmse":        rmse_fn(actual, preds),
        "mae":         mae_fn(actual, preds),
        "r2_oos":      r2["r2"],
        "r2_ci_lo":    r2["ci_lower"],
        "r2_ci_hi":    r2["ci_upper"],
        "hit_rate":    hit_rate(actual, preds),
        "pt_stat":     pt["pt_stat"],
        "pt_pval":     pt["p_value"],
    }

    if "lambda_median" in result:
        m["lambda_median"] = result["lambda_median"]
        m["lambda_iqr"]    = result["lambda_iqr"]

    return m


def evaluate_all(all_results: dict) -> pd.DataFrame:
    """
    Compute all metrics and the full statistical test battery
    (LaTeX tab:test_battery) for all model-horizon pairs.

    BH correction is applied simultaneously to all 32 primary p-values
    (DM + CW combined) as specified in LaTeX §mht.

    Returns a DataFrame with one row per (model, horizon).
    """
    rows: list[dict] = []

    # Collect all p-values for joint BH correction
    all_pvals: list[float]  = []
    pval_keys: list[tuple]  = []   # (model, horizon, test_type)

    for horizon, h_res in all_results.items():
        hac_q = andrews_bandwidth(
            h_res[next(iter(h_res))]["n_oos"]
        )

        # ── Per-model metrics ─────────────────────────────────────────
        for model_name, result in h_res.items():
            row = compute_metrics(result)

            # Regime-conditional breakdown (§regime_perf)
            if "regimes" in result:
                rc = regime_metrics(
                    result["actuals"], result["predictions"],
                    result["regimes"],
                )
                for k, kmetrics in rc.items():
                    for mname, val in kmetrics.items():
                        row[f"{mname}_{k.lower()}"] = val

            # Placeholder columns for test results
            for col in ("dm_mse_stat", "dm_mse_pval",
                        "dm_mae_stat", "dm_mae_pval",
                        "cw_stat",     "cw_pval",
                        "mz_a", "mz_b", "mz_wald_p"):
                row[col] = float("nan")

            rows.append(row)

        # ── DM tests (LaTeX: non-nested pairs) ───────────────────────
        for bench_name, model_name in _DM_PAIRS:
            if bench_name not in h_res or model_name not in h_res:
                continue
            r_bench = h_res[bench_name]
            r_model = h_res[model_name]
            n_min   = min(r_bench["n_oos"], r_model["n_oos"])

            for loss in ("mse", "mae"):
                dm = diebold_mariano_hln(
                    r_bench["errors"][-n_min:],
                    r_model["errors"][-n_min:],
                    horizon=horizon, loss=loss,
                )
                stat_col = f"dm_{loss}_stat"
                pval_col = f"dm_{loss}_pval"
                for row in rows:
                    if row["model"] == model_name and row["horizon"] == horizon:
                        row[stat_col] = dm["dm_stat"]
                        row[pval_col] = dm["p_value"]

                if not np.isnan(dm["p_value"]):
                    all_pvals.append(dm["p_value"])
                    pval_keys.append((model_name, horizon, f"dm_{loss}"))

        # ── CW tests (LaTeX: nested pairs) ───────────────────────────
        for restricted, unrestricted in _CW_PAIRS:
            if restricted not in h_res or unrestricted not in h_res:
                continue
            r1    = h_res[restricted]
            r2    = h_res[unrestricted]
            n_min = min(r1["n_oos"], r2["n_oos"])

            cw = clark_west(
                r1["errors"][-n_min:],
                r2["errors"][-n_min:],
                r1["predictions"][-n_min:],
                r2["predictions"][-n_min:],
                horizon=horizon,
            )
            for row in rows:
                if row["model"] == unrestricted and row["horizon"] == horizon:
                    row["cw_stat"] = cw["cw_stat"]
                    row["cw_pval"] = cw["p_value"]

            if not np.isnan(cw["p_value"]):
                all_pvals.append(cw["p_value"])
                pval_keys.append((unrestricted, horizon, "cw"))

        # ── Mincer-Zarnowitz calibration ──────────────────────────────
        for mname in _MZ_MODELS:
            if mname not in h_res:
                continue
            result = h_res[mname]
            mz = mincer_zarnowitz(
                result["actuals"], result["predictions"],
                hac_lags=hac_q,
            )
            for row in rows:
                if row["model"] == mname and row["horizon"] == horizon:
                    row["mz_a"]      = mz["a_hat"]
                    row["mz_b"]      = mz["b_hat"]
                    row["mz_wald_p"] = mz["wald_p"]

    # ── Joint BH correction on all primary p-values (LaTeX §mht) ─────
    if all_pvals:
        bh_adj = benjamini_hochberg(all_pvals)
        df     = pd.DataFrame(rows)

        df["bh_pval"] = float("nan")
        for i, (model_name, horizon, test_type) in enumerate(pval_keys):
            mask = (df["model"] == model_name) & (df["horizon"] == horizon)
            # Store adjusted p-value in a test-specific column
            col = f"{test_type}_pval_bh"
            if col not in df.columns:
                df[col] = float("nan")
            df.loc[mask, col] = bh_adj[i]
    else:
        df = pd.DataFrame(rows)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from config import OUTPUT_DIR

    all_results = run_all_models()
    metrics_df  = evaluate_all(all_results)

    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE RESULTS SUMMARY")
    print("=" * 60)

    display_cols = [
        "horizon", "model", "n_oos",
        "rmse", "mae", "r2_oos", "r2_ci_lo", "r2_ci_hi",
        "hit_rate", "dm_mse_stat", "dm_mse_pval", "cw_stat", "cw_pval",
    ]
    avail = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[avail].to_string(index=False, float_format="%.4f"))

    out_csv = OUTPUT_DIR / "metrics_summary.csv"
    out_pkl = OUTPUT_DIR / "all_results.pkl"
    metrics_df.to_csv(out_csv, index=False)
    with open(out_pkl, "wb") as fh:
        pickle.dump(all_results, fh)

    print(f"\n  ✓ metrics_summary.csv  → {out_csv}")
    print(f"  ✓ all_results.pkl      → {out_pkl}")