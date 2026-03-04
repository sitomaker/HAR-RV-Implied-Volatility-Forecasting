"""
analysis/robustness.py — Robustness Checks.

Produces:
- Sub-period analysis (Table 8)
- Alternative VRP definitions (Table 9)
- Alternative RV estimators (Table 10)
- Alternative target specification (Table 11)
- Alternative training window sizes (Table 12)
- Ridge vs Lasso vs Elastic Net (Table 13)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROC, OUTPUT_DIR, TABLE_DIR,
    FEATURES_M4, WF_T_MIN, WF_TAU,
    TRADING_DAYS_YEAR, HAR_MONTHLY,
)


def load_features(estimator: str = "gk") -> pd.DataFrame:
    path = DATA_PROC / f"features_{estimator}.parquet"
    return pd.read_parquet(path)


def load_master() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROC / "master.parquet")


# ══════════════════════════════════════════════════════════════
#  UTILITY: QUICK WALK-FORWARD FOR ROBUSTNESS
# ══════════════════════════════════════════════════════════════

def quick_walk_forward(features: pd.DataFrame,
                       feature_cols: list,
                       target_col: str = "y5",
                       T_min: int = WF_T_MIN,
                       tau:   int = WF_TAU,
                       use_ridge: bool = True) -> dict:
    """
    Simplified walk-forward for robustness checks.

    Uses inner_cv_lambda for lambda selection (never hardcoded alpha).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LinearRegression
    from evaluation.walk_forward import inner_cv_lambda

    available = [c for c in feature_cols if c in features.columns]
    cols  = available + [target_col]
    valid = features[cols].dropna()

    n = len(valid)
    predictions, actuals = [], []
    cutoff = T_min

    while cutoff < n:
        test_end = min(cutoff + tau, n)
        train    = valid.iloc[:cutoff]
        test     = valid.iloc[cutoff:test_end]

        if len(test) == 0:
            break

        X_train = train[available].values
        y_train = train[target_col].values
        X_test  = test[available].values
        y_test  = test[target_col].values

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        if use_ridge:
            best_lam, _cv_mses = inner_cv_lambda(X_train_sc, y_train)
            model = Ridge(alpha=best_lam)
        else:
            model = LinearRegression()

        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)

        predictions.extend(preds.tolist())
        actuals.extend(y_test.tolist())
        cutoff = test_end

    preds_arr  = np.array(predictions)
    actual_arr = np.array(actuals)

    from evaluation.metrics import oos_r2, hit_rate
    from evaluation.losses  import rmse as rmse_fn, mse as mse_fn
    from evaluation.tests   import diebold_mariano_hln

    e_rw    = actual_arr            # RW errors (predict zero)
    e_model = actual_arr - preds_arr

    dm = diebold_mariano_hln(e_rw, e_model, horizon=5, loss="mse")

    return {
        "rmse":     rmse_fn(actual_arr, preds_arr),
        "mse":      mse_fn(actual_arr, preds_arr),
        "r2_oos":   oos_r2(actual_arr, preds_arr),
        "hit_rate": hit_rate(actual_arr, preds_arr),
        "dm_stat":  dm["dm_stat"],
        "dm_pval":  dm["p_value"],
        "n_oos":    len(preds_arr),
        "actuals":       actual_arr,
        "predictions":   preds_arr,
    }


# ══════════════════════════════════════════════════════════════
#  1. SUB-PERIOD ANALYSIS
# ══════════════════════════════════════════════════════════════

def subperiod_analysis(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute OOS metrics for each sub-period.
    NOTE: sub-period is applied to the OOS window, not the training
    window.  We run the full walk-forward once and then slice results.

    FIX: replaced the non-existent walk_forward_single() with the
    standard walk_forward() from evaluation.walk_forward using
    model_class=HARExtended.
    """
    print("\n  1. Sub-period analysis...")

    # FIX: import walk_forward (exists) instead of walk_forward_single (did not exist)
    from evaluation.walk_forward import walk_forward
    from models.har_extended import HARExtended
    from evaluation.metrics import oos_r2, hit_rate
    from evaluation.losses  import rmse as rmse_fn
    from evaluation.tests   import diebold_mariano_hln

    # Run full walk-forward for M4 using the correct function
    result = walk_forward(
        features,
        target_col="y5",
        feature_cols=FEATURES_M4,
        model_name="M4_sub",
        horizon=5,
        model_class=HARExtended,
        use_ridge=True,
    )

    # Reconstruct a DataFrame with dates for slicing
    valid   = features[FEATURES_M4 + ["y5", "VIX_level"]].dropna()
    oos_idx = valid.index[WF_T_MIN:]
    oos_idx = oos_idx[:len(result["actuals"])]

    oos_df = pd.DataFrame({
        "actual": result["actuals"],
        "pred":   result["predictions"],
    }, index=oos_idx)

    periods = {
        "Pre-COVID (2012-2019)":  ("2012-01-01", "2019-12-31"),
        "COVID (2020-2021)":      ("2020-01-01", "2021-12-31"),
        "Post-COVID (2022-2024)": ("2022-01-01", "2024-12-31"),
        "Full Sample":            (None, None),
    }

    rows = []
    for period_name, (start, end) in periods.items():
        sub = oos_df.loc[start:end] if start else oos_df
        if len(sub) < 10:
            rows.append({"Period": period_name, "RMSE": np.nan,
                         "R2_oos": np.nan, "DM_stat": np.nan,
                         "DM_pval": np.nan, "N_oos": 0})
            continue

        e_rw    = sub["actual"].values
        e_model = (sub["actual"] - sub["pred"]).values

        dm = diebold_mariano_hln(e_rw, e_model, horizon=5, loss="mse")

        rows.append({
            "Period":  period_name,
            "RMSE":    rmse_fn(sub["actual"].values, sub["pred"].values),
            "R2_oos":  oos_r2( sub["actual"].values, sub["pred"].values),
            "DM_stat": dm["dm_stat"],
            "DM_pval": dm["p_value"],
            "N_oos":   len(sub),
        })
        print(f"    {period_name}: R²oos={rows[-1]['R2_oos']:.4f}, "
              f"N={rows[-1]['N_oos']}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  2. ALTERNATIVE VRP DEFINITIONS
# ══════════════════════════════════════════════════════════════

def alternative_vrp(features: pd.DataFrame) -> pd.DataFrame:
    """Test robustness to VRP construction."""
    print("\n  2. Alternative VRP definitions...")

    master = load_master()
    vix    = master["VIX"].reindex(features.index)
    ret    = master["SPY_ret"].reindex(features.index)

    rows = []

    # Convention 1: baseline (VIX² - RV_m²) — already in features
    result_1 = quick_walk_forward(features, FEATURES_M4)
    rows.append({
        "VRP Definition": "Conv 1: VIX² - RV_22_GK² (baseline)",
        "R2_oos":  result_1["r2_oos"],
        "DM_pval": result_1["dm_pval"],
    })

    # Convention 2: VIX² - 252*r_t²  (1-day annualised proxy)
    feat_v2        = features.copy()
    feat_v2["VRP"] = vix ** 2 - 252 * (ret * 100) ** 2
    result_2 = quick_walk_forward(feat_v2, FEATURES_M4)
    rows.append({
        "VRP Definition": "Conv 2: VIX² - 252·(r_t·100)²",
        "R2_oos":  result_2["r2_oos"],
        "DM_pval": result_2["dm_pval"],
    })

    # Convention 3: VIX² - RV_22_CC²  (close-to-close)
    from features.pipeline import rv_close_to_close
    rv_cc_m = rv_close_to_close(ret, window=HAR_MONTHLY)
    feat_v3        = features.copy()
    feat_v3["VRP"] = vix ** 2 - rv_cc_m ** 2
    result_3 = quick_walk_forward(feat_v3, FEATURES_M4)
    rows.append({
        "VRP Definition": "Conv 3: VIX² - RV_22_CC²",
        "R2_oos":  result_3["r2_oos"],
        "DM_pval": result_3["dm_pval"],
    })

    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        print(f"    {r['VRP Definition']}: R²oos={r['R2_oos']:.4f}")
    return df


# ══════════════════════════════════════════════════════════════
#  3. ALTERNATIVE RV ESTIMATORS
# ══════════════════════════════════════════════════════════════

def alternative_rv() -> pd.DataFrame:
    """Test robustness to RV estimator choice."""
    print("\n  3. Alternative RV estimators...")

    rows = []
    for est in ["gk", "parkinson", "cc"]:
        from features.pipeline import build_all_features
        try:
            features = build_all_features(rv_estimator=est, verify_leak=False)
        except Exception as e:
            print(f"    {est} failed to build: {e}")
            continue

        result = quick_walk_forward(features, FEATURES_M4, use_ridge=True)
        rows.append({
            "RV Estimator": est.upper(),
            "RMSE":    result["rmse"],
            "R2_oos":  result["r2_oos"],
            "DM_pval": result["dm_pval"],
        })
        print(f"    {est}: R²oos={result['r2_oos']:.4f}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  4. ALTERNATIVE TARGET
# ══════════════════════════════════════════════════════════════

def alternative_target(features: pd.DataFrame) -> pd.DataFrame:
    """Test robustness to log vs arithmetic target."""
    print("\n  4. Alternative target specification...")

    master = load_master()
    vix    = master["VIX"].reindex(features.index)

    rows = []

    result_arith = quick_walk_forward(features, FEATURES_M4, target_col="y5")
    rows.append({
        "Target":  "ΔVIX (arithmetic, baseline)",
        "RMSE":    result_arith["rmse"],
        "R2_oos":  result_arith["r2_oos"],
        "DM_pval": result_arith["dm_pval"],
    })

    feat_log           = features.copy()
    feat_log["y5_log"] = np.log(vix.shift(-5) / vix)
    result_log = quick_walk_forward(
        feat_log, FEATURES_M4, target_col="y5_log"
    )
    rows.append({
        "Target":  "log(VIX_{t+5}/VIX_t)",
        "RMSE":    result_log["rmse"],
        "R2_oos":  result_log["r2_oos"],
        "DM_pval": result_log["dm_pval"],
    })

    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        print(f"    {r['Target']}: R²oos={r['R2_oos']:.4f}")
    return df


# ══════════════════════════════════════════════════════════════
#  5. TRAINING WINDOW SENSITIVITY
# ══════════════════════════════════════════════════════════════

def window_sensitivity(features: pd.DataFrame) -> pd.DataFrame:
    """Test sensitivity to minimum training window."""
    print("\n  5. Training window sensitivity...")

    rows = []
    for t_min in [250, 500, 750]:
        result = quick_walk_forward(features, FEATURES_M4, T_min=t_min)
        rows.append({
            "T_min":   t_min,
            "RMSE":    result["rmse"],
            "R2_oos":  result["r2_oos"],
            "DM_pval": result["dm_pval"],
            "N_oos":   result["n_oos"],
        })
        print(f"    T_min={t_min}: R²oos={result['r2_oos']:.4f}, "
              f"N_oos={result['n_oos']}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  6. REGULARISATION COMPARISON
# ══════════════════════════════════════════════════════════════

def regularization_comparison(features: pd.DataFrame) -> pd.DataFrame:
    """Compare Ridge vs Lasso vs Elastic Net vs OLS."""
    print("\n  6. Regularization comparison...")

    from sklearn.linear_model import (
        Ridge, Lasso, ElasticNet, LinearRegression,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    from evaluation.metrics import oos_r2
    from evaluation.losses  import rmse as rmse_fn

    available  = [c for c in FEATURES_M4 if c in features.columns]
    target_col = "y5"
    cols       = available + [target_col]
    valid      = features[cols].dropna()

    methods = {
        "OLS":         LinearRegression(),
        "Ridge":       Ridge(alpha=1.0),
        "Lasso":       Lasso(alpha=0.1, max_iter=10000),
        "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    }

    rows = []
    for method_name, model_template in methods.items():
        n = len(valid)
        predictions, actuals = [], []
        cutoff = WF_T_MIN

        while cutoff < n:
            test_end = min(cutoff + WF_TAU, n)
            train = valid.iloc[:cutoff]
            test  = valid.iloc[cutoff:test_end]
            if len(test) == 0:
                break

            X_train = train[available].values
            y_train = train[target_col].values
            X_test  = test[available].values
            y_test  = test[target_col].values

            scaler     = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)

            model = clone(model_template)
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)

            predictions.extend(preds.tolist())
            actuals.extend(y_test.tolist())
            cutoff = test_end

        preds_arr  = np.array(predictions)
        actual_arr = np.array(actuals)

        scaler_full  = StandardScaler()
        X_full       = scaler_full.fit_transform(valid[available].values)
        y_full       = valid[target_col].values
        model_full   = clone(model_template)
        model_full.fit(X_full, y_full)
        n_active = (int(np.sum(np.abs(model_full.coef_) > 1e-6))
                    if hasattr(model_full, "coef_") else len(available))

        rows.append({
            "Method":          method_name,
            "RMSE":            rmse_fn(actual_arr, preds_arr),
            "R2_oos":          oos_r2(actual_arr, preds_arr),
            "Active Features": int(n_active),
        })
        print(f"    {method_name}: R²oos={rows[-1]['R2_oos']:.4f}, "
              f"active={n_active}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  MASTER RUNNER
# ══════════════════════════════════════════════════════════════

def run_robustness():
    """Run all robustness checks."""
    print("=" * 60)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 60)

    features = load_features()
    results  = {}

    results["subperiods"] = subperiod_analysis(features)
    results["subperiods"].to_csv(TABLE_DIR / "robustness_subperiods.csv", index=False)

    results["vrp"] = alternative_vrp(features)
    results["vrp"].to_csv(TABLE_DIR / "robustness_vrp.csv", index=False)

    print("\n  [Rebuilding features for RV robustness...]")
    results["rv"] = alternative_rv()
    results["rv"].to_csv(TABLE_DIR / "robustness_rv.csv", index=False)

    results["target"] = alternative_target(features)
    results["target"].to_csv(TABLE_DIR / "robustness_target.csv", index=False)

    results["window"] = window_sensitivity(features)
    results["window"].to_csv(TABLE_DIR / "robustness_window.csv", index=False)

    results["regularization"] = regularization_comparison(features)
    results["regularization"].to_csv(
        TABLE_DIR / "robustness_regularization.csv", index=False
    )

    print(f"\n  ✓ Robustness tables saved to {TABLE_DIR}")
    return results


if __name__ == "__main__":
    run_robustness()