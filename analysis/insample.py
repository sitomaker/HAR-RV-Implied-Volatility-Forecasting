import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROC, OUTPUT_DIR, TABLE_DIR,
    FEATURES_M2, FEATURES_M3, FEATURES_M4,
    HORIZONS,
)


def load_features() -> pd.DataFrame:
    path = DATA_PROC / "features_gk.parquet"
    return pd.read_parquet(path)


def andrews_hac_lags(T: int) -> int:
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


#  OLS WITH HAC STANDARD ERRORS
def ols_hac(X: pd.DataFrame, y: pd.Series,
            hac_lags: int = None) -> dict:
    """
    OLS estimation with Newey-West HAC standard errors.

    Returns dict with coefficients, standard errors, t-stats,
    p-values, R², and diagnostic tests.
    """
    import statsmodels.api as sm

    X_const = sm.add_constant(X)
    T = len(y)

    if hac_lags is None:
        hac_lags = andrews_hac_lags(T)

    model = sm.OLS(y, X_const)
    results = model.fit(
        cov_type='HAC',
        cov_kwds={'maxlags': hac_lags}
    )

    # Diagnostics
    resid = results.resid

    # Ljung-Box test for serial correlation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    try:
        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_stat = lb["lb_stat"].values[0]
        lb_pval = lb["lb_pvalue"].values[0]
    except Exception:
        lb_stat, lb_pval = np.nan, np.nan

    # ARCH LM test for conditional heteroskedasticity
    from statsmodels.stats.diagnostic import het_arch
    try:
        arch_lm, arch_p, _, _ = het_arch(resid, nlags=5)
    except Exception:
        arch_lm, arch_p = np.nan, np.nan

    # Durbin-Watson
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(resid)

    # Variable names
    var_names = ["Intercept"] + list(X.columns)

    return {
        "coefficients": results.params.values,
        "std_errors": results.bse.values,
        "t_stats": results.tvalues.values,
        "p_values": results.pvalues.values,
        "var_names": var_names,
        "r_squared": results.rsquared,
        "adj_r_squared": results.rsquared_adj,
        "n_obs": int(results.nobs),
        "hac_lags": hac_lags,
        "lb_stat_10": lb_stat,
        "lb_pval_10": lb_pval,
        "arch_lm_stat": arch_lm,
        "arch_lm_pval": arch_p,
        "durbin_watson": dw,
        "aic": results.aic,
        "bic": results.bic,
        "residuals": resid.values,
        "fitted": results.fittedvalues.values,
        "results_obj": results,
    }


#  HAR-OLS ESTIMATION (TABLE 3)
def estimate_har_ols(features: pd.DataFrame) -> dict:
    """
    Estimate HAR-OLS (Model 2) for both horizons.
    Full sample, HAC standard errors.
    """
    results = {}

    for h in HORIZONS:
        target = f"y{h}"
        cols = FEATURES_M2 + [target]
        valid = features[cols].dropna()

        X = valid[FEATURES_M2]
        y = valid[target]

        res = ols_hac(X, y)
        results[h] = res

        print(f"\n    HAR-OLS (h={h}):")
        print(f"    N={res['n_obs']}, R²={res['r_squared']:.4f}, "
              f"Adj R²={res['adj_r_squared']:.4f}")
        for i, name in enumerate(res["var_names"]):
            sig = ""
            p = res["p_values"][i]
            if p < 0.01:
                sig = "***"
            elif p < 0.05:
                sig = "**"
            elif p < 0.10:
                sig = "*"
            print(f"      {name:12s}: {res['coefficients'][i]:8.4f} "
                  f"({res['std_errors'][i]:.4f}) {sig}")

        print(f"    Ljung-Box Q(10): {res['lb_stat_10']:.2f} "
              f"(p={res['lb_pval_10']:.4f})")
        print(f"    ARCH LM(5):      {res['arch_lm_stat']:.2f} "
              f"(p={res['arch_lm_pval']:.4f})")

    return results


#  SHAR ESTIMATION + ASYMMETRY TEST (TABLE 4)
def estimate_shar(features: pd.DataFrame) -> dict:
    """
    Estimate SHAR (Model 3) and test β_d+ = β_d-.
    """
    results = {}

    for h in HORIZONS:
        target = f"y{h}"
        cols = FEATURES_M3 + [target]
        valid = features[cols].dropna()

        X = valid[FEATURES_M3]
        y = valid[target]

        res = ols_hac(X, y)

        # F-test: β_d+ = β_d- (asymmetry test)
        # In the coefficient vector: [const, RV_d_plus, RV_d_minus, RV_w, RV_m]
        # Test: coef[1] = coef[2]
        import statsmodels.api as sm
        results_obj = res["results_obj"]

        try:
            # R * beta = q  →  beta[1] - beta[2] = 0
            f_test = results_obj.f_test("RV_d_plus = RV_d_minus")
            f_stat = float(f_test.fvalue)
            f_pval = float(f_test.pvalue)
        except Exception:
            # Manual F-test if named test fails
            try:
                R = np.zeros((1, len(res["coefficients"])))
                R[0, 1] = 1   # RV_d_plus
                R[0, 2] = -1  # RV_d_minus
                f_test = results_obj.f_test(R)
                f_stat = float(f_test.fvalue)
                f_pval = float(f_test.pvalue)
            except Exception:
                f_stat, f_pval = np.nan, np.nan

        res["f_stat_asymmetry"] = f_stat
        res["f_pval_asymmetry"] = f_pval

        results[h] = res

        print(f"\n    SHAR (h={h}):")
        print(f"    N={res['n_obs']}, R²={res['r_squared']:.4f}")
        for i, name in enumerate(res["var_names"]):
            sig = ""
            p = res["p_values"][i]
            if p < 0.01:
                sig = "***"
            elif p < 0.05:
                sig = "**"
            elif p < 0.10:
                sig = "*"
            print(f"      {name:14s}: {res['coefficients'][i]:8.4f} "
                  f"({res['std_errors'][i]:.4f}) {sig}")
        print(f"    F-test (β_d+ = β_d-): F={f_stat:.3f}, p={f_pval:.4f}")

    return results


#  HAR-EXTENDED ESTIMATION (TABLE 5)
def estimate_har_extended(features: pd.DataFrame) -> dict:
    """
    Estimate HAR-Extended (Model 4) by OLS with HAC.
    Full sample for interpretability.
    """
    results = {}

    for h in HORIZONS:
        target = f"y{h}"
        available_feats = [c for c in FEATURES_M4
                           if c in features.columns]
        cols = available_feats + [target]
        valid = features[cols].dropna()

        X = valid[available_feats]
        y = valid[target]

        res = ols_hac(X, y)
        results[h] = res

        print(f"\n    HAR-Extended (h={h}):")
        print(f"    N={res['n_obs']}, R²={res['r_squared']:.4f}, "
              f"Adj R²={res['adj_r_squared']:.4f}")
        for i, name in enumerate(res["var_names"]):
            sig = ""
            p = res["p_values"][i]
            if p < 0.01:
                sig = "***"
            elif p < 0.05:
                sig = "**"
            elif p < 0.10:
                sig = "*"
            print(f"      {name:14s}: {res['coefficients'][i]:8.4f} "
                  f"({res['std_errors'][i]:.4f}) {sig}")

    return results


#  RESIDUAL DIAGNOSTICS
def residual_diagnostics(results: dict, model_name: str) -> dict:
    """
    Compute residual diagnostics for a given estimation result.
    ACF, squared ACF statistics, normality tests.
    """
    diagnostics = {}

    for h, res in results.items():
        resid = res["residuals"]

        from statsmodels.stats.diagnostic import acorr_ljungbox
        from scipy.stats import jarque_bera, shapiro

        # ACF of residuals
        lb = acorr_ljungbox(resid, lags=[5, 10, 20], return_df=True)

        # ACF of squared residuals (ARCH effects)
        lb_sq = acorr_ljungbox(resid**2, lags=[5, 10, 20], return_df=True)

        # Normality
        jb_stat, jb_p = jarque_bera(resid)

        diagnostics[h] = {
            "lb_resid": lb.to_dict(),
            "lb_sq_resid": lb_sq.to_dict(),
            "jb_stat": jb_stat,
            "jb_pval": jb_p,
            "resid_mean": np.mean(resid),
            "resid_std": np.std(resid),
            "resid_skew": pd.Series(resid).skew(),
            "resid_kurt": pd.Series(resid).kurtosis(),
        }

        print(f"\n    Diagnostics {model_name} (h={h}):")
        print(f"      Resid mean: {np.mean(resid):.6f}")
        print(f"      Resid skew: {pd.Series(resid).skew():.3f}, "
              f"kurt: {pd.Series(resid).kurtosis():.3f}")
        print(f"      Jarque-Bera: {jb_stat:.2f} (p={jb_p:.4f})")
        print(f"      LB Q(10) resid:    p={lb['lb_pvalue'].iloc[1]:.4f}")
        print(f"      LB Q(10) resid²:   p={lb_sq['lb_pvalue'].iloc[1]:.4f}")

    return diagnostics


#  STRUCTURAL STABILITY (CUSUM)
def structural_stability(features: pd.DataFrame,
                         horizon: int = 5) -> dict:
    """
    CUSUM test and recursive coefficient estimates.
    Uses HAR-OLS (Model 2) for simplicity.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    target = f"y{horizon}"
    cols = FEATURES_M2 + [target]
    valid = features[cols].dropna()

    X = sm.add_constant(valid[FEATURES_M2])
    y = valid[target]

    # Full OLS for CUSUM
    model = sm.OLS(y, X)
    results = model.fit()

    # CUSUM test
    try:
        cusum_stat, cusum_pval, cusum_critical = breaks_cusumolsresid(
            results.resid
        )
    except Exception:
        cusum_stat, cusum_pval = np.nan, np.nan

    # Recursive coefficients
    # Estimate OLS on expanding windows and track coefficient evolution
    min_obs = 100
    n = len(valid)
    step = max(1, n // 200)  # ~200 points for plot

    recursive_betas = []
    recursive_dates = []
    recursive_se = []

    for t in range(min_obs, n, step):
        X_sub = X.iloc[:t]
        y_sub = y.iloc[:t]

        try:
            res_sub = sm.OLS(y_sub, X_sub).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': max(4, andrews_hac_lags(t))}
            )
            recursive_betas.append(res_sub.params.values)
            recursive_se.append(res_sub.bse.values)
            recursive_dates.append(valid.index[t - 1])
        except Exception:
            continue

    result = {
        "cusum_stat": cusum_stat,
        "cusum_pval": cusum_pval,
        "recursive_betas": np.array(recursive_betas),
        "recursive_se": np.array(recursive_se),
        "recursive_dates": recursive_dates,
        "var_names": ["Intercept"] + FEATURES_M2,
        "residuals": results.resid.values,
        "dates": valid.index,
    }

    print(f"\n    Structural Stability (h={horizon}):")
    print(f"      CUSUM stat: {cusum_stat:.4f}")
    if not np.isnan(cusum_pval) if isinstance(cusum_pval, float) else cusum_pval is not None:
        print(f"      CUSUM p-val: {cusum_pval:.4f}")

    return result


#  MASTER IN-SAMPLE RUNNER
def run_insample():
    """Run all in-sample analyses."""
    print("=" * 60)
    print("IN-SAMPLE ESTIMATION")
    print("=" * 60)

    features = load_features()

    # 1. HAR-OLS
    print("\n── HAR-OLS (Model 2) ──")
    har_results = estimate_har_ols(features)

    # 2. SHAR
    print("\n── SHAR (Model 3) ──")
    shar_results = estimate_shar(features)

    # 3. HAR-Extended
    print("\n── HAR-Extended (Model 4) ──")
    ext_results = estimate_har_extended(features)

    # 4. Residual diagnostics
    print("\n── Residual Diagnostics ──")
    har_diag = residual_diagnostics(har_results, "HAR-OLS")
    ext_diag = residual_diagnostics(ext_results, "HAR-Extended")

    # 5. Structural stability
    print("\n── Structural Stability ──")
    stability = structural_stability(features, horizon=5)

    # Save all results
    import pickle
    insample_output = {
        "har_ols": har_results,
        "shar": shar_results,
        "har_extended": ext_results,
        "diagnostics_har": har_diag,
        "diagnostics_ext": ext_diag,
        "stability": stability,
    }

    # Remove non-serializable objects before saving
    for model_results in [har_results, shar_results, ext_results]:
        for h in model_results:
            if "results_obj" in model_results[h]:
                del model_results[h]["results_obj"]

    with open(OUTPUT_DIR / "insample_results.pkl", "wb") as f:
        pickle.dump(insample_output, f)

    print(f"\n  ✓ In-sample results saved to {OUTPUT_DIR}")
    return insample_output


if __name__ == "__main__":
    run_insample()
