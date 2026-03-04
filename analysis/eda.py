"""
analysis/eda.py — Exploratory Data Analysis.

Produces:
- Descriptive statistics table
- ADF and KPSS stationarity tests
- Correlation matrix
- Variance Inflation Factors (VIF)
- Distribution analysis of targets

All outputs saved to output/tables/ and output/figures/.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROC, OUTPUT_DIR, TABLE_DIR, FIGURE_DIR,
    FEATURES_M4, HORIZONS,
)


def load_features() -> pd.DataFrame:
    """Load feature matrix."""
    path = DATA_PROC / "features_gk.parquet"
    return pd.read_parquet(path)


# ══════════════════════════════════════════════════════════════
#  DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════

def descriptive_statistics(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive stats for all key variables.
    Matches Table 1 in the paper.
    """
    # Load master for raw VIX and returns
    master = pd.read_parquet(DATA_PROC / "master.parquet")

    # Build stats DataFrame
    variables = {}

    # SPY returns (in %)
    ret = master["SPY_ret"].dropna() * 100
    variables["r_t (SPY, %)"] = ret

    # VIX level
    variables["VIX_t"] = master["VIX"].dropna()

    # VIX changes
    variables["ΔVIX_t,1"] = features["y1"].dropna()
    variables["ΔVIX_t,5"] = features["y5"].dropna()

    # RV daily (GK)
    variables["RV_d (GK)"] = features["RV_d"].dropna()

    # VRP
    variables["VRP_t"] = features["VRP"].dropna()

    # Slope
    variables["slope_t"] = features["slope"].dropna()

    # dy10
    variables["Δy_10Y"] = features["dy10"].dropna()

    # MOVE
    variables["MOVE_t"] = features["MOVE"].dropna()

    # Credit spread
    variables["spread_t"] = features["spread"].dropna()

    # Build table
    rows = []
    for name, series in variables.items():
        s = series.dropna()
        rows.append({
            "Variable": name,
            "N": len(s),
            "Mean": s.mean(),
            "Std Dev": s.std(),
            "Min": s.min(),
            "Q25": s.quantile(0.25),
            "Median": s.median(),
            "Q75": s.quantile(0.75),
            "Max": s.max(),
            "Skew": s.skew(),
            "Kurt": s.kurtosis(),
        })

    df = pd.DataFrame(rows)
    return df


# ══════════════════════════════════════════════════════════════
#  STATIONARITY TESTS
# ══════════════════════════════════════════════════════════════

def stationarity_tests(features: pd.DataFrame) -> pd.DataFrame:
    """
    ADF and KPSS tests for key series.
    Matches Table 2 in the paper.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    master = pd.read_parquet(DATA_PROC / "master.parquet")

    series_dict = {
        "VIX_t (level)": master["VIX"].dropna(),
        "ΔVIX_t,1": features["y1"].dropna(),
        "ΔVIX_t,5": features["y5"].dropna(),
        "RV_d (GK)": features["RV_d"].dropna(),
        "VRP_t": features["VRP"].dropna(),
    }

    rows = []
    for name, s in series_dict.items():
        s_clean = s.dropna()
        if len(s_clean) < 50:
            continue

        # ADF test (H0: unit root)
        try:
            adf_stat, adf_p, adf_lags, _, _, _ = adfuller(
                s_clean, autolag="BIC"
            )
        except Exception:
            adf_stat, adf_p = np.nan, np.nan

        # KPSS test (H0: stationarity)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_stat, kpss_p, kpss_lags, _ = kpss(
                    s_clean, regression="c", nlags="auto"
                )
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan

        rows.append({
            "Series": name,
            "ADF Stat": adf_stat,
            "ADF p-val": adf_p,
            "KPSS Stat": kpss_stat,
            "KPSS p-val": kpss_p,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════

def correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix of all features + targets.
    Used for the correlation heatmap figure.
    """
    cols = FEATURES_M4 + ["y1", "y5"]
    available = [c for c in cols if c in features.columns]
    return features[available].corr()


# ══════════════════════════════════════════════════════════════
#  VARIANCE INFLATION FACTORS
# ══════════════════════════════════════════════════════════════

def compute_vif(features: pd.DataFrame) -> pd.DataFrame:
    """
    VIF for Model 4 feature set.
    VIF > 10 indicates problematic multicollinearity.
    Motivates Ridge regularization.
    """
    from statsmodels.stats.outliers_influence import (
        variance_inflation_factor,
    )

    cols = FEATURES_M4
    available = [c for c in cols if c in features.columns]
    X = features[available].dropna()

    # Add constant for VIF calculation
    from statsmodels.tools import add_constant
    X_const = add_constant(X)

    vif_data = []
    for i, col in enumerate(available):
        try:
            vif_val = variance_inflation_factor(
                X_const.values, i + 1  # +1 because constant is at 0
            )
        except Exception:
            vif_val = np.nan

        vif_data.append({"Feature": col, "VIF": vif_val})

    return pd.DataFrame(vif_data)


# ══════════════════════════════════════════════════════════════
#  DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════

def distribution_analysis(features: pd.DataFrame) -> dict:
    """
    Distribution statistics for targets.
    Jarque-Bera normality test.
    """
    results = {}
    for col in ["y1", "y5"]:
        s = features[col].dropna()
        jb_stat, jb_p = sp_stats.jarque_bera(s)
        results[col] = {
            "mean": s.mean(),
            "std": s.std(),
            "skewness": s.skew(),
            "kurtosis": s.kurtosis(),
            "jb_stat": jb_stat,
            "jb_pval": jb_p,
            "shapiro_p": sp_stats.shapiro(
                s.sample(min(5000, len(s)), random_state=42)
            )[1],
        }
    return results


# ══════════════════════════════════════════════════════════════
#  MASTER EDA RUNNER
# ══════════════════════════════════════════════════════════════

def run_eda():
    """Run all EDA analyses and save results."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    features = load_features()

    # 1. Descriptive statistics
    print("\n  1. Descriptive statistics...")
    desc = descriptive_statistics(features)
    desc.to_csv(TABLE_DIR / "descriptive_stats.csv", index=False)
    print(desc.to_string(index=False, float_format="%.4f"))

    # 2. Stationarity tests
    print("\n  2. Stationarity tests...")
    station = stationarity_tests(features)
    station.to_csv(TABLE_DIR / "stationarity_tests.csv", index=False)
    print(station.to_string(index=False, float_format="%.4f"))

    # 3. Correlation matrix
    print("\n  3. Correlation matrix...")
    corr = correlation_matrix(features)
    corr.to_csv(TABLE_DIR / "correlation_matrix.csv")
    print(f"    Shape: {corr.shape}")
    print(f"    Max off-diagonal: {corr.where(np.eye(len(corr)) == 0).abs().max().max():.3f}")

    # 4. VIF
    print("\n  4. Variance Inflation Factors...")
    vif = compute_vif(features)
    vif.to_csv(TABLE_DIR / "vif.csv", index=False)
    print(vif.to_string(index=False, float_format="%.2f"))

    # 5. Distribution analysis
    print("\n  5. Distribution analysis...")
    dist = distribution_analysis(features)
    for col, stats_dict in dist.items():
        print(f"    {col}: skew={stats_dict['skewness']:.3f}, "
              f"kurt={stats_dict['kurtosis']:.3f}, "
              f"JB p={stats_dict['jb_pval']:.4f}")

    # Save all as JSON
    import json

    eda_results = {
        "descriptive": desc.to_dict(orient="records"),
        "stationarity": station.to_dict(orient="records"),
        "vif": vif.to_dict(orient="records"),
        "distribution": dist,
    }

    with open(OUTPUT_DIR / "eda_results.json", "w") as f:
        json.dump(eda_results, f, indent=2, default=str)

    print(f"\n  ✓ EDA results saved to {OUTPUT_DIR}")
    return eda_results


if __name__ == "__main__":
    run_eda()