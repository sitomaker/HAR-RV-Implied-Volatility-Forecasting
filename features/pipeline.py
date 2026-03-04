import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROC, TRADING_DAYS_YEAR,
    HAR_DAILY, HAR_WEEKLY, HAR_MONTHLY,
    REGIME_BINS, REGIME_LABELS,
    RV_ESTIMATORS, RV_PRIMARY,
)


#  LOW-LEVEL RV ESTIMATORS
def garman_klass_daily(spy: pd.DataFrame) -> pd.Series:
    log_hl = np.log(spy["SPY_High"]  / spy["SPY_Low"])
    log_co = np.log(spy["SPY_Close"] / spy["SPY_Open"])
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return gk.clip(lower=0)


def parkinson_daily(spy: pd.DataFrame) -> pd.Series:
    log_hl = np.log(spy["SPY_High"] / spy["SPY_Low"])
    return (log_hl ** 2) / (4 * np.log(2))


def cc_daily(spy: pd.DataFrame) -> pd.Series:
    log_ret = np.log(spy["SPY_Close"] / spy["SPY_Close"].shift(1))
    return log_ret ** 2


def rv_rolling(daily_var: pd.Series, window: int) -> pd.Series:
    return np.sqrt(
        (TRADING_DAYS_YEAR / window) * daily_var.rolling(window).sum()
    ) * 100


def rv_close_to_close(returns: pd.Series, window: int) -> pd.Series:
    return np.sqrt(
        (TRADING_DAYS_YEAR / window) * (returns ** 2).rolling(window).sum()
    ) * 100


#  HAR COMPONENTS
def har_components(daily_var: pd.Series) -> pd.DataFrame:
    """
    Corsi (2009) HAR triple: RV_d, RV_w, RV_m.
    """
    rv_d = rv_rolling(daily_var, HAR_DAILY)
    rv_w = rv_d.rolling(HAR_WEEKLY).mean()
    rv_m = rv_d.rolling(HAR_MONTHLY).mean()
    return pd.DataFrame({"RV_d": rv_d, "RV_w": rv_w, "RV_m": rv_m})


#  SEMI-VARIANCE COMPONENTS
def signed_semi_variances(ret: pd.Series) -> pd.DataFrame:
    rs_plus  = (ret ** 2) * (ret >= 0).astype(float)
    rs_minus = (ret ** 2) * (ret <  0).astype(float)

    assert np.allclose(
        rs_plus + rs_minus, ret ** 2, equal_nan=True,
    ), "Semi-variance identity RS+ + RS- = r² violated."

    rv_d_plus  = np.sqrt(TRADING_DAYS_YEAR * rs_plus)  * 100
    rv_d_minus = np.sqrt(TRADING_DAYS_YEAR * rs_minus) * 100

    return pd.DataFrame({"RV_d_plus": rv_d_plus, "RV_d_minus": rv_d_minus})


#  VARIANCE RISK PREMIUM
def variance_risk_premium(vix: pd.Series, rv_m: pd.Series) -> pd.Series:
    vrp_raw = vix ** 2 - rv_m ** 2

    # ── Expanding-window winsorization (no look-ahead) ────────
    p01 = vrp_raw.expanding(min_periods=50).quantile(0.01)
    p99 = vrp_raw.expanding(min_periods=50).quantile(0.99)
    vrp = vrp_raw.clip(lower=p01, upper=p99)

    return vrp


#  VIX TERM STRUCTURE
def vix_term_structure(
    vix9d: pd.Series,
    vix: pd.Series,
    vix3m: pd.Series,
) -> pd.DataFrame:
    """
    slope_t = VIX3M_t - VIX9D_t
    curv_t  = VIX9D_t - 2*VIX_t + VIX3M_t
    """
    slope = vix3m - vix9d
    curv  = vix9d - 2 * vix + vix3m
    return pd.DataFrame({"slope": slope, "curv": curv})


#  REGIME CLASSIFICATION
def classify_regime(vix: pd.Series) -> pd.Series:
    """
    Three-state regime: Low (<15), Medium (15-25), High (>=25).
    Thresholds fixed ex ante.
    """
    return pd.cut(
        vix, bins=REGIME_BINS, labels=REGIME_LABELS,
    ).astype(str)


#  PREDICTION TARGETS
def build_targets(vix: pd.Series) -> pd.DataFrame:
 
    # Aritméticos (original)
    y1 = vix.shift(-1) - vix
    y5 = vix.shift(-5) - vix
    
    # Logarítmicos (nuevo)
    y1_log = np.log(vix.shift(-1) / vix) * 100
    y5_log = np.log(vix.shift(-5) / vix) * 100
    
    # Winsorización de targets extremos (expanding window, sin look-ahead)
    def winsorize_expanding(s: pd.Series, lower_q=0.005, upper_q=0.995, min_periods=100):
        """Winsoriza usando percentiles expandidos (sin look-ahead)."""
        p_lo = s.expanding(min_periods=min_periods).quantile(lower_q)
        p_hi = s.expanding(min_periods=min_periods).quantile(upper_q)
        return s.clip(lower=p_lo, upper=p_hi)
    
    # Aplicar winsorización suave a targets aritméticos
    y1_win = winsorize_expanding(y1)
    y5_win = winsorize_expanding(y5)
    
    return pd.DataFrame({
        "y1": y1,           # Original
        "y5": y5,           # Original
        "y1_log": y1_log,   # Logarítmico
        "y5_log": y5_log,   # Logarítmico
        "y1_win": y1_win,   # Winsorizado
        "y5_win": y5_win,   # Winsorizado
    })

#  MAIN FEATURE BUILDER
def build_all_features(
    rv_estimator: str = RV_PRIMARY,
    verify_leak: bool = True,
) -> pd.DataFrame: 
    master = pd.read_parquet(DATA_PROC / "master.parquet")
    feat   = pd.DataFrame(index=master.index)

    # ── 1-2. RV components ────────────────────────────────────
    if rv_estimator == "gk":
        daily_var = garman_klass_daily(master)
    elif rv_estimator == "parkinson":
        daily_var = parkinson_daily(master)
    elif rv_estimator == "cc":
        daily_var = master["SPY_ret"] ** 2
    else:
        raise ValueError(f"Unknown rv_estimator: {rv_estimator!r}")

    har = har_components(daily_var)
    feat[["RV_d", "RV_w", "RV_m"]] = har

    # ── 3. Signed semi-variances ──────────────────────────────
    semivar = signed_semi_variances(master["SPY_ret"])
    feat[["RV_d_plus", "RV_d_minus"]] = semivar

    # ── 4. VRP (with expanding-window winsorization) ──────────
    vix = master["VIX"]
    feat["VRP"] = variance_risk_premium(vix, feat["RV_m"])

    # ── 5. VIX term structure ─────────────────────────────────
    vix9d = master["VIX9D"].copy()
    n_missing_before = vix9d.isna().sum()
    vix9d = vix9d.fillna(vix)           # pre-2011: VIX9D ≈ VIX
    vix9d = vix9d.ffill(limit=5)        # residual gaps (holidays)
    n_missing_after = vix9d.isna().sum()
    if n_missing_before > 0:
        print(f"  NOTE: VIX9D proxied with VIX for {n_missing_before} days "
              f"({n_missing_after} still missing after fill)")

    ts = vix_term_structure(vix9d, vix, master["VIX3M"])
    feat[["slope", "curv"]] = ts

    # ── 6. Macro covariates ───────────────────────────────────
    feat["dy10"] = master["DGS10"].diff(1)

    # MOVE: always from master (clean.py already handles proxy fallback)
    feat["MOVE"] = master["MOVE"] if "MOVE" in master.columns else np.nan

    # Credit spread: always from master (clean.py computes BBB - AAA)
    feat["spread"] = master["spread"] if "spread" in master.columns else np.nan

    # ── 7. Lagged returns ─────────────────────────────────────
    ret = master["SPY_ret"]
    feat["r_lag1"] = ret.shift(1)
    feat["r_lag5"] = ret.rolling(5).sum().shift(1)

    # ── 8. VIX level + regime ─────────────────────────────────
    feat["VIX_level"] = vix
    feat["regime"] = classify_regime(vix)

    # ── 9. Targets ────────────────────────────────────────────
    targets = build_targets(vix)
    feat[["y1", "y5", "y1_log", "y5_log", "y1_win", "y5_win"]] = targets

    if verify_leak:
        _verify_no_leakage(feat, master)
    
    report_target_distributions(feat)

    return feat


#  ANTI-LEAKAGE VERIFICATION
def _verify_no_leakage(feat: pd.DataFrame, master: pd.DataFrame) -> None:
    fwd_ret = master["SPY_ret"].shift(-1)
    common  = feat.index.intersection(fwd_ret.dropna().index)

    suspicious = []
    for col in ["RV_d", "RV_w", "RV_m", "VRP", "slope"]:
        if col not in feat.columns:
            continue
        s = feat[col].reindex(common).dropna()
        f = fwd_ret.reindex(s.index)
        if len(s) < 10:
            continue
        corr = s.corr(f)
        if abs(corr) > 0.30:
            suspicious.append((col, corr))

    if suspicious:
        print("  WARNING: possible leakage detected:")
        for col, corr in suspicious:
            print(f"    {col}: corr with fwd_ret = {corr:.3f}")
    else:
        print("  ✓ No obvious leakage detected.")

def report_target_distributions(feat: pd.DataFrame) -> None:
    from scipy.stats import jarque_bera, skew, kurtosis
    
    print("\n── Target Distribution Analysis ──")
    
    targets = ["y1", "y5", "y1_log", "y5_log", "y1_win", "y5_win"]
    
    for t in targets:
        if t not in feat.columns:
            continue
        s = feat[t].dropna()
        if len(s) < 50:
            continue
        
        sk = skew(s)
        kt = kurtosis(s)  # Excess kurtosis
        jb_stat, jb_p = jarque_bera(s)
        
        print(f"  {t:8s}: N={len(s):5d}, "
              f"mean={s.mean():7.3f}, std={s.std():6.3f}, "
              f"skew={sk:6.2f}, kurt={kt:6.2f}, "
              f"JB p={jb_p:.4f}")
        

#  BUILD AND SAVE ALL ESTIMATOR VARIANTS
def build_and_save_all() -> None:
    """Build and persist feature parquets for all three RV estimators."""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    for est in RV_ESTIMATORS:
        print(f"\n── Building features: RV estimator = {est.upper()} ──")
        try:
            feat = build_all_features(rv_estimator=est, verify_leak=True)
            out_path = DATA_PROC / f"features_{est}.parquet"
            feat.to_parquet(out_path)
            n_valid = feat.dropna(subset=["y1", "y5"]).shape[0]

            # Report VRP stats after winsorization
            vrp_stats = feat["VRP"].dropna()
            print(
                f"  ✓ Saved {out_path.name}  |  shape={feat.shape}  "
                f"|  valid rows (y1+y5 not NaN)={n_valid}"
            )
            print(
                f"    VRP after winsorization: "
                f"mean={vrp_stats.mean():.1f}, "
                f"std={vrp_stats.std():.1f}, "
                f"min={vrp_stats.min():.1f}, "
                f"max={vrp_stats.max():.1f}"
            )
        except Exception as e:
            print(f"  ERROR building {est}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✓ All feature files saved to data/processed/")


if __name__ == "__main__":
    if "--all" in sys.argv:
        build_and_save_all()
    else:
        print("Building GK features...")
        feat = build_all_features(rv_estimator="gk", verify_leak=True)
        out_path = DATA_PROC / "features_gk.parquet"
        feat.to_parquet(out_path)
        print(f"✓ Saved {out_path}")
        print(f"  Shape: {feat.shape}")
        print(f"  Columns: {list(feat.columns)}")
        missing = feat.isna().sum()
        if missing[missing > 0].any():
            print(f"  Missing values:\n{missing[missing > 0]}")
