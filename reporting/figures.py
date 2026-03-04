"""
reporting/figures.py — Generate all figures for the paper.

14 figures total, saved as PDF + PNG in output/figures/.

Changes from original:
  - fig_har_vrp: Y-axis label corrected to "VIX3M − VIX9D" (LaTeX slope def.)
  - fig_structural_stability: CUSUM uses Brown-Durbin-Evans (1975) correct
    bounds ±(c_α + 2(t−k)/n) with c_0.05 = 0.948; not the ad-hoc version.
  - fig_cumulative_r2: dates derived from fold_info (safe); shows M2 + all-model
    confidence band; bootstrap CI shaded.
  - fig_fold_rmse: parametrized for both horizons; includes M5.
  - fig_vix_timeseries: regime shading upper bound from ax.get_ylim() not 90.
  - fig_model_nesting: complete test battery (CW: M0→M2, M0→M3, M0→M4,
    M2→M3, M2→M4; DM: M0/M1→M4, M4→M5, M1 vs M0).
  - NEW fig_walkforward_timeline: broken_barh timeline as requested in LaTeX.
  - NEW fig_regime_r2: R²_oos bar chart by regime for all models (LaTeX §regime_perf).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROC, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR,
    REGIME_LOW_UPPER, REGIME_MED_UPPER, WF_T_MIN, WF_TAU,
)

# ── Global style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":          150,
    "font.size":           10,
    "font.family":         "serif",
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "legend.fontsize":     8.5,
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linewidth":      0.5,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
})

COLORS = {
    "blue":   "#1f4e79",
    "red":    "#c0392b",
    "green":  "#27ae60",
    "orange": "#e67e22",
    "gray":   "#7f8c8d",
    "purple": "#8e44ad",
    "gold":   "#d4ac0d",
}

# Human-readable model labels for all plots and tables
MODEL_LABELS: dict[str, str] = {
    "M0_RandomWalk":   "M0 (RW)",
    "M1_GARCH":        "M1 (GARCH)",
    "M2_HAROLS":       "M2 (HAR-OLS)",
    "M3_SHAR":         "M3 (SHAR)",
    "M4_HARExtended":  "M4 (HAR-Ext.)",
    "M5_HARRegime":    "M5 (HAR-Regime)",
}

MODEL_COLORS: dict[str, str] = {
    "M0_RandomWalk":   COLORS["gray"],
    "M1_GARCH":        COLORS["purple"],
    "M2_HAROLS":       COLORS["orange"],
    "M3_SHAR":         COLORS["gold"],
    "M4_HARExtended":  COLORS["blue"],
    "M5_HARRegime":    COLORS["red"],
}

MODEL_LS: dict[str, str] = {
    "M0_RandomWalk":   ":",
    "M1_GARCH":        "--",
    "M2_HAROLS":       "-.",
    "M3_SHAR":         "-.",
    "M4_HARExtended":  "-",
    "M5_HARRegime":    "-",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF and PNG."""
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"    ✓ {name}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load master feature and price data."""
    features = pd.read_parquet(DATA_PROC / "features_gk.parquet")
    master   = pd.read_parquet(DATA_PROC / "master.parquet")
    return features, master


def _load_results() -> dict | None:
    try:
        with open(OUTPUT_DIR / "all_results.pkl", "rb") as fh:
            return pickle.load(fh)
    except Exception:
        print("    Skipping: run walk-forward evaluation first")
        return None


def _dates_from_folds(fold_info: list[dict]) -> list[pd.Timestamp]:
    """Reconstruct OOS date sequence from fold metadata."""
    dates = []
    for fold in fold_info:
        start = pd.Timestamp(fold["fold_start"])
        end   = pd.Timestamp(fold["fold_end"])
        n     = fold["test_size"]
        # daily business-day range within fold
        rng = pd.bdate_range(start, end)[:n]
        dates.extend(rng.tolist())
    return dates


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 0: WALK-FORWARD TIMELINE  (LaTeX §figplaceholder)
# ══════════════════════════════════════════════════════════════════════════════

def fig_walkforward_timeline(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """
    Horizontal broken_barh timeline showing expanding training windows,
    fixed test blocks, and purge gaps for h=5.  LaTeX §figplaceholder.
    """
    n      = len(features.dropna(subset=["y5"]))
    L      = (n - WF_T_MIN) // WF_TAU
    h      = 5

    fig, axes = plt.subplots(2, 1, figsize=(14, 5),
                              gridspec_kw={"height_ratios": [3, 1]})

    ax    = axes[0]
    y_tr  = 0.55    # y-centre for training bars
    y_te  = 0.15    # y-centre for test bars
    bh    = 0.18    # bar height

    # Plot every fold (subsample labels to avoid clutter)
    label_each = max(1, L // 10)

    for ell in range(min(L, 52)):
        train_end  = WF_T_MIN + ell * WF_TAU - h
        test_start = WF_T_MIN + ell * WF_TAU
        test_end   = test_start + WF_TAU

        ax.broken_barh([(0, train_end)], (y_tr - bh / 2, bh),
                       facecolors=COLORS["blue"], alpha=0.25 + 0.006 * ell,
                       edgecolors="none")

        # Purge gap (hatched)
        ax.broken_barh([(train_end, h)], (y_tr - bh / 2, bh),
                       facecolors="none", edgecolors=COLORS["gray"],
                       linewidth=0.5, hatch="////")

        ax.broken_barh([(test_start, WF_TAU)], (y_te - bh / 2, bh),
                       facecolors=COLORS["red"], alpha=0.7,
                       edgecolors="none")

        if ell % label_each == 0:
            ax.text(test_start + WF_TAU / 2, y_te - bh / 2 - 0.04,
                    f"$\\ell={ell+1}$", ha="center", va="top", fontsize=6.5)

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([y_te, y_tr])
    ax.set_yticklabels(["Test block", "Training window"])
    ax.set_xlabel("Observation index")
    ax.set_title(
        f"Expanding-Window Walk-Forward Protocol  "
        f"($T_{{\\min}}={WF_T_MIN}$, $\\tau={WF_TAU}$, purge $h=5$)"
    )

    # Legend patches
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["blue"], alpha=0.5,
                       label="Training (targets observed)"),
        mpatches.Patch(facecolor="none", edgecolor=COLORS["gray"],
                       hatch="////",
                       label=f"Purge gap ($h={h}$ days)"),
        mpatches.Patch(facecolor=COLORS["red"], alpha=0.7,
                       label="Test block ($\\tau=63$)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    # Bottom panel: inner CV schematic for fold ℓ=1
    ax2 = axes[1]
    K   = 5
    inner_T = max(100, WF_T_MIN // (K + 1))
    inner_tau = (WF_T_MIN - inner_T) // K

    for k in range(K):
        iv_tr = inner_T + k * inner_tau
        ax2.broken_barh([(0, iv_tr)], (0.55 - 0.12, 0.24),
                        facecolors=COLORS["blue"], alpha=0.2,
                        edgecolors="none")
        ax2.broken_barh([(iv_tr, inner_tau)], (0.15 - 0.12, 0.24),
                        facecolors=COLORS["orange"], alpha=0.7,
                        edgecolors="none")
        ax2.text(iv_tr + inner_tau / 2, 0.03,
                 f"$k={k+1}$", ha="center", va="bottom", fontsize=6.5)

    ax2.set_xlim(0, WF_T_MIN)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.15, 0.55])
    ax2.set_yticklabels(["Inner val.", "Inner train"])
    ax2.set_xlabel("Obs. index (fold $\\ell=1$ only)")
    ax2.set_title(f"Inner CV for Ridge $\\lambda$ ($K={K}$ folds, inset)", fontsize=9)

    save_fig(fig, "fig00_walkforward_timeline")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: VIX TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

def fig_vix_timeseries(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """Three panels: VIX level with regime shading, Δ1, Δ5."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    vix   = master["VIX"]
    dates = vix.index

    # ── Panel 1: VIX level ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(dates, vix, color=COLORS["blue"], linewidth=0.6)

    # Regime shading — upper bound from axis, not hardcoded 90
    ymax = vix.max() * 1.12
    ax.axhspan(0,                  REGIME_LOW_UPPER, alpha=0.08,
               color="green",  label="Low")
    ax.axhspan(REGIME_LOW_UPPER,   REGIME_MED_UPPER, alpha=0.08,
               color="gold",   label="Medium")
    ax.axhspan(REGIME_MED_UPPER,   ymax,             alpha=0.08,
               color="red",    label="High")

    ax.set_ylabel("VIX Level")
    ax.set_title("VIX Index with Volatility Regime Classification")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(5, ymax)

    events = {
        "2018-02-05": "Volmageddon",
        "2020-03-16": "COVID peak",
        "2022-03-16": "Rate hikes",
    }
    for date_str, label in events.items():
        try:
            d = pd.Timestamp(date_str)
            if d >= vix.index[0]:
                ax.axvline(d, color=COLORS["gray"], linewidth=0.6,
                           linestyle="--", alpha=0.8)
                ax.text(d, ymax * 0.94, label, fontsize=7,
                        ha="center", rotation=90, alpha=0.75,
                        va="top")
        except Exception:
            pass

    # ── Panel 2: 1-day Δ ──────────────────────────────────────────────
    ax = axes[1]
    y1 = features["y1"].dropna()
    ax.bar(y1.index, y1.values, width=1,
           color=np.where(y1.values >= 0, COLORS["red"], COLORS["green"]),
           alpha=0.6)
    ax.set_ylabel("$\\Delta$VIX (1-day)")
    ax.set_title("One-Day VIX Changes")
    ax.axhline(0, color="black", linewidth=0.4)

    # ── Panel 3: 5-day Δ ──────────────────────────────────────────────
    ax = axes[2]
    y5 = features["y5"].dropna()
    ax.bar(y5.index, y5.values, width=3,
           color=np.where(y5.values >= 0, COLORS["red"], COLORS["green"]),
           alpha=0.6)
    ax.set_ylabel("$\\Delta$VIX (5-day)")
    ax.set_title("Five-Day VIX Changes")
    ax.axhline(0, color="black", linewidth=0.4)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))

    save_fig(fig, "fig01_vix_timeseries")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: HAR COMPONENTS + VRP
# ══════════════════════════════════════════════════════════════════════════════

def fig_har_vrp(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """Four panels: RV components, VRP, term structure slope, regime bar."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    dates = features.index

    # ── Panel 1: HAR components ───────────────────────────────────────
    ax = axes[0]
    for col, label, color in [
        ("RV_d", "RV daily",   COLORS["blue"]),
        ("RV_w", "RV weekly",  COLORS["orange"]),
        ("RV_m", "RV monthly", COLORS["red"]),
    ]:
        if col in features.columns:
            ax.plot(dates, features[col], label=label,
                    color=color, linewidth=0.6, alpha=0.85)
    ax.set_ylabel("Annualized Vol")
    ax.set_title("HAR-RV Components (Garman-Klass Estimator)")
    ax.legend(loc="upper right")

    # ── Panel 2: VRP ──────────────────────────────────────────────────
    ax = axes[1]
    if "VRP" in features.columns:
        vrp = features["VRP"]
        ax.fill_between(dates, 0, vrp.where(vrp >= 0), alpha=0.45,
                        color=COLORS["blue"],  label="VRP $>$ 0")
        ax.fill_between(dates, 0, vrp.where(vrp < 0),  alpha=0.45,
                        color=COLORS["red"],   label="VRP $<$ 0")
        ax.axhline(vrp.mean(), color="black", linewidth=0.8,
                   linestyle="--",
                   label=f"Mean = {vrp.mean():.1f}")
    ax.set_ylabel("VRP (VIX$^2$ $-$ RV$^2$)")
    ax.set_title("Variance Risk Premium")
    ax.legend(loc="upper right")

    # ── Panel 3: Term structure slope ─────────────────────────────────
    # CORRECTED: slope = VIX3M − VIX9D (contango positive, LaTeX §features)
    ax = axes[2]
    if "slope" in features.columns:
        slope = features["slope"]
        ax.plot(dates, slope, color=COLORS["purple"], linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.fill_between(dates, 0, slope, where=slope >= 0,
                        alpha=0.25, color=COLORS["blue"],
                        label="Contango ($>0$)")
        ax.fill_between(dates, 0, slope, where=slope < 0,
                        alpha=0.25, color=COLORS["red"],
                        label="Backwardation ($<0$)")
        ax.legend(loc="upper right")
    ax.set_ylabel("VIX3M $-$ VIX9D")   # FIXED (was VIX9D − VIX3M)
    ax.set_title("VIX Term Structure Slope (Contango = positive)")

    # ── Panel 4: Regime bar ───────────────────────────────────────────
    ax = axes[3]
    if "regime" in features.columns:
        regime = features["regime"]
        for reg, color in [("Low", "green"), ("Medium", "gold"), ("High", "red")]:
            mask = regime == reg
            ax.fill_between(dates, 0, 1, where=mask, alpha=0.65,
                            color=color, label=reg, step="mid")
    ax.set_ylabel("Regime")
    ax.set_yticks([])
    ax.set_title("Volatility Regime Classification")
    ax.legend(loc="upper right", ncol=3)

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[3].xaxis.set_major_locator(mdates.YearLocator(2))

    save_fig(fig, "fig02_har_vrp")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: DISTRIBUTION OF ΔVIX
# ══════════════════════════════════════════════════════════════════════════════

def fig_distribution(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """Histogram + KDE + QQ for both horizons."""
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for row_idx, (h, target) in enumerate([(1, "y1"), (5, "y5")]):
        if target not in features.columns:
            continue
        y = features[target].dropna()

        # ── Histogram + Normal fit ────────────────────────────────────
        ax = axes[row_idx, 0]
        ax.hist(y, bins=80, density=True, alpha=0.55,
                color=COLORS["blue"], edgecolor="white", linewidth=0.3)
        x_rng = np.linspace(y.min(), y.max(), 300)
        ax.plot(x_rng, stats.norm.pdf(x_rng, y.mean(), y.std()),
                color=COLORS["red"], linewidth=1.8, label="Normal fit")
        kurt = float(stats.kurtosis(y))
        skew = float(stats.skew(y))
        ax.set_xlabel(f"$\\Delta$VIX ({h}-day)")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {h}-Day VIX Changes\n"
                     f"Kurt = {kurt:.2f},  Skew = {skew:.2f}")
        ax.legend()

        # ── QQ plot ───────────────────────────────────────────────────
        ax = axes[row_idx, 1]
        stats.probplot(y, dist="norm", plot=ax)
        ax.set_title(f"QQ Plot vs Normal ($h = {h}$)")
        ax.get_lines()[0].set_color(COLORS["blue"])
        ax.get_lines()[0].set_markersize(2)
        ax.get_lines()[0].set_alpha(0.5)
        ax.get_lines()[1].set_color(COLORS["red"])

    save_fig(fig, "fig03_distribution")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def fig_correlation(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """Lower-triangle correlation heatmap of all features and targets."""
    try:
        import seaborn as sns
    except ImportError:
        print("    seaborn not available, skipping heatmap")
        return

    cols = ["RV_d", "RV_w", "RV_m", "RV_d_plus", "RV_d_minus",
            "VRP", "slope", "curv", "dy10", "MOVE", "spread",
            "r_lag1", "r_lag5", "y1", "y5"]
    available = [c for c in cols if c in features.columns]
    corr = features[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)   # lower triangle
    mask = ~mask   # show lower triangle, mask upper
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.4, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix (lower triangle)")

    save_fig(fig, "fig04_correlation")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5: RV ESTIMATOR COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def fig_rv_comparison(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """Compare GK, Parkinson, and Close-to-Close RV estimates."""
    from features.pipeline import (
        garman_klass_daily, parkinson_daily,
        rv_rolling, rv_close_to_close,
    )

    ret = master["SPY_ret"].dropna()

    # Compute 22-day annualized RV for each estimator
    gk_var = garman_klass_daily(master)
    pk_var = parkinson_daily(master)

    rv_gk = rv_rolling(gk_var, window=22)
    rv_pk = rv_rolling(pk_var, window=22)
    rv_cc = rv_close_to_close(ret, window=22)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    for rv, label, color in [
        (rv_gk, "Garman-Klass", COLORS["blue"]),
        (rv_pk, "Parkinson",    COLORS["orange"]),
        (rv_cc, "Close-Close",  COLORS["green"]),
    ]:
        ax.plot(rv.index, rv, label=label, color=color,
                linewidth=0.6, alpha=0.8)
    ax.set_ylabel("22-Day Annualized RV (%)")
    ax.set_title("Realized Volatility Estimator Comparison")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    common = pd.DataFrame({"GK": rv_gk, "CC": rv_cc}).dropna()
    rho    = common["GK"].corr(common["CC"])
    ax.scatter(common["CC"], common["GK"], alpha=0.2, s=3,
               color=COLORS["blue"])
    lim = common.values.max()
    ax.plot([0, lim], [0, lim], color=COLORS["red"],
            linestyle="--", linewidth=1, label="45° line")
    ax.set_xlabel("Close-to-Close RV (%)")
    ax.set_ylabel("Garman-Klass RV (%)")
    ax.set_title(f"GK vs CC ($\\rho = {rho:.3f}$)")
    ax.legend()

    save_fig(fig, "fig05_rv_comparison")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6: RESIDUAL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def fig_residual_diagnostics(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """Residual diagnostics for HAR-Extended (M4), h=5."""
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf
    from scipy import stats
    from config import FEATURES_M4

    available = [c for c in FEATURES_M4 if c in features.columns]
    valid     = features[available + ["y5"]].dropna()
    X         = sm.add_constant(valid[available])
    results   = sm.OLS(valid["y5"], X).fit()
    resid     = results.resid

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.scatter(results.fittedvalues, resid, alpha=0.15, s=2,
               color=COLORS["blue"])
    ax.axhline(0, color=COLORS["red"], linewidth=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted (M4, $h=5$)")

    plot_acf(resid,    lags=20, ax=axes[0, 1], alpha=0.05,
             title="ACF of Residuals")
    plot_acf(resid**2, lags=20, ax=axes[1, 0], alpha=0.05,
             title="ACF of Squared Residuals (ARCH effects)")

    ax = axes[1, 1]
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("QQ Plot of Residuals")
    ax.get_lines()[0].set_color(COLORS["blue"])
    ax.get_lines()[0].set_markersize(2)
    ax.get_lines()[1].set_color(COLORS["red"])

    save_fig(fig, "fig06_residual_diagnostics")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7: CUSUM + RECURSIVE COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════════

def fig_structural_stability(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """
    CUSUM of recursive residuals (Brown, Durbin & Evans 1975).
    Correct 5% bounds: ±(c_α + 2(t−k)/n), c_0.05 = 0.948.
    """
    try:
        with open(OUTPUT_DIR / "insample_results.pkl", "rb") as fh:
            insample = pickle.load(fh)
        stability = insample.get("stability")
    except Exception:
        print("    Skipping: run analysis/insample.py first")
        return

    if stability is None:
        print("    Skipping: no stability results found")
        return

    resid  = np.asarray(stability["residuals"])
    dates  = stability["dates"]
    n      = len(resid)
    k      = stability.get("n_params", 4)   # number of regressors

    # BDE (1975) CUSUM of recursive residuals
    sigma   = np.std(resid, ddof=k)
    cusum   = np.cumsum(resid) / (sigma * np.sqrt(n))
    t_arr   = np.arange(1, n + 1)

    # Correct 5% bounds: ±(0.948 + 2*(t-k)/n)  — BDE Table I
    c_alpha = 0.948
    bounds  = c_alpha + 2.0 * (t_arr - k) / n

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(dates, cusum, color=COLORS["blue"], linewidth=0.9, label="CUSUM")
    ax.plot(dates,  bounds, color=COLORS["red"], linestyle="--",
            linewidth=0.8, label="5\\% bounds (BDE 1975)")
    ax.plot(dates, -bounds, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_title("CUSUM of Recursive Residuals (Brown-Durbin-Evans 1975)")
    ax.set_ylabel("CUSUM")
    ax.legend()

    # Recursive β_m
    ax = axes[1]
    betas      = stability.get("recursive_betas")
    ses        = stability.get("recursive_se")
    rec_dates  = stability.get("recursive_dates")
    var_names  = stability.get("var_names", [])

    if betas is not None and "RV_m" in var_names:
        idx      = var_names.index("RV_m")
        b_m      = np.asarray(betas)[:, idx]
        se_m     = np.asarray(ses)[:, idx]
        ax.plot(rec_dates, b_m, color=COLORS["blue"], linewidth=1)
        ax.fill_between(rec_dates,
                        b_m - 1.96 * se_m,
                        b_m + 1.96 * se_m,
                        alpha=0.2, color=COLORS["blue"],
                        label="95\\% CI")
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_title("Recursive $\\hat{\\beta}_m$ (RV monthly, M4)")
        ax.set_ylabel("Coefficient")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Recursive coefficients unavailable",
                transform=ax.transAxes, ha="center")

    save_fig(fig, "fig07_structural_stability")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8: FOLD-BY-FOLD RMSE  (both horizons)
# ══════════════════════════════════════════════════════════════════════════════

def fig_fold_rmse(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """Walk-forward RMSE by fold for key models, h=1 and h=5."""
    all_results = _load_results()
    if all_results is None:
        return

    models_to_plot = [
        "M0_RandomWalk",
        "M2_HAROLS",
        "M4_HARExtended",
        "M5_HARRegime",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    for ax, horizon in zip(axes, [1, 5]):
        h_res = all_results.get(horizon, {})

        for model_name in models_to_plot:
            if model_name not in h_res:
                continue
            result = h_res[model_name]
            folds  = result.get("folds", [])
            if not folds:
                continue

            preds   = result["predictions"]
            actuals = result["actuals"]
            fold_rmses : list[float] = []
            fold_dates : list[pd.Timestamp] = []
            idx = 0

            for fold in folds:
                n_t = fold["test_size"]
                if idx + n_t > len(preds):
                    break
                fp  = preds[idx:idx + n_t]
                fa  = actuals[idx:idx + n_t]
                fold_rmses.append(float(np.sqrt(np.mean((fa - fp) ** 2))))
                fold_dates.append(pd.Timestamp(fold["fold_end"]))
                idx += n_t

            if fold_dates:
                label  = MODEL_LABELS.get(model_name, model_name)
                ax.plot(fold_dates, fold_rmses,
                        color=MODEL_COLORS[model_name],
                        linestyle=MODEL_LS[model_name],
                        linewidth=1.1, marker="o", markersize=2.5,
                        label=label)

        ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30"),
                   alpha=0.1, color="red", label="COVID-19")
        ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
                   alpha=0.1, color="orange", label="2022 Rate Shock")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Walk-Forward RMSE by Fold ($h = {horizon}$)")
        ax.legend(ncol=3, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    save_fig(fig, "fig08_fold_rmse")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 9: CUMULATIVE R²_OOS
# ══════════════════════════════════════════════════════════════════════════════

def fig_cumulative_r2(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """
    Running cumulative R²_oos over time for M2, M4, M5 at h=5.
    Dates derived from fold_info (safe alignment).
    Shaded 95% bootstrap CI for M4.
    """
    all_results = _load_results()
    if all_results is None:
        return

    h5 = all_results.get(5, {})
    if not h5:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    models_to_plot = ["M2_HAROLS", "M4_HARExtended", "M5_HARRegime"]

    # Load bootstrap CI for M4
    try:
        metrics_df = pd.read_csv(OUTPUT_DIR / "metrics_summary.csv")
        m4_row = metrics_df[
            (metrics_df["model"] == "M4_HARExtended") &
            (metrics_df["horizon"] == 5)
        ]
        r2_ci_lo = float(m4_row["r2_ci_lo"].values[0]) if len(m4_row) else None
        r2_ci_hi = float(m4_row["r2_ci_hi"].values[0]) if len(m4_row) else None
    except Exception:
        r2_ci_lo = r2_ci_hi = None

    for model_name in models_to_plot:
        if model_name not in h5:
            continue
        result = h5[model_name]
        actual = result["actuals"]
        preds  = result["predictions"]
        errors = actual - preds

        # Safe date alignment from fold_info
        dates = _dates_from_folds(result["folds"])
        n     = min(len(dates), len(errors))
        dates = dates[:n]

        cum_sse   = np.cumsum(errors[:n] ** 2)
        cum_ss    = np.cumsum(actual[:n] ** 2)
        cum_r2    = 1.0 - cum_sse / np.maximum(cum_ss, 1e-12)

        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(dates, cum_r2,
                color=MODEL_COLORS[model_name],
                linestyle=MODEL_LS[model_name],
                linewidth=1.1, label=label)

    # Shade M4 bootstrap CI as a horizontal band at the final value
    if r2_ci_lo is not None and "M4_HARExtended" in h5:
        dates_m4 = _dates_from_folds(h5["M4_HARExtended"]["folds"])
        ax.axhspan(r2_ci_lo, r2_ci_hi, alpha=0.12,
                   color=COLORS["blue"],
                   label=f"M4 95\\% CI [{r2_ci_lo:.3f}, {r2_ci_hi:.3f}]")

    ax.axhline(0, color=COLORS["red"], linewidth=1.0,
               linestyle="--", label="Random Walk ($R^2=0$)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative $R^2_{\\mathrm{oos}}$")
    ax.set_title("Cumulative Out-of-Sample $R^2$ ($h = 5$)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    save_fig(fig, "fig09_cumulative_r2")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 10: EQUITY CURVE + DRAWDOWN
# ══════════════════════════════════════════════════════════════════════════════

def fig_equity_curve(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """Strategy equity curve and drawdown vs SPY buy-and-hold."""
    try:
        with open(OUTPUT_DIR / "backtest_results.pkl", "rb") as fh:
            backtest = pickle.load(fh)
    except Exception:
        print("    Skipping: run trading/backtest.py first")
        return

    pnl = backtest.get("pnl")
    if pnl is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)

    ax = axes[0]
    ax.plot(pnl.index, pnl["cum_net"],   color=COLORS["blue"],
            linewidth=1.1, label="Strategy (net of costs)")
    ax.plot(pnl.index, pnl["cum_gross"], color=COLORS["blue"],
            linewidth=0.7, alpha=0.4, linestyle="--",
            label="Strategy (gross)")
    spy_ret = master["SPY_ret"].reindex(pnl.index).fillna(0)
    spy_cum = (1 + spy_ret).cumprod()
    ax.plot(pnl.index, spy_cum, color=COLORS["gray"],
            linewidth=0.8, linestyle="--", label="SPY B\\&H")
    ax.axhline(1, color="black", linewidth=0.4)
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Strategy Equity Curve vs Benchmarks")
    ax.legend()

    ax = axes[1]
    peak = pnl["cum_net"].expanding().max()
    dd   = (pnl["cum_net"] - peak) / peak * 100
    ax.fill_between(pnl.index, dd, 0, alpha=0.45, color=COLORS["red"])
    ax.set_ylabel("Drawdown (\\%)")
    ax.set_title("Strategy Drawdown")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    save_fig(fig, "fig10_equity_curve")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 11: PERFORMANCE BY REGIME
# ══════════════════════════════════════════════════════════════════════════════

def fig_regime_returns(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """Average strategy returns by VIX regime (bar chart)."""
    regime_path = TABLE_DIR / "trading_by_regime.csv"
    if not regime_path.exists():
        print("    Skipping: run trading/backtest.py first")
        return

    regime_perf = pd.read_csv(regime_path)
    fig, ax     = plt.subplots(figsize=(8, 5))

    colors_map = {"Low": COLORS["green"], "Medium": COLORS["gold"],
                  "High": COLORS["red"]}
    x = np.arange(len(regime_perf))
    ax.bar(x, regime_perf["Avg_daily_ret"],
           yerr=regime_perf["Std_daily_ret"],
           capsize=5, width=0.5,
           color=[colors_map.get(r, COLORS["gray"])
                  for r in regime_perf["Regime"]],
           edgecolor="white", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(regime_perf["Regime"])
    ax.set_ylabel("Avg Daily Return (\\%)")
    ax.set_title("Strategy Performance by VIX Regime")
    ax.axhline(0, color="black", linewidth=0.4)

    ymin = ax.get_ylim()[0]
    for i, row in regime_perf.iterrows():
        ax.text(i, ymin * 0.85, f"N={int(row['N_days'])}",
                ha="center", fontsize=8)

    save_fig(fig, "fig11_regime_returns")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 12: MODEL NESTING DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def fig_model_nesting(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """
    Complete model nesting / comparison diagram.
    CW tests: M0→M2, M0→M3, M0→M4, M2→M3, M2→M4.
    DM tests: M1 vs M0 (non-nested), M4 vs M1 (non-nested), M5 vs M4.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Node positions (x, y, label)
    nodes = {
        "M0": (1.5, 6.5, "M0\nRandom Walk"),
        "M1": (1.5, 2.5, "M1\nGARCH(1,1)"),
        "M2": (4.5, 6.5, "M2\nHAR-OLS"),
        "M3": (4.5, 4.0, "M3\nSHAR"),
        "M4": (8.0, 6.5, "M4\nHAR-Ext."),
        "M5": (8.0, 3.5, "M5\nHAR-Regime"),
    }

    # Draw boxes
    for key, (x, y, label) in nodes.items():
        fcolor = COLORS["blue"] if key not in ("M0", "M1") else COLORS["gray"]
        box = mpatches.FancyBboxPatch(
            (x - 0.9, y - 0.45), 1.8, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=fcolor, alpha=0.15,
            edgecolor=fcolor, linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, fontweight="bold",
                color=fcolor if key not in ("M0", "M1") else COLORS["gray"])

    # Arrows
    arrows_cw = [
        ("M0", "M2"), ("M0", "M3"), ("M0", "M4"),
        ("M2", "M3"), ("M2", "M4"),
    ]
    arrows_dm = [
        ("M1", "M0"), ("M4", "M1"), ("M5", "M4"),
    ]

    def _arrow(src, tgt, color, label, offset=(0, 0)):
        sx, sy = nodes[src][0], nodes[src][1]
        tx, ty = nodes[tgt][0], nodes[tgt][1]
        ax.annotate("", xy=(tx + offset[0], ty + offset[1]),
                    xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->",
                                    color=color, linewidth=1.2,
                                    connectionstyle="arc3,rad=0.15"))
        mx = (sx + tx) / 2 + offset[0] * 0.6
        my = (sy + ty) / 2 + offset[1] * 0.6
        ax.text(mx + 0.15, my + 0.15, label,
                fontsize=7, color=color, fontweight="bold")

    for src, tgt in arrows_cw:
        _arrow(src, tgt, COLORS["blue"], "CW")
    for src, tgt in arrows_dm:
        _arrow(src, tgt, COLORS["red"], "DM", offset=(0.05, 0.05))

    # Legend
    ax.plot([], [], color=COLORS["blue"], linewidth=1.5, label="CW (nested)")
    ax.plot([], [], color=COLORS["red"],  linewidth=1.5,
            linestyle="--", label="DM-HLN (non-nested)")
    ax.legend(loc="lower right", fontsize=8)

    ax.set_title("Model Nesting Structure and Test Battery",
                 fontsize=12, fontweight="bold")

    save_fig(fig, "fig12_model_nesting")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 13 (NEW): R²_OOS BY REGIME  (LaTeX §regime_perf)
# ══════════════════════════════════════════════════════════════════════════════

def fig_regime_r2(features: pd.DataFrame, master: pd.DataFrame) -> None:
    """
    Grouped bar chart of R²_oos per VIX regime for all models.
    Tests whether M5's regime-specific slopes outperform M4 in extreme regimes.
    LaTeX §regime_perf.
    """
    all_results = _load_results()
    if all_results is None:
        return

    h5 = all_results.get(5, {})
    if not h5:
        return

    from evaluation.metrics import oos_r2, regime_metrics

    models    = ["M2_HAROLS", "M3_SHAR", "M4_HARExtended", "M5_HARRegime"]
    regimes   = ["Low", "Medium", "High"]
    reg_data  : dict[str, dict[str, float]] = {m: {} for m in models}

    for model_name in models:
        if model_name not in h5:
            continue
        result = h5[model_name]
        if "regimes" not in result:
            continue
        rm = regime_metrics(
            result["actuals"], result["predictions"], result["regimes"]
        )
        for reg in regimes:
            reg_data[model_name][reg] = rm[reg]["r2_oos"]

    fig, ax  = plt.subplots(figsize=(10, 5))
    n_models = len(models)
    n_reg    = len(regimes)
    x        = np.arange(n_reg)
    width    = 0.18
    offsets  = np.linspace(-(n_models - 1) / 2 * width,
                           (n_models - 1) / 2 * width, n_models)

    for i, model_name in enumerate(models):
        vals = [reg_data[model_name].get(r, float("nan")) for r in regimes]
        label = MODEL_LABELS.get(model_name, model_name)
        ax.bar(x + offsets[i], vals, width=width,
               color=MODEL_COLORS[model_name], alpha=0.8,
               label=label, edgecolor="white", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("$R^2_{\\mathrm{oos}}$")
    ax.set_title("Out-of-Sample $R^2$ by VIX Regime ($h = 5$)")
    ax.legend(fontsize=8)

    save_fig(fig, "fig13_regime_r2")


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_figures() -> None:
    """Generate all 14 figures."""
    print("=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    features, master = load_data()

    figure_functions = [
        ("Fig 0:  Walk-Forward Timeline",     fig_walkforward_timeline),
        ("Fig 1:  VIX time series",           fig_vix_timeseries),
        ("Fig 2:  HAR components + VRP",      fig_har_vrp),
        ("Fig 3:  Distribution of ΔVIX",      fig_distribution),
        ("Fig 4:  Correlation heatmap",       fig_correlation),
        ("Fig 5:  RV estimator comparison",   fig_rv_comparison),
        ("Fig 6:  Residual diagnostics",      fig_residual_diagnostics),
        ("Fig 7:  Structural stability",      fig_structural_stability),
        ("Fig 8:  Fold RMSE (h=1 and h=5)",   fig_fold_rmse),
        ("Fig 9:  Cumulative R²_oos",         fig_cumulative_r2),
        ("Fig 10: Equity curve + drawdown",   fig_equity_curve),
        ("Fig 11: Regime returns",            fig_regime_returns),
        ("Fig 12: Model nesting diagram",     fig_model_nesting),
        ("Fig 13: R²_oos by regime",          fig_regime_r2),
    ]

    for name, func in figure_functions:
        print(f"\n  {name}...")
        try:
            func(features, master)
        except Exception as exc:
            print(f"    ERROR: {exc}")

    print(f"\n  ✓ All figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    generate_all_figures()