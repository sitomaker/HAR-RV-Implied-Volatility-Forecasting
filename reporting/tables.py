from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, TABLE_DIR


# ── Shared utilities ──────────────────────────────────────────────────────────

MODEL_LABELS: dict[str, str] = {
    "M0_RandomWalk":   "M0 (RW)",
    "M1_GARCH":        "M1 (GARCH)",
    "M2_HAROLS":       "M2 (HAR-OLS)",
    "M3_SHAR":         "M3 (SHAR)",
    "M4_HARExtended":  "M4 (HAR-Ext.)",
    "M5_HARRegime":    "M5 (HAR-Regime)",
}


def sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "$^{***}$"
    if p < 0.05:
        return "$^{**}$"
    if p < 0.10:
        return "$^{*}$"
    return ""


def fmt(val: float, decimals: int = 4) -> str:
    """Format a number to fixed decimals, or '---' for NaN."""
    if pd.isna(val):
        return "---"
    return f"{val:.{decimals}f}"


def _model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", r"\_"))


def _write_tex(lines: list[str], filename: str) -> None:
    tex = "\n".join(lines)
    (TABLE_DIR / filename).write_text(tex, encoding="utf-8")
    print(f"    ✓ {filename}")


def _load_metrics() -> pd.DataFrame | None:
    csv_path = OUTPUT_DIR / "metrics_summary.csv"
    if not csv_path.exists():
        print("    Skipping: run evaluation first")
        return None
    return pd.read_csv(csv_path)



#  TABLE 1: DESCRIPTIVE STATISTICS
def table_descriptive() -> None:
    """LaTeX fragment: descriptive statistics."""
    csv_path = TABLE_DIR / "descriptive_stats.csv"
    if not csv_path.exists():
        print("    Skipping: run analysis/eda.py first")
        return

    df    = pd.read_csv(csv_path)
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Variable & $N$ & Mean & Std Dev & Min & Q25 & Median & Q75 & Max \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Variable']} & {int(row['N'])} & "
            f"{fmt(row['Mean'])} & {fmt(row['Std Dev'])} & "
            f"{fmt(row['Min'])} & {fmt(row['Q25'])} & "
            f"{fmt(row['Median'])} & {fmt(row['Q75'])} & "
            f"{fmt(row['Max'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_tex(lines, "tab_descriptive.tex")



#  TABLE 2: STATIONARITY TESTS
def table_stationarity() -> None:
    """LaTeX fragment: ADF + KPSS stationarity tests."""
    csv_path = TABLE_DIR / "stationarity_tests.csv"
    if not csv_path.exists():
        print("    Skipping: run analysis/eda.py first")
        return

    df    = pd.read_csv(csv_path)
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Series & ADF Stat & ADF $p$ & KPSS Stat & KPSS $p$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Series']} & "
            f"{fmt(row['ADF Stat'])}{sig_stars(row['ADF p-val'])} & "
            f"{fmt(row['ADF p-val'])} & "
            f"{fmt(row['KPSS Stat'])}{sig_stars(row['KPSS p-val'])} & "
            f"{fmt(row['KPSS p-val'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_tex(lines, "tab_stationarity.tex")



#  TABLES 3–5: IN-SAMPLE COEFFICIENTS
def table_insample_coefficients() -> None:
    """LaTeX fragments for HAR-OLS, SHAR, HAR-Extended coefficient tables."""
    import pickle

    try:
        with open(OUTPUT_DIR / "insample_results.pkl", "rb") as fh:
            insample = pickle.load(fh)
    except Exception:
        print("    Skipping: run analysis/insample.py first")
        return

    _write_coef_table(insample["har_ols"],      "tab_insample_har_ols.tex")
    _write_coef_table(insample["shar"],         "tab_insample_shar.tex",
                      extra_rows_fn=_shar_extra_rows)
    _write_coef_table(insample["har_extended"], "tab_insample_har_ext.tex")


def _write_coef_table(results: dict, filename: str,
                      extra_rows_fn=None) -> None:
    """Write a coefficient table with both horizons side by side."""
    horizons = sorted(results.keys())
    n_h      = len(horizons)

    col_spec = "l" + "cc" * n_h
    lines    = [f"\\begin{{tabular}}{{{col_spec}}}", r"\toprule"]

    if n_h == 2:
        lines.append(
            r"& \multicolumn{2}{c}{$h = 1$} & \multicolumn{2}{c}{$h = 5$} \\"
        )
        lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
        lines.append(r"Variable & Coef. & Std.\ Err. & Coef. & Std.\ Err. \\")
    else:
        lines.append(r"Variable & Coef. & Std.\ Err. \\")

    lines.append(r"\midrule")

    h0        = horizons[0]
    var_names = results[h0]["var_names"]

    for i, name in enumerate(var_names):
        latex_name = name.replace("_", r"\_")
        if n_h == 2:
            h1, h5 = horizons[0], horizons[1]
            parts  = []
            for hk in (h1, h5):
                c = results[hk]["coefficients"][i]
                s = results[hk]["std_errors"][i]
                p = results[hk]["p_values"][i]
                parts.append(f"{fmt(c)}{sig_stars(p)} & ({fmt(s)})")
            lines.append(f"{latex_name} & {' & '.join(parts)} \\\\")
        else:
            c = results[h0]["coefficients"][i]
            s = results[h0]["std_errors"][i]
            p = results[h0]["p_values"][i]
            lines.append(f"{latex_name} & {fmt(c)}{sig_stars(p)} & ({fmt(s)}) \\\\")

    lines.append(r"\midrule")

    if n_h == 2:
        h1, h5 = horizons[0], horizons[1]
        lines.append(
            f"$R^2$ & \\multicolumn{{2}}{{c}}{{{fmt(results[h1]['r_squared'])}}} & "
            f"\\multicolumn{{2}}{{c}}{{{fmt(results[h5]['r_squared'])}}} \\\\"
        )
        lines.append(
            f"Adj.\\ $R^2$ & \\multicolumn{{2}}{{c}}{{{fmt(results[h1]['adj_r_squared'])}}} & "
            f"\\multicolumn{{2}}{{c}}{{{fmt(results[h5]['adj_r_squared'])}}} \\\\"
        )
        lines.append(
            f"$N$ & \\multicolumn{{2}}{{c}}{{{results[h1]['n_obs']}}} & "
            f"\\multicolumn{{2}}{{c}}{{{results[h5]['n_obs']}}} \\\\"
        )
        for diag_key, diag_label in [
            ("lb_pval_10",  "LB $Q(10)$ $p$"),
            ("arch_lm_pval", "ARCH LM $p$"),
        ]:
            v1 = results[h1].get(diag_key, np.nan)
            v5 = results[h5].get(diag_key, np.nan)
            lines.append(
                f"{diag_label} & \\multicolumn{{2}}{{c}}{{{fmt(v1)}}} & "
                f"\\multicolumn{{2}}{{c}}{{{fmt(v5)}}} \\\\"
            )
    else:
        lines.append(
            f"$R^2$ & \\multicolumn{{2}}{{c}}{{{fmt(results[h0]['r_squared'])}}} \\\\"
        )
        lines.append(
            f"$N$ & \\multicolumn{{2}}{{c}}{{{results[h0]['n_obs']}}} \\\\"
        )

    if extra_rows_fn is not None:
        lines.extend(extra_rows_fn(results, n_h))

    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_tex(lines, filename)


def _shar_extra_rows(results: dict, n_h: int) -> list[str]:
    """Wald test H0: β_d⁺ = β_d⁻ for SHAR table (LaTeX §shar)."""
    lines = [r"\midrule"]
    if n_h == 2:
        h1, h5 = sorted(results.keys())
        for hk, hlabel in [(h1, "h=1"), (h5, "h=5")]:
            f  = results[hk].get("f_stat_asymmetry", np.nan)
            p  = results[hk].get("f_pval_asymmetry", np.nan)
            lines.append(
                f"$F$-test ($\\beta_d^+ = \\beta_d^-$), ${hlabel}$ "
                f"& \\multicolumn{{2}}{{c}}{{{fmt(f, 2)} ($p = {fmt(p)}$)}} "
                f"& \\multicolumn{{2}}{{c}}{{{fmt(f, 2)} ($p = {fmt(p)}$)}} \\\\"
            )
    return lines



#  TABLE: VIF
def table_vif() -> None:
    """LaTeX fragment: Variance Inflation Factors."""
    csv_path = TABLE_DIR / "vif.csv"
    if not csv_path.exists():
        return

    df    = pd.read_csv(csv_path)
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Feature & VIF \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        name    = str(row["Feature"]).replace("_", r"\_")
        vif_val = row["VIF"]
        flag    = " $\\dagger$" if vif_val > 10 else ""
        lines.append(f"{name} & {fmt(vif_val, 2)}{flag} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_tex(lines, "tab_vif.tex")



#  TABLE 6: OOS PERFORMANCE
def table_oos_performance() -> None:
    """
    LaTeX fragment: OOS performance metrics.
    Columns: RMSE, MAE, R²_oos [CI], Hit Rate, CW p, DM p (BH-adjusted).
    """
    df = _load_metrics()
    if df is None:
        return

    lines = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"$h$ & Model & RMSE & MAE & $R^2_{\mathrm{oos}}$ [95\% CI] "
        r"& Hit Rate & CW $p$ & DM $p^{\mathrm{BH}}$ \\",
        r"\midrule",
    ]

    for h in [1, 5]:
        subset = df[df["horizon"] == h].copy()
        if subset.empty:
            continue

        first = True
        for _, row in subset.iterrows():
            h_str = f"$h={h}$" if first else ""
            first = False

            model = _model_label(row["model"])

            # R²_oos with bootstrap CI
            r2_lo = row.get("r2_ci_lo", np.nan)
            r2_hi = row.get("r2_ci_hi", np.nan)
            if not pd.isna(r2_lo) and not pd.isna(r2_hi):
                r2_str = (f"{fmt(row['r2_oos'])} "
                          f"[{fmt(r2_lo)}, {fmt(r2_hi)}]")
            else:
                r2_str = fmt(row["r2_oos"])

            # CW p-value (BH if available, else raw)
            cw_p = row.get("cw_pval_bh",
                   row.get("cw_pval", np.nan))
            cw_str = f"{fmt(cw_p)}{sig_stars(cw_p)}"

            # DM p-value (BH-adjusted)
            dm_p = row.get("dm_mse_pval_bh",
                   row.get("dm_mse_pval", np.nan))
            dm_str = f"{fmt(dm_p)}{sig_stars(dm_p)}"

            lines.append(
                f"{h_str} & {model} & "
                f"{fmt(row['rmse'])} & "
                f"{fmt(row.get('mae', np.nan))} & "
                f"{r2_str} & "
                f"{fmt(row.get('hit_rate', np.nan))} & "
                f"{cw_str} & "
                f"{dm_str} \\\\"
            )

        if h == 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\smallskip",
        r"{\footnotesize",
        r"\textit{Notes:} $R^2_{\mathrm{oos}}$ bootstrap CI: 1,000 block",
        r"replications, block $b = \max(2h, 10)$. CW: Clark-West one-sided",
        r"test (nested). DM: Diebold-Mariano HLN two-sided test. All",
        r"$p$-values BH-adjusted at $q_{\mathrm{FDR}} = 0.10$.",
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"}",
    ]
    _write_tex(lines, "tab_oos_performance.tex")



#  TABLE 7: FULL TEST BATTERY 
# Mirrors LaTeX tab:test_battery exactly. Each entry: (model, benchmark, test_type, loss_col)
_TEST_BATTERY = [
    # (model,             benchmark,        test, stat_col,          pval_col,          bh_col)
    ("M1_GARCH",        "M0_RandomWalk",  "DM", "dm_mse_stat", "dm_mse_pval", "dm_mse_pval_bh"),
    ("M2_HAROLS",       "M0_RandomWalk",  "CW", "cw_stat",     "cw_pval",     "cw_pval_bh"),
    ("M3_SHAR",         "M0_RandomWalk",  "CW", "cw_stat",     "cw_pval",     "cw_pval_bh"),
    ("M3_SHAR",         "M2_HAROLS",      "CW", "cw_stat",     "cw_pval",     "cw_pval_bh"),
    ("M4_HARExtended",  "M0_RandomWalk",  "CW", "cw_stat",     "cw_pval",     "cw_pval_bh"),
    ("M4_HARExtended",  "M2_HAROLS",      "CW", "cw_stat",     "cw_pval",     "cw_pval_bh"),
    ("M4_HARExtended",  "M1_GARCH",       "DM", "dm_mse_stat", "dm_mse_pval", "dm_mse_pval_bh"),
    ("M5_HARRegime",    "M4_HARExtended", "DM", "dm_mse_stat", "dm_mse_pval", "dm_mse_pval_bh"),
]


def table_dm_cw() -> None:
    """
    LaTeX fragment: full test battery (LaTeX tab:test_battery).
    One row per (comparison, horizon). BH-adjusted p-values included.
    """
    df = _load_metrics()
    if df is None:
        return

    lines = [
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Comparison & Bench. & Test & $h$ & Stat & $p^{\mathrm{BH}}$ \\",
        r"\midrule",
    ]

    for h in [1, 5]:
        subset  = df[df["horizon"] == h]
        h_first = True

        for (model, bench, test, stat_col, pval_col, bh_col) in _TEST_BATTERY:
            row = subset[subset["model"] == model]
            if row.empty:
                continue
            row = row.iloc[0]

            h_str      = f"$h={h}$" if h_first else ""
            h_first    = False
            model_nice = _model_label(model)
            bench_nice = _model_label(bench)
            stat_val   = row.get(stat_col, np.nan)
            pval_bh    = row.get(bh_col,   row.get(pval_col, np.nan))

            lines.append(
                f"{h_str} & "
                f"{model_nice} vs\\ {bench_nice} & "
                f"{test} & "
                f"$h={h}$ & "
                f"{fmt(stat_val, 3)} & "
                f"{fmt(pval_bh)}{sig_stars(pval_bh)} \\\\"
            )

        if h == 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\smallskip",
        r"{\footnotesize",
        r"\textit{Notes:} DM = Diebold-Mariano with HLN correction,",
        r"two-sided, $t_{T-1}$ critical values.",
        r"CW = Clark-West, one-sided, $z_{1-\alpha}$ critical values.",
        r"All 32 $p$-values BH-adjusted simultaneously at",
        r"$q_{\mathrm{FDR}} = 0.10$.",
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"}",
    ]
    _write_tex(lines, "tab_dm_cw.tex")



#  TABLE 8: MINCER-ZARNOWITZ CALIBRATION
def table_mincer_zarnowitz() -> None:
    """
    LaTeX fragment: MZ calibration results.
    Columns: h, Model, â, b̂, Wald p.   (5 columns, spec = llccc)
    """
    df = _load_metrics()
    if df is None:
        return

    mz_df = df.dropna(subset=["mz_a", "mz_b"])
    if mz_df.empty:
        print("    Skipping MZ table: no results")
        return

    lines = [
        r"\begin{tabular}{llccc}",   # FIXED: was {llcccc} (6 cols, 5 headers)
        r"\toprule",
        r"$h$ & Model & $\hat{a}$ & $\hat{b}$ & Wald $p$ \\",
        r"\midrule",
    ]
    for _, row in mz_df.iterrows():
        lines.append(
            f"$h={int(row['horizon'])}$ & {_model_label(row['model'])} & "
            f"{fmt(row['mz_a'])} & {fmt(row['mz_b'])} & "
            f"{fmt(row['mz_wald_p'])}{sig_stars(row['mz_wald_p'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\smallskip",
        r"{\footnotesize",
        r"\textit{Notes:} MZ regression: $y_{t+h} = a + b\hat{y}_{t+h} + u_t$.",
        r"Wald test $H_0$: $(a, b) = (0, 1)$ with Newey-West HAC",
        r"($q^* = 8$ lags, Andrews 1991). $\hat{b} < 1$: overconfident;",
        r"$\hat{b} > 1$: underconfident; $\hat{a} \neq 0$: unconditional bias.",
        r"}",
    ]
    _write_tex(lines, "tab_mincer_zarnowitz.tex")



#  TABLE 9: R²_OOS BY REGIME 
def table_regime_r2() -> None:
    """
    LaTeX fragment: R²_oos per VIX regime for h=5.
    Tests whether M5 outperforms M4 in extreme regimes (§regime_perf).
    """
    df = _load_metrics()
    if df is None:
        return

    h5   = df[df["horizon"] == 5]
    regs = ["Low", "Medium", "High"]

    # Column names produced by evaluate_all → regime_metrics
    r2_cols = {r: f"r2_oos_{r.lower()}" for r in regs}
    n_cols  = {r: f"n_{r.lower()}"       for r in regs}

    # Check columns exist
    missing = [c for c in r2_cols.values() if c not in df.columns]
    if missing:
        print(f"    Skipping regime R² table: columns {missing} not found")
        return

    models = ["M0_RandomWalk", "M2_HAROLS", "M3_SHAR",
              "M4_HARExtended", "M5_HARRegime"]

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Full Sample & Low & Medium & High \\",
        r"\midrule",
    ]

    for model_name in models:
        row = h5[h5["model"] == model_name]
        if row.empty:
            continue
        row = row.iloc[0]

        r2_full = fmt(row.get("r2_oos", np.nan))
        vals    = [fmt(row.get(r2_cols[r], np.nan)) for r in regs]
        lines.append(
            f"{_model_label(model_name)} & {r2_full} & "
            + " & ".join(vals) + r" \\"
        )

    # Sample sizes (from first model that has them)
    n_row = h5[h5["model"] == "M4_HARExtended"]
    if not n_row.empty:
        n_row = n_row.iloc[0]
        if all(c in n_row.index for c in n_cols.values()):
            n_full = int(n_row.get("n_oos", 0))
            ns     = [str(int(n_row.get(n_cols[r], 0))) for r in regs]
            lines.append(r"\midrule")
            lines.append(
                f"$N$ & {n_full} & " + " & ".join(ns) + r" \\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\smallskip",
        r"{\footnotesize",
        r"\textit{Notes:} $R^2_{\mathrm{oos}}$ (Campbell-Thompson 2008) for $h=5$.",
        r"Regime classification: Low ($\mathrm{VIX} \leq 15$),",
        r"Medium ($15 < \mathrm{VIX} \leq 25$), High ($\mathrm{VIX} > 25$).",
        r"No multiple-testing correction applied to regime metrics",
        r"(exploratory decomposition, §\ref{subsec:regime_perf}).",
        r"}",
    ]
    _write_tex(lines, "tab_regime_r2.tex")



#  TABLE: TRADING PERFORMANCE
def table_trading() -> None:
    """LaTeX fragment: trading strategy performance."""
    csv_path = TABLE_DIR / "trading_performance.csv"
    if not csv_path.exists():
        print("    Skipping: run trading/backtest.py first")
        return

    df = pd.read_csv(csv_path)

    gross_keys = [
        ("total_return_gross",  "Total Return (\\%)"),
        ("annual_return_gross", "Annualized Return (\\%)"),
        ("annual_vol_gross",    "Annualized Volatility (\\%)"),
        ("sharpe_gross",        "Sharpe Ratio"),
        ("sortino_gross",       "Sortino Ratio"),
        ("max_dd_gross",        "Maximum Drawdown (\\%)"),
        ("calmar_gross",        "Calmar Ratio"),
    ]

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Gross & Net of Costs \\",
        r"\midrule",
    ]
    for gk, label in gross_keys:
        nk    = gk.replace("gross", "net")
        g_row = df[df["Metric"] == gk]
        n_row = df[df["Metric"] == nk]
        g_val = g_row["Value"].values[0] if len(g_row) > 0 else np.nan
        n_val = n_row["Value"].values[0] if len(n_row) > 0 else np.nan
        lines.append(f"{label} & {fmt(g_val, 2)} & {fmt(n_val, 2)} \\\\")

    lines.append(r"\midrule")
    for key, label in [
        ("win_rate",        "Win Rate (\\%)"),
        ("n_trades",        "Number of Trades"),
        ("break_even_cost", "Break-Even Cost (\\%)"),
    ]:
        row = df[df["Metric"] == key]
        val = row["Value"].values[0] if len(row) > 0 else np.nan
        if key == "n_trades":
            n_str = str(int(val)) if not pd.isna(val) else "---"
            lines.append(
                f"{label} & \\multicolumn{{2}}{{c}}{{{n_str}}} \\\\"
            )
        else:
            lines.append(
                f"{label} & \\multicolumn{{2}}{{c}}{{{fmt(val, 4)}}} \\\\"
            )

    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_tex(lines, "tab_trading.tex")



#  ROBUSTNESS TABLES
def table_robustness() -> None:
    """LaTeX fragments for all robustness check tables."""
    robustness_files = {
        "robustness_subperiods.csv":     ("tab_rob_subperiods.tex",
                                          ["Period", "RMSE", "R2_oos",
                                           "DM_stat", "DM_pval"]),
        "robustness_vrp.csv":            ("tab_rob_vrp.tex",
                                          ["VRP Definition", "R2_oos", "DM_pval"]),
        "robustness_rv.csv":             ("tab_rob_rv.tex",
                                          ["RV Estimator", "RMSE", "R2_oos",
                                           "DM_pval"]),
        "robustness_target.csv":         ("tab_rob_target.tex",
                                          ["Target", "RMSE", "R2_oos", "DM_pval"]),
        "robustness_window.csv":         ("tab_rob_window.tex",
                                          ["T_min", "RMSE", "R2_oos",
                                           "DM_pval", "N_oos"]),
        "robustness_regularization.csv": ("tab_rob_regularization.tex",
                                          ["Method", "RMSE", "R2_oos",
                                           "Active Features"]),
    }

    for csv_name, (tex_name, cols) in robustness_files.items():
        csv_path = TABLE_DIR / csv_name
        if not csv_path.exists():
            print(f"    Skipping {csv_name}: not found")
            continue

        df         = pd.read_csv(csv_path)
        avail_cols = [c for c in cols if c in df.columns]
        if not avail_cols:
            continue

        col_spec = "l" + "c" * (len(avail_cols) - 1)
        lines    = [f"\\begin{{tabular}}{{{col_spec}}}", r"\toprule"]

        header = " & ".join(
            c.replace("_", r"\_").replace("R2", "$R^2$")
            for c in avail_cols
        )
        lines.append(header + r" \\")
        lines.append(r"\midrule")

        for _, row in df.iterrows():
            vals = []
            for c in avail_cols:
                v = row[c]
                if isinstance(v, (int, np.integer)):
                    vals.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    if "pval" in c.lower():
                        vals.append(f"{fmt(v)}{sig_stars(v)}")
                    else:
                        vals.append(fmt(v))
                else:
                    vals.append(str(v).replace("_", r"\_"))
            lines.append(" & ".join(vals) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}"]
        _write_tex(lines, tex_name)



#  MASTER GENERATOR
def generate_all_tables() -> None:
    """Generate all LaTeX table fragments."""
    print("=" * 60)
    print("GENERATING LATEX TABLES")
    print("=" * 60)

    table_functions = [
        ("Descriptive stats",         table_descriptive),
        ("Stationarity tests",        table_stationarity),
        ("In-sample coefficients",    table_insample_coefficients),
        ("VIF",                       table_vif),
        ("OOS performance",           table_oos_performance),
        ("DM and CW test battery",    table_dm_cw),
        ("Mincer-Zarnowitz",          table_mincer_zarnowitz),
        ("R²_oos by regime (NEW)",    table_regime_r2),
        ("Trading performance",       table_trading),
        ("Robustness checks",         table_robustness),
    ]

    for name, func in table_functions:
        print(f"\n  {name}...")
        try:
            func()
        except Exception as exc:
            print(f"    ERROR: {exc}")

    print(f"\n  ✓ All tables saved to {TABLE_DIR}")


if __name__ == "__main__":
    generate_all_tables()
