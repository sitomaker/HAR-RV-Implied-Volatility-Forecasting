"""
main.py — Master pipeline runner.

Executes all stages in order with a fixed random seed.
Every number in the paper is reproducible from a single run of this script.

Usage:
    python main.py                    # full pipeline
    python main.py --skip-download    # skip data download (use cached)
    python main.py --only-features    # only rebuild features
    python main.py --only-eval        # only run walk-forward evaluation
    python main.py --only-figures     # only generate figures (after full run)
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np

# ── Set global seed ONCE here (not at import time in config) ──
from config import SEED
np.random.seed(SEED)

import random
random.seed(SEED)


def stage(n: int, name: str):
    print(f"\n{'='*60}")
    print(f"STAGE {n}: {name}")
    print(f"{'='*60}")


def run_pipeline(skip_download: bool = False,
                 only_features: bool = False,
                 only_eval:     bool = False,
                 only_figures:  bool = False):

    t0 = time.time()

    # ── Stage 1: Data Download ────────────────────────────────
    if not skip_download and not only_features and not only_eval and not only_figures:
        stage(1, "DATA DOWNLOAD")
        from data.download import download_all
        download_all()
    else:
        print("\n  [Skipping Stage 1: Data Download]")

    # ── Stage 2: Data Cleaning ────────────────────────────────
    if not only_features and not only_eval and not only_figures:
        stage(2, "DATA CLEANING")
        from data.clean import build_master_frame
        master = build_master_frame()
        print(f"  Master frame: {master.shape[0]} rows, "
              f"{master.shape[1]} columns")
    else:
        print("\n  [Skipping Stage 2: Data Cleaning]")

    # ── Stage 3: Feature Engineering ─────────────────────────
    if not only_eval and not only_figures:
        stage(3, "FEATURE ENGINEERING")
        from features.pipeline import build_and_save_all
        build_and_save_all()
    else:
        print("\n  [Skipping Stage 3: Feature Engineering]")

    # ── Stage 4a: EDA ─────────────────────────────────────────
    if not only_features and not only_eval and not only_figures:
        stage(4, "EXPLORATORY DATA ANALYSIS")
        try:
            from analysis.eda import run_eda
            run_eda()
        except Exception as e:
            print(f"  EDA failed (non-fatal): {e}")

    # ── Stage 4b: In-Sample Estimation ────────────────────────
    if not only_features and not only_eval and not only_figures:
        stage(4, "IN-SAMPLE ESTIMATION")
        try:
            from analysis.insample import run_insample
            run_insample()
        except Exception as e:
            print(f"  In-sample failed (non-fatal): {e}")

    # ── Stage 5: Walk-Forward Evaluation ──────────────────────
    if not only_features and not only_figures:
        stage(5, "WALK-FORWARD EVALUATION")
        from evaluation.walk_forward import run_all_models, evaluate_all
        from config import OUTPUT_DIR
        import pickle

        all_results = run_all_models()
        metrics_df  = evaluate_all(all_results)

        metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
        with open(OUTPUT_DIR / "all_results.pkl", "wb") as f:
            pickle.dump(all_results, f)

        print(f"\n  Results saved to {OUTPUT_DIR}")
        display_cols = [
            "horizon", "model", "n_oos",
            "rmse", "mae", "r2_oos", "r2_ci_lo", "r2_ci_hi",
            "hit_rate", "dm_mse_stat", "dm_mse_pval",
            "cw_stat", "cw_pval",
        ]
        avail = [c for c in display_cols if c in metrics_df.columns]
        print(metrics_df[avail].to_string(index=False, float_format="%.4f"))

    # ── Stage 6: Robustness ────────────────────────────────────
    if not only_features and not only_eval and not only_figures:
        stage(6, "ROBUSTNESS CHECKS")
        try:
            from analysis.robustness import run_robustness
            run_robustness()
        except Exception as e:
            print(f"  Robustness failed (non-fatal): {e}")
            import traceback; traceback.print_exc()

    # ── Stage 7: Figures ───────────────────────────────────────
    if not only_features and not only_eval:
        stage(7, "GENERATING FIGURES")
        try:
            from reporting.figures import generate_all_figures
            generate_all_figures()
        except Exception as e:
            print(f"  Figure generation failed (non-fatal): {e}")
            import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE  ({elapsed/60:.1f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAR-RV IV Forecasting Pipeline")
    parser.add_argument("--skip-download",  action="store_true",
                        help="Skip Stage 1 (use cached data)")
    parser.add_argument("--only-features",  action="store_true",
                        help="Only run Stage 3 (feature engineering)")
    parser.add_argument("--only-eval",      action="store_true",
                        help="Only run Stage 5 (walk-forward evaluation)")
    parser.add_argument("--only-figures",   action="store_true",
                        help="Only run Stage 7 (generate figures)")
    args = parser.parse_args()

    run_pipeline(
        skip_download=args.skip_download,
        only_features=args.only_features,
        only_eval=args.only_eval,
        only_figures=args.only_figures,
    )