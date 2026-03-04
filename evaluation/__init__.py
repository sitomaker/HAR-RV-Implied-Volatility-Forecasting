"""
evaluation/ — Forecast evaluation package.

Modules
-------
losses        Loss functions: MSE, RMSE, MAE, QLIKE, loss_differential
metrics       Performance metrics: R²_oos (+ bootstrap CI), HitRate,
              Pesaran-Timmermann test, regime-conditional breakdown
tests         Statistical tests: DM-HLN, Clark-West, Mincer-Zarnowitz,
              Benjamini-Hochberg FDR correction, Andrews bandwidth
walk_forward  Expanding-window walk-forward evaluator for M0–M5;
              inner CV for Ridge λ; full test battery; BH-corrected table
"""
from .losses        import mse, rmse, mae, qlike, loss_differential
from .metrics       import (
    oos_r2, oos_r2_bootstrap_ci,
    hit_rate, pesaran_timmermann,
    regime_metrics,
)
from .tests         import (
    diebold_mariano_hln, clark_west,
    mincer_zarnowitz, benjamini_hochberg,
    andrews_bandwidth,
)

__all__ = [
    # losses
    "mse", "rmse", "mae", "qlike", "loss_differential",
    # metrics
    "oos_r2", "oos_r2_bootstrap_ci",
    "hit_rate", "pesaran_timmermann",
    "regime_metrics",
    # tests
    "diebold_mariano_hln", "clark_west",
    "mincer_zarnowitz", "benjamini_hochberg",
    "andrews_bandwidth",
]