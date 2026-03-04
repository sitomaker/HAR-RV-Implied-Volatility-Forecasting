"""
config.py — Global constants and paths.
Single source of truth for all parameters used across the project.
"""
from pathlib import Path
import numpy as np

# ── Reproducibility ──────────────────────────────────────────
SEED = 42

# ═══════════════════════════════════════════════════════════════
#  FRED API KEY
# ═══════════════════════════════════════════════════════════════
FRED_API_KEY = "c556555b7139a057759c545216ec6243"

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR   = PROJECT_ROOT / "output"
TABLE_DIR    = OUTPUT_DIR / "tables"
FIGURE_DIR   = OUTPUT_DIR / "figures"

for d in [DATA_RAW, DATA_PROC, OUTPUT_DIR, TABLE_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Sample Period ────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"

# ── Trading Calendar ─────────────────────────────────────────
TRADING_DAYS_YEAR = 252

# ── Yahoo Finance Tickers ────────────────────────────────────
YAHOO_TICKERS = {
    "SPY":   "SPY",
    "VIX":   "^VIX",
    "VIX9D": "^VIX9D",
    "VIX3M": "^VIX3M",
    "VIX6M": "^VIX6M",
    "VXX":   "VXX",
}

# ── FRED Series ──────────────────────────────────────────────
FRED_SERIES = {
    "DGS10":        "DGS10",
    "BAMLC0A1CAAA": "BAMLC0A1CAAA",
    "BAMLC0A4CBBB": "BAMLC0A4CBBB",
    "BAMLMKV2Y5Y":  "BAMLMKV2Y5Y",
}

# ── HAR Windows ──────────────────────────────────────────────
HAR_DAILY   = 1
HAR_WEEKLY  = 5
HAR_MONTHLY = 22

# ── VIX Regime Thresholds (fixed ex-ante) ────────────────────
REGIME_LOW_UPPER  = 15.0
REGIME_MED_UPPER  = 25.0
REGIME_LABELS     = ["Low", "Medium", "High"]
REGIME_BINS       = [-np.inf, REGIME_LOW_UPPER, REGIME_MED_UPPER, np.inf]

# ── Walk-Forward Parameters ──────────────────────────────────
WF_T_MIN  = 500
WF_TAU    = 63

# ── Ridge Hyperparameter Grid ────────────────────────────────
RIDGE_LAMBDAS  = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
INNER_CV_FOLDS = 5

# ── Newey-West HAC Lags ──────────────────────────────────────
def andrews_hac_lags(T: int) -> int:
    """Andrews (1991) optimal lag selection."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))

# ── Forecast Horizons ────────────────────────────────────────
HORIZONS = [1, 5]

# ── Transaction Costs ────────────────────────────────────────
COST_SPREAD     = 0.0010
COST_SLIPPAGE   = 0.0010
COST_COMMISSION = 0.0002
COST_TOTAL      = COST_SPREAD + COST_SLIPPAGE + COST_COMMISSION

# ── Statistical Testing ──────────────────────────────────────
BH_FDR_LEVEL        = 0.10
SIGNIFICANCE_LEVELS = [0.01, 0.05, 0.10]

# ── RV Estimators ────────────────────────────────────────────
RV_ESTIMATORS = ["gk", "parkinson", "cc"]
RV_PRIMARY    = "gk"

# ══════════════════════════════════════════════════════════════
#  TARGET SPECIFICATION (NUEVO)
# ══════════════════════════════════════════════════════════════
# Opciones: "arithmetic", "log", "winsorized"
TARGET_TYPE = "arithmetic"

# Targets disponibles por tipo
TARGET_COLUMNS = {
    "arithmetic": {"y1": "y1", "y5": "y5"},
    "log":        {"y1": "y1_log", "y5": "y5_log"},
    "winsorized": {"y1": "y1_win", "y5": "y5_win"},
}

# Winsorización: percentiles para expanding window
WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995
WINSOR_MIN_PERIODS = 100

# ── Feature Column Groups ────────────────────────────────────
FEATURES_M2 = ["RV_d", "RV_w", "RV_m"]

FEATURES_M3 = ["RV_d_plus", "RV_d_minus", "RV_w", "RV_m"]

FEATURES_M4 = [
    "RV_d", "RV_w", "RV_m",
    "VRP", "slope", "curv",
    "dy10", "MOVE", "spread",
    "r_lag1", "r_lag5",
]

FEATURES_M5_BASE = ["RV_d", "RV_w", "RV_m", "VRP"]

FEATURES_M5_INTERACT = ["VRP"]