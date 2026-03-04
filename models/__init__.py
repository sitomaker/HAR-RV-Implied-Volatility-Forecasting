"""
models/ — Model zoo for HAR-RV implied volatility forecasting.

Six specifications (M0–M5) form a hierarchy of progressively richer
information sets, all sharing a common fit/predict interface via BaseModel.

    M0  RandomWalk   — zero-change benchmark (eq. 1)
    M1  GARCHModel   — GARCH(1,1) QMLE physical-measure benchmark (eqs. 2–5)
    M2  HAROLS       — HAR-OLS academic baseline (eq. 6)
    M3  SHAR         — Semi-HAR with signed semi-variances (eq. 7)
    M4  HARExtended  — HAR + VRP + macro, Ridge regularised (eqs. 8–9)
    M5  HARRegime    — M4 with regime interactions, Ridge regularised (eq. 10)

Nesting structure:
    M0 ⊂ M2 ⊂ M3  (Clark-West test)
    M0 ⊂ M2 ⊂ M4  (Clark-West test)
    M1, M3 vs M4, M4 vs M5  (Diebold-Mariano test)
"""
from .base         import BaseModel
from .random_walk  import RandomWalk
from .garch        import GARCHModel
from .har_ols      import HAROLS
from .shar         import SHAR
from .har_extended import HARExtended
from .har_regime   import HARRegime

__all__ = [
    "BaseModel",
    "RandomWalk",
    "GARCHModel",
    "HAROLS",
    "SHAR",
    "HARExtended",
    "HARRegime",
]