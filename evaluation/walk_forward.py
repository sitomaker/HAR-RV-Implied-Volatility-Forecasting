from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    WF_T_MIN, WF_TAU, RIDGE_LAMBDAS, INNER_CV_FOLDS,
    FEATURES_M2, FEATURES_M3, FEATURES_M4, FEATURES_M5_BASE,
    HORIZONS, DATA_PROC,
)
from models.random_walk import RandomWalk
from models.garch import GARCHModel
from models.har_ols import HAROLS
from models.shar import SHAR
from models.har_extended import HARExtended
from models.har_regime import HARRegime

from evaluation.losses import mse as mse_fn, rmse as rmse_fn, mae as mae_fn
from evaluation.metrics import (
    oos_r2_bootstrap_ci,
    hit_rate, pesaran_timmermann,
    regime_metrics,
)
from evaluation.tests import (
    diebold_mariano_hln, clark_west, mincer_zarnowitz,
    benjamini_hochberg, andrews_bandwidth,
)

_STD_FLOOR = 1e-8


def _standardise(X_train: np.ndarray,
                 X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise using training mean/std only."""
    mu = X_train.mean(axis=0)
    sig = X_train.std(axis=0, ddof=1)
    sig = np.where(sig < _STD_FLOOR, 1.0, sig)
    return (X_train - mu) / sig, (X_test - mu) / sig


def inner_cv_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambdas: list = RIDGE_LAMBDAS,
    n_inner: int = INNER_CV_FOLDS,
    horizon: int = 1,
) -> tuple[float, list[float]]:
    """Expanding-window inner CV for Ridge λ selection."""
    n = len(X_train)
    T_inner = max(100, n // (n_inner + 1))
    tau_inner = max(1, (n - T_inner) // n_inner)

    if tau_inner < 5:
        return float(lambdas[len(lambdas) // 2]), []

    best_lambda = lambdas[0]
    best_mse = np.inf
    all_cv_mses = []

    for lam in lambdas:
        fold_mses = []
        for k in range(n_inner):
            train_end = T_inner + k * tau_inner
            val_start = train_end + horizon
            val_end = min(val_start + tau_inner, n)

            if val_start >= n or val_end > n or val_start >= val_end:
                break

            X_tr_raw = X_train[:train_end]
            y_tr = y_train[:train_end]
            X_va_raw = X_train[val_start:val_end]
            y_va = y_train[val_start:val_end]

            X_tr_sc, X_va_sc = _standardise(X_tr_raw, X_va_raw)

            mdl = Ridge(alpha=lam, fit_intercept=True)
            mdl.fit(X_tr_sc, y_tr)
            pred = mdl.predict(X_va_sc)
            fold_mses.append(float(np.mean((y_va - pred) ** 2)))

        avg_mse = float(np.mean(fold_mses)) if fold_mses else np.inf
        all_cv_mses.append(avg_mse)

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_lambda = lam

    return best_lambda, all_cv_mses


def walk_forward(
    features: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_name: str,
    horizon: int,
    model_class,
    model_kwargs: dict | None = None,
    use_ridge: bool = False,
    is_regime: bool = False,
    T_min: int = WF_T_MIN,
    tau: int = WF_TAU,
) -> dict:
    """Expanding-window walk-forward evaluation."""
    if model_kwargs is None:
        model_kwargs = {}

    needed = feature_cols + [target_col, "VIX_level"]
    if is_regime:
        needed.append("regime")

    missing = [c for c in needed if c not in features.columns]
    if missing:
        raise ValueError(f"walk_forward [{model_name}]: missing columns {missing}.")

    valid = features[needed].dropna().copy()
    n = len(valid)

    predictions, actuals = [], []
    vix_levels, regimes_oos = [], []
    fold_info, lambda_trace = [], []

    n_folds = max(1, (n - T_min) // tau)
    pbar = tqdm(total=n_folds, desc=f"{model_name} h={horizon}", leave=True)
    cutoff = T_min

    while cutoff < n:
        test_end = min(cutoff + tau, n)
        test = valid.iloc[cutoff:test_end]
        if len(test) == 0:
            break

        train_end_purged = max(cutoff - horizon, 0)
        train = valid.iloc[:train_end_purged]

        if len(train) < 10:
            cutoff = test_end
            pbar.update(1)
            continue

        y_train = train[target_col].values
        y_test = test[target_col].values

        X_train_raw = train[feature_cols].values
        X_test_raw = test[feature_cols].values

        X_train_sc, X_test_sc = _standardise(X_train_raw, X_test_raw)

        if use_ridge:
            lam, _ = inner_cv_lambda(
                X_train_raw, y_train, horizon=horizon
            )
            kwargs = dict(model_kwargs, alpha=lam)
            lambda_trace.append(lam)
        else:
            kwargs = model_kwargs
            lam = None

        model = model_class(**kwargs)

        if is_regime:
            X_tr_df = pd.DataFrame(X_train_sc, columns=feature_cols, index=train.index)
            X_te_df = pd.DataFrame(X_test_sc, columns=feature_cols, index=test.index)
            model.fit(X_tr_df, y_train, train["regime"])
            preds = model.predict(X_te_df, test["regime"])
        else:
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)

        predictions.extend(preds.tolist())
        actuals.extend(y_test.tolist())
        vix_levels.extend(test["VIX_level"].values.tolist())

        if is_regime or "regime" in valid.columns:
            regimes_oos.extend(test["regime"].values.tolist())

        fold_info.append({
            "fold_start": str(test.index[0].date()),
            "fold_end": str(test.index[-1].date()),
            "train_size": train_end_purged,
            "test_size": len(test),
            "lambda_selected": lam,
        })

        cutoff = test_end
        pbar.update(1)

    pbar.close()

    result = {
        "model": model_name,
        "horizon": horizon,
        "predictions": np.array(predictions),
        "actuals": np.array(actuals),
        "errors": np.array(actuals) - np.array(predictions),
        "vix_levels": np.array(vix_levels),
        "folds": fold_info,
        "n_oos": len(predictions),
    }

    if regimes_oos:
        result["regimes"] = np.array(regimes_oos)

    if lambda_trace:
        result["lambda_median"] = float(np.median(lambda_trace))
        result["lambda_iqr"] = float(
            np.percentile(lambda_trace, 75) -
            np.percentile(lambda_trace, 25)
        )

    return result


def walk_forward_garch(
    features: pd.DataFrame,
    target_col: str,
    horizon: int,
    T_min: int = WF_T_MIN,
    tau: int = WF_TAU,
) -> dict:
    """Walk-forward for GARCH(1,1)."""
    if "VIX_level" not in features.columns:
        raise ValueError("walk_forward_garch: 'VIX_level' missing.")

    master = pd.read_parquet(DATA_PROC / "master.parquet")
    returns_full = master["SPY_ret"].dropna()

    needed = [target_col, "VIX_level"]
    valid = features[needed].dropna().copy()
    common_idx = valid.index.intersection(returns_full.index)
    valid = valid.loc[common_idx]
    ret_series = returns_full.loc[common_idx]

    n = len(valid)
    predictions, actuals, vix_oos = [], [], []
    fold_info = []

    n_folds = max(1, (n - T_min) // tau)
    pbar = tqdm(total=n_folds, desc=f"M1_GARCH h={horizon}", leave=True)
    cutoff = T_min

    while cutoff < n:
        test_end = min(cutoff + tau, n)
        test = valid.iloc[cutoff:test_end]
        if len(test) == 0:
            break

        train_end_purged = max(cutoff - horizon, 0)

        train_returns = ret_series.iloc[:train_end_purged].values
        train_y = valid[target_col].iloc[:train_end_purged].values
        purge_rets_pct = ret_series.iloc[train_end_purged:cutoff].values * 100.0
        test_returns = ret_series.iloc[cutoff:test_end].values
        vix_current = test["VIX_level"].values

        if len(train_returns) < 50:
            cutoff = test_end
            pbar.update(1)
            continue

        garch = GARCHModel(horizon=horizon)
        garch.fit(
            X=np.zeros((len(train_returns), 1)),
            y=train_y,
            returns=train_returns,
        )

        sigma2 = garch.sigma2_T_
        for r_pct in purge_rets_pct:
            eps2 = (r_pct - garch.mu_) ** 2
            sigma2 = (
                garch.omega_
                + garch.alpha_ * eps2
                + garch.beta_ * sigma2
            )
            sigma2 = max(sigma2, 1e-8)

        garch.sigma2_T_ = sigma2

        garch.set_test_info(
            vix_current=vix_current,
            returns_test=test_returns,
        )
        preds = garch.predict(np.zeros((len(test), 1)))

        predictions.extend(preds.tolist())
        actuals.extend(test[target_col].values.tolist())
        vix_oos.extend(vix_current.tolist())

        fold_info.append({
            "fold_start": str(test.index[0].date()),
            "fold_end": str(test.index[-1].date()),
            "train_size": train_end_purged,
            "test_size": len(test),
        })

        cutoff = test_end
        pbar.update(1)

    pbar.close()

    return {
        "model": "M1_GARCH",
        "horizon": horizon,
        "predictions": np.array(predictions),
        "actuals": np.array(actuals),
        "errors": np.array(actuals) - np.array(predictions),
        "vix_levels": np.array(vix_oos),
        "folds": fold_info,
        "n_oos": len(predictions),
    }
