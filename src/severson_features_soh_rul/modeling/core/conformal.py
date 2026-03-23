"""MAPIE-based quantile conformal regression helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class ConformalModelBundle:
    """Persisted conformal model payload."""

    model: Any
    confidence_level: float
    conformal_enabled: bool


def fit_conformal_model(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    conformal_enabled: bool,
    confidence_level: float,
    calibration_proportion: float,
    random_seed: int,
    sample_weight: np.ndarray | None = None,
) -> ConformalModelBundle:
    """Fit quantile-conformal model with group-aware calibration split."""
    X_train_values = _to_model_matrix(X_train)
    y_train_values = _to_model_vector(y_train)

    if not conformal_enabled:
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        base_model.fit(X_train_values, y_train_values, **fit_kwargs)
        return ConformalModelBundle(
            model=base_model,
            confidence_level=float(confidence_level),
            conformal_enabled=False,
        )

    cqr_model = _fit_mapie_quantile_prefit(
        base_model=base_model,
        X_train=X_train_values,
        y_train=y_train_values,
        groups_train=groups_train,
        confidence_level=confidence_level,
        calibration_proportion=calibration_proportion,
        random_seed=random_seed,
        sample_weight=sample_weight,
    )
    return ConformalModelBundle(
        model=cqr_model,
        confidence_level=float(confidence_level),
        conformal_enabled=True,
    )


def predict_with_intervals(
    model_bundle: ConformalModelBundle,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict point values and conformal intervals from model bundle."""
    X_values = _to_model_matrix(X)

    if not model_bundle.conformal_enabled:
        y_pred = np.asarray(model_bundle.model.predict(X_values)).reshape(-1)
        nan_arr = np.full(shape=y_pred.shape[0], fill_value=np.nan)
        return y_pred, nan_arr, nan_arr

    interval_result = _predict_quantile_interval(
        model=model_bundle.model,
        X=X_values,
    )
    y_pred, interval_array = interval_result
    y_pred = np.asarray(y_pred).reshape(-1)
    interval_array = np.asarray(interval_array)
    if interval_array.ndim == 3:
        interval_array = np.squeeze(interval_array, axis=-1)
    if (
        interval_array.ndim == 2
        and interval_array.shape[0] == 2
        and interval_array.shape[1] == y_pred.shape[0]
    ):
        interval_array = interval_array.T
    y_pred_lo = np.minimum(interval_array[:, 0], interval_array[:, 1])
    y_pred_hi = np.maximum(interval_array[:, 0], interval_array[:, 1])
    return y_pred, y_pred_lo, y_pred_hi


def _fit_mapie_quantile_prefit(
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: pd.Series,
    confidence_level: float,
    calibration_proportion: float,
    random_seed: int,
    sample_weight: np.ndarray | None,
) -> Any:
    """Fit MAPIE quantile conformal model in prefit mode."""
    train_idx, calib_idx = _group_aware_calibration_split(
        groups=groups_train,
        calibration_proportion=calibration_proportion,
        random_seed=random_seed,
    )

    X_fit = X_train[train_idx]
    y_fit = y_train[train_idx]
    X_calib = X_train[calib_idx]
    y_calib = y_train[calib_idx]

    alpha = float(1.0 - confidence_level)
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    quantiles = [lower_q, upper_q, 0.5]

    fit_kwargs: dict[str, np.ndarray] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight[train_idx]
    estimators: list[Any] = []
    for quantile in quantiles:
        estimator = clone(base_model)
        estimator.set_params(default_quantiles=float(quantile))
        estimator.fit(X_fit, y_fit, **fit_kwargs)
        estimators.append(estimator)

    return _build_mapie_quantile_model(
        estimators=estimators,
        X_calib=X_calib,
        y_calib=y_calib,
        confidence_level=confidence_level,
    )


def _build_mapie_quantile_model(
    estimators: list[Any],
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    confidence_level: float,
) -> Any:
    """Build and conformalize MAPIE v1 quantile model."""
    try:
        from mapie.regression import ConformalizedQuantileRegressor
    except ImportError as exc:
        raise ImportError(
            "MAPIE v1 is required for conformal prediction. Install "
            "dependency 'mapie>=1,<2'."
        ) from exc

    cqr_model = ConformalizedQuantileRegressor(
        estimator=estimators,
        confidence_level=float(confidence_level),
        prefit=True,
    )
    cqr_model.conformalize(X_calib, y_calib)
    return cqr_model


def _predict_quantile_interval(
    model: Any,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict quantile-conformal intervals for MAPIE v1."""
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.INFO)
        y_pred, intervals = model.predict_interval(
            X,
            symmetric_correction=True,
        )
    finally:
        logging.disable(previous_disable)
    return np.asarray(y_pred), np.asarray(intervals)


def _to_model_matrix(X: pd.DataFrame | np.ndarray | Any) -> np.ndarray:
    """Convert tabular input to a 2D numpy matrix for model IO."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _to_model_vector(y: pd.Series | np.ndarray | Any) -> np.ndarray:
    """Convert target input to a 1D numpy vector for model IO."""
    if isinstance(y, pd.Series):
        return y.to_numpy()
    return np.asarray(y).reshape(-1)


def _group_aware_calibration_split(
    groups: pd.Series,
    calibration_proportion: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build train/cal index split while respecting group boundaries."""
    clean_groups = groups.astype(str)
    unique_groups = clean_groups.nunique()

    if unique_groups >= 3:
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=float(calibration_proportion),
            random_state=int(random_seed),
        )
        split = gss.split(
            X=np.zeros(shape=(clean_groups.shape[0], 1)),
            y=np.zeros(shape=(clean_groups.shape[0],)),
            groups=clean_groups,
        )
        train_idx, calib_idx = next(split)
        return np.asarray(train_idx), np.asarray(calib_idx)

    all_indices = np.arange(clean_groups.shape[0])
    train_idx, calib_idx = train_test_split(
        all_indices,
        test_size=float(calibration_proportion),
        random_state=int(random_seed),
        shuffle=True,
    )
    return np.asarray(train_idx), np.asarray(calib_idx)
