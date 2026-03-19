"""MAPIE-based conformal regression helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class ConformalModelBundle:
    """Persisted conformal model payload."""

    model: Any
    alpha: float
    conformal_enabled: bool


def fit_conformal_model(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    conformal_enabled: bool,
    alpha: float,
    calibration_proportion: float,
    random_seed: int,
    sample_weight: np.ndarray | None = None,
) -> ConformalModelBundle:
    """Fit MAPIE model with group-aware calibration split."""
    if not conformal_enabled:
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        base_model.fit(X_train, y_train, **fit_kwargs)
        return ConformalModelBundle(
            model=base_model,
            alpha=float(alpha),
            conformal_enabled=False,
        )

    mapie_model = _fit_mapie_prefit(
        base_model=base_model,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        calibration_proportion=calibration_proportion,
        random_seed=random_seed,
        sample_weight=sample_weight,
    )
    return ConformalModelBundle(
        model=mapie_model,
        alpha=float(alpha),
        conformal_enabled=True,
    )


def predict_with_intervals(
    model_bundle: ConformalModelBundle,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict point values and conformal intervals from model bundle."""
    if not model_bundle.conformal_enabled:
        y_pred = model_bundle.model.predict(X)
        nan_arr = np.full(shape=y_pred.shape[0], fill_value=np.nan)
        return y_pred, nan_arr, nan_arr

    y_pred, intervals = model_bundle.model.predict(X, alpha=model_bundle.alpha)
    interval_array = np.asarray(intervals)
    if interval_array.ndim == 3:
        interval_array = interval_array[:, :, 0]
    y_pred_lo = interval_array[:, 0]
    y_pred_hi = interval_array[:, 1]
    return y_pred, y_pred_lo, y_pred_hi


def _fit_mapie_prefit(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    calibration_proportion: float,
    random_seed: int,
    sample_weight: np.ndarray | None,
) -> Any:
    """Fit MAPIE in prefit mode using a held-out calibration split."""
    try:
        from mapie.regression import MapieRegressor
    except ImportError as exc:
        raise ImportError(
            "MAPIE is required for conformal prediction. Install dependency "
            "'mapie'."
        ) from exc

    train_idx, calib_idx = _group_aware_calibration_split(
        groups=groups_train,
        calibration_proportion=calibration_proportion,
        random_seed=random_seed,
    )

    X_fit = X_train.iloc[train_idx]
    y_fit = y_train.iloc[train_idx]
    X_calib = X_train.iloc[calib_idx]
    y_calib = y_train.iloc[calib_idx]

    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight[train_idx]
    base_model.fit(X_fit, y_fit, **fit_kwargs)

    mapie_model = MapieRegressor(estimator=base_model, cv="prefit")
    calib_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        calib_kwargs["sample_weight"] = sample_weight[calib_idx]

    try:
        mapie_model.fit(X_calib, y_calib, **calib_kwargs)
    except TypeError:
        mapie_model.fit(X_calib, y_calib)

    return mapie_model


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
