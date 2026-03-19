"""Regression metric helpers for modeling stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def rmse(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> float:
    """Compute root-mean-squared error."""
    return float(root_mean_squared_error(y_true, y_pred))


def mae(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> float:
    """Compute mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> float:
    """Compute coefficient of determination."""
    return float(r2_score(y_true, y_pred))


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute standard regression metrics."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
