"""Utilitary functions for modeling."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute regression metrics: MAE, RMSE, R (Pearson), and R².

    Parameters
    ----------
    y_true : np.ndarrat
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Dictionary with keys 'mae', 'rmse', 'r', 'r2'
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Pearson correlation coefficient
    r = np.corrcoef(y_true, y_pred)[0][1]

    return {"mae": mae, "rmse": rmse, "r": r, "r2": r2}
