"""Cross-validation utilities for experiment tracks."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics used across experiment runs."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_grouped_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run grouped K-fold CV and return fold-level and aggregate metrics."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, float | int]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X=X, y=y, groups=groups),
        start=1,
    ):
        fold_model = clone(model)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        fold_model.fit(X_train, y_train)
        y_val_pred = fold_model.predict(X_val)
        fold_metrics = regression_metrics(y_true=y_val, y_pred=y_val_pred)
        fold_rows.append({"fold": fold_idx, **fold_metrics})

    fold_metrics_df = pd.DataFrame(fold_rows).sort_values("fold").reset_index(
        drop=True
    )
    aggregate_metrics = {
        "rmse_mean": float(fold_metrics_df["rmse"].mean()),
        "rmse_std": float(fold_metrics_df["rmse"].std(ddof=0)),
        "mae_mean": float(fold_metrics_df["mae"].mean()),
        "mae_std": float(fold_metrics_df["mae"].std(ddof=0)),
        "r2_mean": float(fold_metrics_df["r2"].mean()),
        "r2_std": float(fold_metrics_df["r2"].std(ddof=0)),
    }
    return fold_metrics_df, aggregate_metrics

