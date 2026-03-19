"""Optimization objective helpers."""

from __future__ import annotations

import numpy as np


def overfit_gap(rmse_train: float, rmse_val: float) -> float:
    """Compute bounded absolute overfitting gap term."""
    return max(0.0, float(rmse_val - rmse_train))


def fold_objective(
    rmse_train: float,
    rmse_val: float,
    lambda_gap: float,
) -> float:
    """Compute per-fold optimization objective value."""
    gap = overfit_gap(rmse_train=rmse_train, rmse_val=rmse_val)
    return float(rmse_val) + float(lambda_gap) * gap


def aggregate_objective(
    rmse_train_values: list[float],
    rmse_val_values: list[float],
    lambda_gap: float,
) -> dict[str, float]:
    """Aggregate fold objective components across CV folds."""
    gaps = [
        overfit_gap(rmse_train=tr, rmse_val=val)
        for tr, val in zip(rmse_train_values, rmse_val_values)
    ]
    penalties = [float(gap) for gap in gaps]

    rmse_val_mean = float(np.mean(rmse_val_values))
    objective = rmse_val_mean + float(lambda_gap) * float(np.mean(penalties))
    return {
        "rmse_train_mean": float(np.mean(rmse_train_values)),
        "rmse_train_std": float(np.std(rmse_train_values, ddof=0)),
        "rmse_val_mean": rmse_val_mean,
        "rmse_val_std": float(np.std(rmse_val_values, ddof=0)),
        "overfit_gap_mean": float(np.mean(gaps)),
        "overfit_gap_std": float(np.std(gaps, ddof=0)),
        "gap_penalty_mean": float(np.mean(penalties)),
        "gap_penalty_std": float(np.std(penalties, ddof=0)),
        "objective": objective,
    }
