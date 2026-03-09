"""Protocol-family robustness utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.experiments.models import build_extratrees


def build_protocol_families(
    features_df: pd.DataFrame,
    cells_rated_capacity: float,
    n_families: int,
) -> pd.DataFrame:
    """Assign protocol families from per-cell average charging C-rate."""
    required_cols = {"cell", "charge_I_mean"}
    if not required_cols.issubset(features_df.columns):
        missing = sorted(list(required_cols - set(features_df.columns)))
        raise ValueError(
            "Missing columns for protocol robustness family assignment: "
            f"{missing}. Ensure charge-only features were extracted."
        )

    cell_stats_df = (
        features_df.groupby("cell", as_index=False)
        .agg(avg_charge_current=("charge_I_mean", "mean"))
        .copy()
    )
    cell_stats_df["avg_charge_c_rate"] = cell_stats_df[
        "avg_charge_current"
    ].abs() / float(cells_rated_capacity)
    # Use quantile bins with duplicate-edge handling.
    cell_stats_df["protocol_family"] = pd.qcut(
        cell_stats_df["avg_charge_c_rate"],
        q=n_families,
        duplicates="drop",
    ).astype(str)
    return cell_stats_df


def run_protocol_family_holdout(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    best_params: dict,
    n_jobs: int,
    family_df: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Hold out one protocol family at a time and evaluate performance."""
    merged_df = features_df.merge(
        family_df[["cell", "protocol_family", "avg_charge_c_rate"]],
        on="cell",
        how="inner",
    )

    rows: list[dict] = []
    for family in sorted(merged_df["protocol_family"].unique().tolist()):
        train_df = merged_df[merged_df["protocol_family"] != family]
        test_df = merged_df[merged_df["protocol_family"] == family]
        if train_df.empty or test_df.empty:
            continue

        model = build_extratrees(
            params=best_params,
            random_seed=random_seed,
            n_jobs=n_jobs,
        )
        model.fit(train_df[feature_columns], train_df[target])
        y_pred = model.predict(test_df[feature_columns])
        y_true = test_df[target]

        rows.append(
            {
                "held_out_family": family,
                "n_train_cells": int(train_df["cell"].nunique()),
                "n_test_cells": int(test_df["cell"].nunique()),
                "n_test_samples": int(test_df.shape[0]),
                "rmse": float(root_mean_squared_error(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
                "avg_charge_c_rate_held_out": float(
                    test_df["avg_charge_c_rate"].mean()
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("held_out_family")
        .reset_index(drop=True)
    )


def summarize_protocol_robustness(results_df: pd.DataFrame) -> dict:
    """Create compact summary payload for protocol-family robustness."""
    if results_df.empty:
        return {
            "n_families_evaluated": 0,
            "rmse_mean": 0.0,
            "rmse_std": 0.0,
            "mae_mean": 0.0,
            "mae_std": 0.0,
            "r2_mean": 0.0,
            "r2_std": 0.0,
        }
    return {
        "n_families_evaluated": int(results_df.shape[0]),
        "rmse_mean": float(results_df["rmse"].mean()),
        "rmse_std": float(results_df["rmse"].std(ddof=0)),
        "mae_mean": float(results_df["mae"].mean()),
        "mae_std": float(results_df["mae"].std(ddof=0)),
        "r2_mean": float(results_df["r2"].mean()),
        "r2_std": float(results_df["r2"].std(ddof=0)),
    }
