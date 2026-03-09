"""Feature ranking utilities for feature analysis track."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

from src.experiments.models import build_extratrees


def compute_feature_rankings(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_columns: list[str],
    model_params: dict,
    n_splits: int,
    seeds: list[int],
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute aggregated permutation and intrinsic feature rankings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        permutation_summary_df, intrinsic_summary_df
    """
    gkf = GroupKFold(n_splits=n_splits)
    permutation_rows: list[dict] = []
    intrinsic_rows: list[dict] = []

    for seed in seeds:
        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X=X, y=y, groups=groups),
            start=1,
        ):
            X_train = X.iloc[train_idx][feature_columns]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx][feature_columns]
            y_val = y.iloc[val_idx]

            model = build_extratrees(
                params=model_params,
                random_seed=seed,
                n_jobs=n_jobs,
            )
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            baseline_rmse = float(root_mean_squared_error(y_val, y_val_pred))

            for feature_name, intrinsic_value in zip(
                feature_columns,
                model.feature_importances_,
            ):
                intrinsic_rows.append(
                    {
                        "seed": seed,
                        "fold": fold_idx,
                        "feature": feature_name,
                        "intrinsic_importance": float(intrinsic_value),
                    }
                )

            for feature_name in feature_columns:
                X_val_shuffled = X_val.copy()
                shuffled_values = (
                    X_val_shuffled[feature_name]
                    .sample(
                        frac=1.0,
                        random_state=seed + fold_idx,
                    )
                    .to_numpy()
                )
                X_val_shuffled.loc[:, feature_name] = shuffled_values
                y_val_shuffled_pred = model.predict(X_val_shuffled)
                shuffled_rmse = float(
                    root_mean_squared_error(y_val, y_val_shuffled_pred)
                )
                rmse_increase = shuffled_rmse - baseline_rmse
                permutation_rows.append(
                    {
                        "seed": seed,
                        "fold": fold_idx,
                        "feature": feature_name,
                        "baseline_rmse": baseline_rmse,
                        "shuffled_rmse": shuffled_rmse,
                        "rmse_increase": rmse_increase,
                    }
                )

    permutation_raw_df = pd.DataFrame(permutation_rows)
    intrinsic_raw_df = pd.DataFrame(intrinsic_rows)

    permutation_summary_df = (
        permutation_raw_df.groupby("feature", as_index=False)
        .agg(
            permutation_rmse_increase_mean=("rmse_increase", "mean"),
            permutation_rmse_increase_std=("rmse_increase", "std"),
            baseline_rmse_mean=("baseline_rmse", "mean"),
        )
        .fillna(0.0)
        .sort_values("permutation_rmse_increase_mean", ascending=False)
        .reset_index(drop=True)
    )
    permutation_summary_df["permutation_rank"] = (
        np.arange(permutation_summary_df.shape[0]) + 1
    )

    intrinsic_summary_df = (
        intrinsic_raw_df.groupby("feature", as_index=False)
        .agg(
            intrinsic_importance_mean=("intrinsic_importance", "mean"),
            intrinsic_importance_std=("intrinsic_importance", "std"),
        )
        .fillna(0.0)
        .sort_values("intrinsic_importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    intrinsic_summary_df["intrinsic_rank"] = (
        np.arange(intrinsic_summary_df.shape[0]) + 1
    )

    return permutation_summary_df, intrinsic_summary_df
