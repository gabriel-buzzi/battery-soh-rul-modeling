"""Feature ablation utilities for feature analysis track."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

from severson_features_soh_rul.modeling.models import build_extratrees


def evaluate_feature_subset_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_columns: list[str],
    model_params: dict,
    n_splits: int,
    random_seed: int,
    n_jobs: int,
) -> dict[str, float]:
    """Evaluate one feature subset using grouped CV."""
    gkf = GroupKFold(n_splits=n_splits)
    model = build_extratrees(
        params=model_params,
        random_seed=random_seed,
        n_jobs=n_jobs,
    )

    fold_rows: list[dict[str, float | int]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X=X, y=y, groups=groups),
        start=1,
    ):
        fold_model = clone(model)
        X_train = X.iloc[train_idx][feature_columns]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx][feature_columns]
        y_val = y.iloc[val_idx]

        fold_model.fit(X_train, y_train)
        y_train_pred = fold_model.predict(X_train)
        y_val_pred = fold_model.predict(X_val)

        train_rmse = float(root_mean_squared_error(y_train, y_train_pred))
        val_rmse = float(root_mean_squared_error(y_val, y_val_pred))
        relative_gap = (
            abs(train_rmse - val_rmse) / val_rmse
            if val_rmse > 0
            else float("inf")
        )
        objective_score = val_rmse + relative_gap
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "relative_gap": relative_gap,
                "objective_score": objective_score,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return {
        "n_features": int(len(feature_columns)),
        "train_rmse_mean": float(fold_df["train_rmse"].mean()),
        "train_rmse_std": float(fold_df["train_rmse"].std(ddof=0)),
        "val_rmse_mean": float(fold_df["val_rmse"].mean()),
        "val_rmse_std": float(fold_df["val_rmse"].std(ddof=0)),
        "relative_gap_mean": float(fold_df["relative_gap"].mean()),
        "relative_gap_std": float(fold_df["relative_gap"].std(ddof=0)),
        "objective_score_mean": float(fold_df["objective_score"].mean()),
        "objective_score_std": float(fold_df["objective_score"].std(ddof=0)),
    }


def run_topk_sweep(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    ranked_features: list[str],
    k_values: list[int],
    baseline_metrics: dict[str, float],
    model_params: dict,
    n_splits: int,
    random_seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Run grouped-CV sweep for top-k ranked features."""
    rows: list[dict] = []
    total_features = len(ranked_features)
    unique_k = sorted(set([*k_values, total_features]), reverse=True)
    baseline_val_rmse = float(baseline_metrics["val_rmse_mean"])

    for k in unique_k:
        selected = ranked_features[:k]
        if k == total_features:
            metrics = {
                "n_features": int(total_features),
                "train_rmse_mean": float(baseline_metrics["train_rmse_mean"]),
                "train_rmse_std": float(baseline_metrics["train_rmse_std"]),
                "val_rmse_mean": float(baseline_metrics["val_rmse_mean"]),
                "val_rmse_std": float(baseline_metrics["val_rmse_std"]),
                "relative_gap_mean": float(
                    baseline_metrics["relative_gap_mean"]
                ),
                "relative_gap_std": float(
                    baseline_metrics["relative_gap_std"]
                ),
                "objective_score_mean": float(
                    baseline_metrics["objective_score_mean"]
                ),
                "objective_score_std": float(
                    baseline_metrics["objective_score_std"]
                ),
            }
        else:
            metrics = evaluate_feature_subset_cv(
                X=X,
                y=y,
                groups=groups,
                feature_columns=selected,
                model_params=model_params,
                n_splits=n_splits,
                random_seed=random_seed,
                n_jobs=n_jobs,
            )
        rows.append(
            {
                "k": int(k),
                "selected_features": ",".join(selected),
                "val_rmse_delta_from_baseline": float(
                    metrics["val_rmse_mean"] - baseline_val_rmse
                ),
                **metrics,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("k", ascending=False)
        .reset_index(drop=True)
    )


def run_leave_one_out(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    selected_features: list[str],
    model_params: dict,
    n_splits: int,
    random_seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Run leave-one-feature-out CV inside an already selected subset."""
    rows: list[dict] = []
    for feature_to_drop in selected_features:
        subset = [f for f in selected_features if f != feature_to_drop]
        metrics = evaluate_feature_subset_cv(
            X=X,
            y=y,
            groups=groups,
            feature_columns=subset,
            model_params=model_params,
            n_splits=n_splits,
            random_seed=random_seed,
            n_jobs=n_jobs,
        )
        rows.append(
            {
                "dropped_feature": feature_to_drop,
                "selected_features": ",".join(subset),
                **metrics,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("val_rmse_mean").reset_index(drop=True)
    )
