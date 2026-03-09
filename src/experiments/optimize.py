"""Hyperparameter optimization for experiment tracks."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

from src.experiments.models import build_extratrees

OBJECTIVE_NAME = "val_rmse_plus_relative_gap"
OBJECTIVE_FORMULA = (
    "objective = RMSE_val + abs(RMSE_train - RMSE_val) / RMSE_val"
)


def optimize_extratrees_tpe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_splits: int,
    n_trials: int,
    random_seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run TPE optimization with RMSE + relative RMSE gap objective.

    The objective mirrors the paper strategy:
    objective = RMSE_val + abs(RMSE_train - RMSE_val) / RMSE_val
    """
    gkf = GroupKFold(n_splits=n_splits)
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "criterion": "squared_error",
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None],
            ),
        }

        model = build_extratrees(
            params=trial_params,
            random_seed=random_seed,
            n_jobs=n_jobs,
        )
        fold_rows: list[dict[str, float | int]] = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X=X_train, y=y_train, groups=groups_train),
            start=1,
        ):
            fold_model = clone(model)
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_model.fit(X_tr, y_tr)
            y_tr_pred = fold_model.predict(X_tr)
            y_val_pred = fold_model.predict(X_val)

            train_rmse = float(root_mean_squared_error(y_tr, y_tr_pred))
            val_rmse = float(root_mean_squared_error(y_val, y_val_pred))
            relative_gap = (
                abs(train_rmse - val_rmse) / val_rmse
                if val_rmse > 0.0
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

        fold_metrics_df = pd.DataFrame(fold_rows).sort_values("fold")
        agg = {
            "train_rmse_mean": float(fold_metrics_df["train_rmse"].mean()),
            "train_rmse_std": float(fold_metrics_df["train_rmse"].std(ddof=0)),
            "val_rmse_mean": float(fold_metrics_df["val_rmse"].mean()),
            "val_rmse_std": float(fold_metrics_df["val_rmse"].std(ddof=0)),
            "relative_gap_mean": float(fold_metrics_df["relative_gap"].mean()),
            "relative_gap_std": float(
                fold_metrics_df["relative_gap"].std(ddof=0)
            ),
            "objective_score_mean": float(
                fold_metrics_df["objective_score"].mean()
            ),
            "objective_score_std": float(
                fold_metrics_df["objective_score"].std(ddof=0)
            ),
        }

        trial.set_user_attr(
            "cv_fold_metrics",
            fold_metrics_df.to_dict(orient="records"),
        )
        trial.set_user_attr("cv_aggregate_metrics", agg)

        trial_rows.append(
            {
                "trial": trial.number,
                "objective_score": agg["objective_score_mean"],
                "val_rmse": agg["val_rmse_mean"],
                "train_rmse": agg["train_rmse_mean"],
                "relative_gap": agg["relative_gap_mean"],
                **trial_params,
            }
        )
        return float(agg["objective_score_mean"])

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    history_df = (
        pd.DataFrame(trial_rows).sort_values("trial").reset_index(drop=True)
    )
    best_trial = study.best_trial
    best_fold_metrics_df = pd.DataFrame(
        best_trial.user_attrs["cv_fold_metrics"]
    )
    best_aggregate_metrics = dict(
        best_trial.user_attrs["cv_aggregate_metrics"]
    )
    return (
        study.best_params,
        history_df,
        best_fold_metrics_df,
        best_aggregate_metrics,
    )
