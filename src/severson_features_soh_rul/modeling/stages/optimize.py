"""Optimization stage implementation."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from severson_features_soh_rul.modeling.artifacts.writer import (
    prepare_stage_dir,
    write_csv_atomic,
    write_json_atomic,
    write_resolved_config,
    write_run_info,
)
from severson_features_soh_rul.modeling.core.models import build_model
from severson_features_soh_rul.modeling.core.weighting import (
    build_sample_weights,
)
from severson_features_soh_rul.modeling.metrics.objectives import (
    aggregate_objective,
    overfit_gap,
)
from severson_features_soh_rul.modeling.metrics.regression import rmse
from severson_features_soh_rul.modeling.stages.common import (
    prepare_runtime_context,
)


def run_stage(cfg: Any) -> dict[str, Any]:
    """Execute optimize stage."""
    print("[optimize] running")
    context = prepare_runtime_context(cfg=cfg, stage="optimize")
    if not context.optimize_cfg.enabled:
        raise ValueError(
            "optimize stage is disabled by config (optimize.enabled=false)."
        )
    stage_dir, skipped = prepare_stage_dir(
        root_dir=context.artifacts_cfg.root_dir,
        run_key=context.run_key,
        stage="optimize",
        required_files=[
            "best_params.json",
            "cv_fold_metrics.csv",
            "cv_aggregate_metrics.json",
            "config.resolved.yaml",
            "run_info.json",
        ],
        overwrite=context.artifacts_cfg.overwrite,
    )
    if skipped:
        return {
            "stage": "optimize",
            "status": "skipped",
            "stage_dir": str(stage_dir),
            "run_key": context.run_key,
        }

    X_train = context.train_df[context.feature_cfg.columns]
    y_train = context.train_df[context.target]
    groups_train = context.train_df["cell"].astype(str)

    gkf = GroupKFold(n_splits=context.optimize_cfg.cv_folds)
    search_space = cfg.optimize.search_space
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        trial_params = _sample_trial_params(
            trial=trial, search_space=search_space
        )
        model = build_model(
            model_name=context.model_cfg.name,
            model_params=trial_params,
            random_seed=context.model_cfg.random_seed,
            n_jobs=context.model_cfg.n_jobs,
        )

        fold_rows: list[dict[str, Any]] = []
        train_rmse_values: list[float] = []
        val_rmse_values: list[float] = []

        for fold_id, (train_idx, val_idx) in enumerate(
            gkf.split(X=X_train, y=y_train, groups=groups_train),
            start=1,
        ):
            fold_model = clone(model)
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            fold_weights = build_sample_weights(
                y_train=y_tr,
                weighting_cfg=context.weighting_cfg,
                reference_series=context.train_df.iloc[train_idx]["RUL"],
            )
            fit_kwargs: dict[str, Any] = {}
            if fold_weights is not None:
                fit_kwargs["sample_weight"] = fold_weights
            fold_model.fit(X_tr, y_tr, **fit_kwargs)

            y_tr_pred = fold_model.predict(X_tr)
            y_val_pred = fold_model.predict(X_val)

            rmse_train = rmse(y_true=y_tr, y_pred=y_tr_pred)
            rmse_val = rmse(y_true=y_val, y_pred=y_val_pred)
            gap = overfit_gap(rmse_train=rmse_train, rmse_val=rmse_val)
            penalty = gap
            objective_fold = (
                rmse_val + context.optimize_cfg.lambda_gap * penalty
            )

            train_rmse_values.append(rmse_train)
            val_rmse_values.append(rmse_val)
            fold_rows.append(
                {
                    "trial": trial.number,
                    "fold": fold_id,
                    "rmse_train": rmse_train,
                    "rmse_val": rmse_val,
                    "overfit_gap": gap,
                    "gap_penalty": penalty,
                    "objective": objective_fold,
                }
            )

        aggregate = aggregate_objective(
            rmse_train_values=train_rmse_values,
            rmse_val_values=val_rmse_values,
            lambda_gap=context.optimize_cfg.lambda_gap,
        )
        trial.set_user_attr("sampled_params", trial_params)
        trial.set_user_attr("cv_fold_metrics", fold_rows)
        trial.set_user_attr("cv_aggregate_metrics", aggregate)

        trial_rows.append(
            {
                "trial": trial.number,
                **trial_params,
                **aggregate,
            }
        )
        return float(aggregate["objective"])

    sampler = optuna.samplers.TPESampler(seed=context.model_cfg.random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=context.optimize_cfg.n_trials,
        n_jobs=context.optimize_cfg.n_jobs,
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs["sampled_params"])
    best_fold_metrics_df = pd.DataFrame(
        best_trial.user_attrs["cv_fold_metrics"]
    )
    best_aggregate_metrics = dict(
        best_trial.user_attrs["cv_aggregate_metrics"]
    )

    write_resolved_config(cfg=context.cfg, stage_dir=stage_dir)
    write_run_info(
        stage_dir=stage_dir,
        run_key=context.run_key,
        context={
            **context.stage_context,
            "run_key_components": context.run_key_components,
            "cv_folds": context.optimize_cfg.cv_folds,
            "opt_n_trials": context.optimize_cfg.n_trials,
            "opt_n_jobs": context.optimize_cfg.n_jobs,
        },
    )
    write_json_atomic(
        output_path=stage_dir / "best_params.json", payload=best_params
    )
    write_csv_atomic(
        output_path=stage_dir / "cv_fold_metrics.csv", df=best_fold_metrics_df
    )
    write_json_atomic(
        output_path=stage_dir / "cv_aggregate_metrics.json",
        payload=best_aggregate_metrics,
    )

    if context.optimize_cfg.save_cv_trials:
        write_csv_atomic(
            output_path=stage_dir / "cv_trials.csv",
            df=pd.DataFrame(trial_rows)
            .sort_values("trial")
            .reset_index(drop=True),
        )

    return {
        "stage": "optimize",
        "status": "ok",
        "stage_dir": str(stage_dir),
        "run_key": context.run_key,
        "best_params": best_params,
    }


def _sample_trial_params(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Sample Optuna trial params from declarative search space."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        spec_type = str(spec["type"])
        if spec_type == "int":
            params[name] = trial.suggest_int(
                name=name,
                low=int(spec["low"]),
                high=int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
            continue
        if spec_type == "float":
            params[name] = trial.suggest_float(
                name=name,
                low=float(spec["low"]),
                high=float(spec["high"]),
                step=(
                    float(spec["step"])
                    if spec.get("step") is not None
                    else None
                ),
                log=bool(spec.get("log", False)),
            )
            continue
        if spec_type == "categorical":
            params[name] = trial.suggest_categorical(
                name=name,
                choices=list(spec["choices"]),
            )
            continue
        if spec_type == "fixed":
            params[name] = spec["value"]
            continue
        raise ValueError(
            "Unsupported optimize.search_space type='{}' "
            "for parameter '{}'".format(spec_type, name)
        )
    return params
