"""Permutation ranking stage with conformal uncertainty outputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from severson_features_soh_rul.modeling.artifacts.resolver import (
    resolve_required_file,
    resolve_unique_stage_dir,
)
from severson_features_soh_rul.modeling.artifacts.writer import (
    prepare_stage_dir,
    write_csv_atomic,
    write_resolved_config,
    write_run_info,
)
from severson_features_soh_rul.modeling.core.conformal import (
    fit_conformal_model,
    predict_with_intervals,
)
from severson_features_soh_rul.modeling.core.models import build_model
from severson_features_soh_rul.modeling.core.weighting import (
    build_sample_weights,
)
from severson_features_soh_rul.modeling.metrics.regression import rmse
from severson_features_soh_rul.modeling.stages.common import (
    build_prediction_dataframe,
    prepare_runtime_context,
)


def run_stage(cfg: Any) -> dict[str, Any]:
    """Execute rank stage."""
    context = prepare_runtime_context(cfg=cfg, stage="rank")
    stage_dir, skipped = prepare_stage_dir(
        root_dir=context.artifacts_cfg.root_dir,
        run_key=context.run_key,
        stage="rank",
        required_files=[
            "predictions_rank_val.csv",
            "ranking_permutation_rmse.csv",
            "ranking_permutation_interval_width.csv",
            "ranking_composite.csv",
            "ranking_stability.csv",
            "config.resolved.yaml",
            "run_info.json",
        ],
        overwrite=context.artifacts_cfg.overwrite,
    )
    if skipped:
        return {
            "stage": "rank",
            "status": "skipped",
            "stage_dir": str(stage_dir),
            "run_key": context.run_key,
        }

    optimize_stage_dir = resolve_unique_stage_dir(
        artifacts_root=context.artifacts_cfg.root_dir,
        stage="optimize",
        match_fields={
            "target": context.target,
            "feature_hash": context.feature_hash,
            "split_seed": context.split_cfg.seed,
            "model_name": context.model_cfg.name,
            "weighting_strategy": context.weighting_cfg.strategy,
        },
        require_exact_match=context.artifacts_cfg.require_exact_match,
    )
    best_params_path = resolve_required_file(
        stage_dir=optimize_stage_dir,
        file_name="best_params.json",
        stage="optimize",
    )
    best_params = json.loads(best_params_path.read_text())

    X_train = context.train_df[context.feature_cfg.columns]
    y_train = context.train_df[context.target]
    groups_train = context.train_df["cell"].astype(str)

    gkf = GroupKFold(n_splits=context.optimize_cfg.cv_folds)
    prediction_rows: list[pd.DataFrame] = []
    permutation_rows: list[dict[str, Any]] = []

    for repeat_idx in range(context.ranking_cfg.n_repeats):
        repeat_seed = context.model_cfg.random_seed + repeat_idx
        for fold_id, (train_idx, val_idx) in enumerate(
            gkf.split(X=X_train, y=y_train, groups=groups_train),
            start=1,
        ):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]
            groups_tr = groups_train.iloc[train_idx]

            fold_weights = build_sample_weights(
                y_train=y_tr,
                weighting_cfg=context.weighting_cfg,
                reference_series=context.train_df.iloc[train_idx]["RUL"],
            )

            base_model = build_model(
                model_name=context.model_cfg.name,
                model_params=best_params,
                random_seed=repeat_seed,
                n_jobs=context.model_cfg.n_jobs,
            )
            model_bundle = fit_conformal_model(
                base_model=base_model,
                X_train=X_tr,
                y_train=y_tr,
                groups_train=groups_tr,
                conformal_enabled=context.conformal_cfg.enabled,
                alpha=context.conformal_cfg.alpha,
                calibration_proportion=context.conformal_cfg.calibration_proportion,
                random_seed=repeat_seed,
                sample_weight=fold_weights,
            )
            y_val_pred, y_val_lo, y_val_hi = predict_with_intervals(
                model_bundle=model_bundle,
                X=X_val,
            )

            val_predictions = build_prediction_dataframe(
                base_df=context.train_df.iloc[val_idx],
                y_true=y_val,
                y_pred=y_val_pred,
                y_pred_lo=y_val_lo,
                y_pred_hi=y_val_hi,
                target=context.target,
                feature_set_id=context.feature_set_id,
                feature_hash=context.feature_hash,
                split_seed=context.split_cfg.seed,
                stage="rank_val",
            )
            val_predictions["repeat"] = repeat_idx
            val_predictions["fold"] = fold_id
            prediction_rows.append(val_predictions)

            baseline_rmse = rmse(y_true=y_val, y_pred=y_val_pred)
            baseline_width = float(np.nanmean(y_val_hi - y_val_lo))

            for feature in context.feature_cfg.columns:
                X_val_perm = X_val.copy()
                feature_seed = int(
                    hashlib.sha256(feature.encode("utf-8")).hexdigest()[:8],
                    16,
                )
                seed = repeat_seed * 1000 + fold_id * 100 + feature_seed % 97
                rng = np.random.default_rng(seed)
                X_val_perm.loc[:, feature] = rng.permutation(
                    X_val_perm[feature].to_numpy()
                )

                y_perm_pred, y_perm_lo, y_perm_hi = predict_with_intervals(
                    model_bundle=model_bundle,
                    X=X_val_perm,
                )
                shuffled_rmse = rmse(y_true=y_val, y_pred=y_perm_pred)
                shuffled_width = float(np.nanmean(y_perm_hi - y_perm_lo))

                permutation_rows.append(
                    {
                        "repeat": repeat_idx,
                        "fold": fold_id,
                        "feature": feature,
                        "baseline_rmse": baseline_rmse,
                        "shuffled_rmse": shuffled_rmse,
                        "rmse_increase": shuffled_rmse - baseline_rmse,
                        "baseline_interval_width": baseline_width,
                        "shuffled_interval_width": shuffled_width,
                        "interval_width_increase": shuffled_width
                        - baseline_width,
                    }
                )

    prediction_df = pd.concat(prediction_rows, ignore_index=True)
    permutation_df = pd.DataFrame(permutation_rows)

    permutation_rmse_df = (
        permutation_df.groupby("feature", as_index=False)
        .agg(
            impact_rmse_mean=("rmse_increase", "mean"),
            impact_rmse_std=("rmse_increase", "std"),
        )
        .fillna(0.0)
        .sort_values("impact_rmse_mean", ascending=False)
        .reset_index(drop=True)
    )

    permutation_width_df = (
        permutation_df.groupby("feature", as_index=False)
        .agg(
            impact_uncertainty_mean=("interval_width_increase", "mean"),
            impact_uncertainty_std=("interval_width_increase", "std"),
        )
        .fillna(0.0)
        .sort_values("impact_uncertainty_mean", ascending=False)
        .reset_index(drop=True)
    )

    merged = permutation_rmse_df.merge(
        permutation_width_df,
        on="feature",
        how="inner",
        validate="one_to_one",
    )
    merged["impact_rmse_norm"] = _quantile_clipped_minmax(
        values=merged["impact_rmse_mean"],
        low_q=context.ranking_cfg.clip_low_q,
        high_q=context.ranking_cfg.clip_high_q,
    )
    merged["impact_uncertainty_norm"] = _quantile_clipped_minmax(
        values=merged["impact_uncertainty_mean"],
        low_q=context.ranking_cfg.clip_low_q,
        high_q=context.ranking_cfg.clip_high_q,
    )
    merged["composite_score"] = (
        context.ranking_cfg.w_rmse * merged["impact_rmse_norm"]
        + context.ranking_cfg.w_uncertainty * merged["impact_uncertainty_norm"]
    )
    ranking_composite_df = merged.sort_values(
        "composite_score", ascending=False
    ).reset_index(drop=True)
    ranking_composite_df["composite_rank"] = (
        np.arange(ranking_composite_df.shape[0]) + 1
    )

    ranking_stability_df = (
        permutation_df.groupby("feature", as_index=False)
        .agg(
            impact_rmse_mean=("rmse_increase", "mean"),
            impact_rmse_std=("rmse_increase", "std"),
            impact_uncertainty_mean=("interval_width_increase", "mean"),
            impact_uncertainty_std=("interval_width_increase", "std"),
        )
        .fillna(0.0)
        .sort_values("impact_rmse_mean", ascending=False)
        .reset_index(drop=True)
    )

    write_resolved_config(cfg=context.cfg, stage_dir=stage_dir)
    write_run_info(
        stage_dir=stage_dir,
        run_key=context.run_key,
        context={
            **context.stage_context,
            "run_key_components": context.run_key_components,
            "n_repeats": context.ranking_cfg.n_repeats,
            "optimize_stage_dir": str(optimize_stage_dir),
        },
    )
    write_csv_atomic(
        output_path=stage_dir / "predictions_rank_val.csv", df=prediction_df
    )
    write_csv_atomic(
        output_path=stage_dir / "ranking_permutation_rmse.csv",
        df=permutation_rmse_df,
    )
    write_csv_atomic(
        output_path=stage_dir / "ranking_permutation_interval_width.csv",
        df=permutation_width_df,
    )
    write_csv_atomic(
        output_path=stage_dir / "ranking_composite.csv",
        df=ranking_composite_df,
    )
    write_csv_atomic(
        output_path=stage_dir / "ranking_stability.csv",
        df=ranking_stability_df,
    )

    return {
        "stage": "rank",
        "status": "ok",
        "stage_dir": str(stage_dir),
        "run_key": context.run_key,
    }


def _quantile_clipped_minmax(
    values: pd.Series,
    low_q: float,
    high_q: float,
) -> pd.Series:
    """Apply quantile clipping then min-max scaling to [0, 1]."""
    low = float(values.quantile(low_q))
    high = float(values.quantile(high_q))
    clipped = values.clip(lower=low, upper=high)
    denom = float(clipped.max() - clipped.min())
    if denom <= 1e-12:
        return pd.Series(np.zeros(clipped.shape[0]), index=values.index)
    return (clipped - clipped.min()) / denom
