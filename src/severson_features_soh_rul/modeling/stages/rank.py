"""Ranking stage that consumes saved permutation prediction artifacts."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

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
from severson_features_soh_rul.modeling.metrics.regression import rmse
from severson_features_soh_rul.modeling.stages.common import (
    prepare_runtime_context,
)

LOGGER = logging.getLogger(__name__)


def run_stage(cfg: Any) -> dict[str, Any]:
    """Execute rank stage from permutation prediction artifacts."""
    LOGGER.info("[rank] running")
    context = prepare_runtime_context(cfg=cfg, stage="rank")
    stage_dir, skipped = prepare_stage_dir(
        root_dir=context.artifacts_cfg.root_dir,
        run_key=context.run_key,
        stage="rank",
        required_files=[
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

    permutation_stage_dir = resolve_unique_stage_dir(
        artifacts_root=context.artifacts_cfg.root_dir,
        stage="permutation_importance",
        match_fields={
            "target": context.target,
            "feature_hash": context.feature_hash,
            "split_seed": context.split_cfg.seed,
            "model_name": context.model_cfg.name,
            "weighting_strategy": context.weighting_cfg.strategy,
        },
        require_exact_match=context.artifacts_cfg.require_exact_match,
    )
    predictions_path = resolve_required_file(
        stage_dir=permutation_stage_dir,
        file_name="predictions_permutation_importance.csv",
        stage="permutation_importance",
    )
    predictions_df = pd.read_csv(predictions_path)
    _validate_predictions_contract(predictions_df=predictions_df)

    group_cols = ["feature", "fold", "permutation"]
    metrics_rows = [
        {
            "feature": str(feature),
            "fold": int(fold),
            "permutation": int(permutation),
            "rmse_shuffled": rmse(
                y_true=chunk["y_true"], y_pred=chunk["y_pred"]
            ),
            "interval_width_shuffled": float(
                np.nanmean(
                    chunk["y_pred_hi"].to_numpy()
                    - chunk["y_pred_lo"].to_numpy()
                )
            ),
        }
        for (feature, fold, permutation), chunk in predictions_df.groupby(
            group_cols, sort=False
        )
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    fold_feature_agg_df = (
        metrics_df.groupby(["feature", "fold"], as_index=False)
        .agg(
            rmse_fold_mean=("rmse_shuffled", "mean"),
            interval_width_fold_mean=("interval_width_shuffled", "mean"),
        )
        .sort_values(["feature", "fold"])
        .reset_index(drop=True)
    )
    permutation_rmse_df = (
        fold_feature_agg_df.groupby("feature", as_index=False)
        .agg(
            impact_rmse_mean=("rmse_fold_mean", "mean"),
            impact_rmse_std=("rmse_fold_mean", "std"),
        )
        .fillna(0.0)
        .sort_values("impact_rmse_mean", ascending=False)
        .reset_index(drop=True)
    )
    permutation_width_df = (
        fold_feature_agg_df.groupby("feature", as_index=False)
        .agg(
            impact_uncertainty_mean=("interval_width_fold_mean", "mean"),
            impact_uncertainty_std=("interval_width_fold_mean", "std"),
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

    fold_permutation_dispersion_df = (
        metrics_df.groupby(["feature", "fold"], as_index=False)
        .agg(
            impact_rmse_perm_std=("rmse_shuffled", "std"),
            impact_uncertainty_perm_std=("interval_width_shuffled", "std"),
            n_permutations=("permutation", "nunique"),
        )
        .fillna(0.0)
    )
    ranking_stability_df = (
        fold_feature_agg_df.merge(
            fold_permutation_dispersion_df,
            on=["feature", "fold"],
            how="inner",
            validate="one_to_one",
        )
        .groupby("feature", as_index=False)
        .agg(
            impact_rmse_mean=("rmse_fold_mean", "mean"),
            impact_rmse_std_across_folds=("rmse_fold_mean", "std"),
            impact_rmse_perm_std_mean=("impact_rmse_perm_std", "mean"),
            impact_uncertainty_mean=("interval_width_fold_mean", "mean"),
            impact_uncertainty_std_across_folds=(
                "interval_width_fold_mean",
                "std",
            ),
            impact_uncertainty_perm_std_mean=(
                "impact_uncertainty_perm_std",
                "mean",
            ),
            n_folds=("fold", "nunique"),
            n_permutations=("n_permutations", "max"),
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
            "heuristic_rule": "weighted_quantile_clipped_minmax",
            "n_permutations": context.ranking_cfg.n_permutations,
            "permutation_importance_stage_dir": str(permutation_stage_dir),
        },
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


def _validate_predictions_contract(predictions_df: pd.DataFrame) -> None:
    """Validate required columns for permutation predictions contract."""
    required_cols = {
        "fold",
        "feature",
        "permutation",
        "y_true",
        "y_pred",
        "y_pred_lo",
        "y_pred_hi",
    }
    missing = sorted(required_cols - set(predictions_df.columns))
    if missing:
        raise ValueError(
            "Permutation predictions artifact missing required columns: "
            + ", ".join(missing)
        )
    if predictions_df.empty:
        raise ValueError(
            "Permutation predictions artifact is empty; cannot build ranking."
        )


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
