"""Feature-analysis track executor."""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig
import pandas as pd

from severson_features_soh_rul.modeling.ablations import (
    evaluate_feature_subset_cv,
    run_leave_one_out,
    run_topk_sweep,
)
from severson_features_soh_rul.modeling.io import (
    collect_run_metadata,
    create_run_dir,
    save_dataframe_csv,
    save_json,
    save_resolved_config,
)
from severson_features_soh_rul.modeling.manifest import append_run_to_manifest
from severson_features_soh_rul.modeling.optimization_helpers import (
    build_optimization_cache_key,
    get_or_run_optimization,
    resolve_optimization_features,
)
from severson_features_soh_rul.modeling.optimize import OBJECTIVE_FORMULA, OBJECTIVE_NAME
from severson_features_soh_rul.modeling.plotting import (
    plot_topk_vs_relative_gap,
    plot_topk_vs_val_rmse,
)
from severson_features_soh_rul.modeling.ranking import compute_feature_rankings
from severson_features_soh_rul.modeling.runtime_helpers import (
    ARTIFACT_SCHEMA_VERSION,
    build_feature_signature,
    build_run_purpose,
    build_split_signature,
    select_k_with_heuristics,
    target_formula,
    target_unit,
    track_family,
)
from severson_features_soh_rul.modeling.schemas import validate_required_columns

logger = logging.getLogger(__name__)


def run_feature_analysis_track(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    artifacts_root: Path,
    train_cells: list[str],
    test_cells: list[str],
    base_feature_columns: list[str],
    no_temp_feature_columns: list[str],
) -> None:
    logger.info(
        "Starting feature-analysis track: target=%s base_features=%d no_temp_features=%d train_rows=%d train_cells=%d",
        str(cfg.target),
        len(base_feature_columns),
        len(no_temp_feature_columns),
        int(train_df.shape[0]),
        int(train_df["cell"].nunique()),
    )
    validate_required_columns(
        features_df=train_df,
        required_columns=[cfg.target, *base_feature_columns],
    )

    X_train = train_df[base_feature_columns]
    y_train = train_df[cfg.target]
    groups_train = train_df["cell"].astype(str)
    (
        optimization_feature_columns,
        optimization_feature_family,
        optimization_scope,
    ) = resolve_optimization_features(
        cfg=cfg,
        requested_feature_columns=base_feature_columns,
    )
    X_train_opt = train_df[optimization_feature_columns]
    optimization_cache_key = build_optimization_cache_key(
        cfg=cfg,
        feature_columns=optimization_feature_columns,
        train_cells=train_cells,
        optimization_scope=optimization_scope,
        optimization_feature_family=optimization_feature_family,
    )

    (
        best_params,
        optimization_history_df,
        _,
        baseline_aggregate_metrics,
    ) = get_or_run_optimization(
        cfg=cfg,
        artifacts_root=artifacts_root,
        X_train=X_train_opt,
        y_train=y_train,
        groups_train=groups_train,
        feature_columns=optimization_feature_columns,
        train_cells=train_cells,
        optimization_scope=optimization_scope,
        optimization_feature_family=optimization_feature_family,
    )
    logger.info(
        "Feature-analysis optimization ready: best_params=%d trials=%d",
        len(best_params),
        int(optimization_history_df.shape[0]),
    )

    permutation_df, intrinsic_df = compute_feature_rankings(
        X=X_train,
        y=y_train,
        groups=groups_train,
        feature_columns=base_feature_columns,
        model_params=best_params,
        n_splits=int(cfg.cv.n_splits),
        seeds=list(cfg.feature_analysis.ranking_seeds),
        n_jobs=int(cfg.model.n_jobs),
    )
    ranked_features = permutation_df["feature"].tolist()
    logger.info(
        "Feature rankings computed: ranked_features=%d seeds=%d",
        len(ranked_features),
        len(list(cfg.feature_analysis.ranking_seeds)),
    )

    requested_k_values = [int(k) for k in cfg.feature_analysis.k_values]
    k_values = [
        k for k in requested_k_values if 1 <= k <= len(ranked_features)
    ]
    logger.info(
        "Running Top-K sweep: requested_k=%s valid_k=%s",
        requested_k_values,
        k_values,
    )
    topk_sweep_df = run_topk_sweep(
        X=X_train,
        y=y_train,
        groups=groups_train,
        ranked_features=ranked_features,
        k_values=k_values,
        baseline_metrics=baseline_aggregate_metrics,
        model_params=best_params,
        n_splits=int(cfg.cv.n_splits),
        random_seed=int(cfg.random_seed),
        n_jobs=int(cfg.model.n_jobs),
    )

    selected_k_cfg = cfg.feature_analysis.selected_k
    heuristic_max_val_rmse_increase_pct = float(
        cfg.feature_analysis.max_val_rmse_increase_pct
    )
    if selected_k_cfg == "heuristics" or selected_k_cfg is None:
        selected_k = select_k_with_heuristics(
            topk_sweep_df=topk_sweep_df,
            allowed_k_values=k_values,
            max_val_rmse_increase_pct=heuristic_max_val_rmse_increase_pct,
        )
        selection_mode = "heuristics"
    else:
        selected_k = int(selected_k_cfg)
        selection_mode = "manual"

    if selected_k not in k_values:
        raise ValueError(
            f"selected_k={selected_k} is invalid for available k_values={k_values}"
        )
    logger.info(
        "Selected feature subset: mode=%s selected_k=%d",
        selection_mode,
        selected_k,
    )

    selected_features = ranked_features[:selected_k]
    loo_df = run_leave_one_out(
        X=X_train,
        y=y_train,
        groups=groups_train,
        selected_features=selected_features,
        model_params=best_params,
        n_splits=int(cfg.cv.n_splits),
        random_seed=int(cfg.random_seed),
        n_jobs=int(cfg.model.n_jobs),
    )
    logger.info(
        "Leave-one-out ablation complete: rows=%d", int(loo_df.shape[0])
    )

    no_temp_metrics = evaluate_feature_subset_cv(
        X=X_train,
        y=y_train,
        groups=groups_train,
        feature_columns=no_temp_feature_columns,
        model_params=best_params,
        n_splits=int(cfg.cv.n_splits),
        random_seed=int(cfg.random_seed),
        n_jobs=int(cfg.model.n_jobs),
    )
    no_temp_metrics["feature_set"] = "no_temperature"
    no_temp_metrics["target"] = str(cfg.target)
    no_temp_metrics["n_features"] = len(no_temp_feature_columns)
    no_temp_metrics["feature_set_id"] = str(cfg.features.set_id)

    run_dir = create_run_dir(
        root_dir=artifacts_root,
        track=str(cfg.track),
        target=str(cfg.target),
        campaign_id=cfg.artifacts.campaign_id,
        run_name=cfg.artifacts.run_name,
    )
    artifact_rows: list[dict[str, str]] = []

    def register_artifact(path: Path, role: str, fmt: str) -> None:
        artifact_rows.append({"path": path.name, "role": role, "format": fmt})

    resolved_config_path = run_dir / "config.resolved.yaml"
    save_resolved_config(cfg=cfg, run_dir=run_dir)
    register_artifact(resolved_config_path, "config", "yaml")

    best_params_path = run_dir / "best_params.json"
    save_json(best_params, best_params_path)
    register_artifact(best_params_path, "model_selection", "json")

    optimization_history_csv_path = run_dir / "optimization.history.csv"
    save_dataframe_csv(optimization_history_df, optimization_history_csv_path)
    register_artifact(
        optimization_history_csv_path, "optimization_history", "csv"
    )

    permutation_csv_path = run_dir / "ranking.permutation.csv"
    save_dataframe_csv(permutation_df, permutation_csv_path)
    register_artifact(permutation_csv_path, "feature_ranking", "csv")

    intrinsic_csv_path = run_dir / "ranking.intrinsic.csv"
    save_dataframe_csv(intrinsic_df, intrinsic_csv_path)
    register_artifact(intrinsic_csv_path, "feature_ranking", "csv")

    topk_csv_path = run_dir / "sweep.topk.csv"
    save_dataframe_csv(topk_sweep_df, topk_csv_path)
    register_artifact(topk_csv_path, "feature_analysis", "csv")

    loo_csv_path = run_dir / "ablation.loo.csv"
    save_dataframe_csv(loo_df, loo_csv_path)
    register_artifact(loo_csv_path, "feature_analysis", "csv")

    no_temp_path = run_dir / "ablation.no_temp.json"
    save_json(no_temp_metrics, no_temp_path)
    register_artifact(no_temp_path, "feature_analysis", "json")
    feature_analysis_summary_path = run_dir / "analysis.summary.json"
    save_json(
        {
            "selected_k": selected_k,
            "selection_mode": selection_mode,
            "heuristic_max_val_rmse_increase_pct": heuristic_max_val_rmse_increase_pct,
            "loo_executed": True,
            "selected_features": selected_features,
            "objective_name": OBJECTIVE_NAME,
            "objective_formula": OBJECTIVE_FORMULA,
        },
        feature_analysis_summary_path,
    )
    register_artifact(
        feature_analysis_summary_path, "feature_analysis", "json"
    )

    metadata = collect_run_metadata(random_seed=int(cfg.random_seed))
    metadata["feature_columns"] = base_feature_columns
    metadata["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    metadata["track_family"] = track_family(str(cfg.track))
    metadata["requested_track"] = str(cfg.track)
    metadata["optimization_cache_key"] = optimization_cache_key
    metadata["n_train_rows"] = int(train_df.shape[0])
    metadata["n_train_cells"] = int(train_df["cell"].nunique())
    run_metadata_path = run_dir / "metadata.json"
    save_json(metadata, run_metadata_path)
    register_artifact(run_metadata_path, "run_metadata", "json")

    run_summary = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "track": str(cfg.track),
        "target": str(cfg.target),
        "target_unit": target_unit(str(cfg.target)),
        "target_formula": target_formula(str(cfg.target)),
        "feature_set_id": str(cfg.features.set_id),
        "n_features": len(base_feature_columns),
        "n_train_rows": int(train_df.shape[0]),
        "n_train_cells": int(train_df["cell"].nunique()),
        "random_seed": int(cfg.random_seed),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "best_params": best_params,
        "selected_k": selected_k,
        "selected_features": selected_features,
        "selection_mode": selection_mode,
        "optimization_cache_key": optimization_cache_key,
        "split_signature": build_split_signature(
            random_seed=int(cfg.random_seed),
            train_cells=train_cells,
            test_cells=test_cells,
        ),
        "feature_signature": build_feature_signature(base_feature_columns),
    }
    selected_k_row = topk_sweep_df[topk_sweep_df["k"] == selected_k]
    selected_metrics = (
        selected_k_row.iloc[0].to_dict() if not selected_k_row.empty else {}
    )
    metrics_payload = {
        "track": str(cfg.track),
        "target": str(cfg.target),
        "feature_set_id": str(cfg.features.set_id),
        "n_features": len(base_feature_columns),
        "selected_k": selected_k,
        "selection_mode": selection_mode,
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "selected_val_rmse_mean": selected_metrics.get("val_rmse_mean"),
        "selected_relative_gap_mean": selected_metrics.get(
            "relative_gap_mean"
        ),
        "no_temp_val_rmse_mean": no_temp_metrics.get("val_rmse_mean"),
    }
    metrics_path = run_dir / "metrics.json"
    save_json(metrics_payload, metrics_path)
    register_artifact(metrics_path, "metrics", "json")
    metrics_csv_path = run_dir / "metrics.csv"
    save_dataframe_csv(pd.DataFrame([metrics_payload]), metrics_csv_path)
    register_artifact(metrics_csv_path, "metrics", "csv")

    run_summary_path = run_dir / "summary.json"
    save_json(run_summary, run_summary_path)
    register_artifact(run_summary_path, "run_summary", "json")

    run_purpose = build_run_purpose(
        cfg=cfg,
        track_name=str(cfg.track),
        feature_columns=base_feature_columns,
        extra={
            "selected_k": selected_k,
            "selection_mode": selection_mode,
            "selected_features": selected_features,
            "no_temperature_feature_count": len(no_temp_feature_columns),
            "is_selection_artifact": True,
        },
    )
    run_purpose_path = run_dir / "purpose.json"
    save_json(run_purpose, run_purpose_path)
    register_artifact(run_purpose_path, "run_purpose", "json")

    if bool(cfg.debug_plots.enabled):
        plot_topk_vs_val_rmse(
            topk_sweep_df=topk_sweep_df,
            output_path=run_dir / "topk_vs_val_rmse.png",
        )
        register_artifact(
            run_dir / "topk_vs_val_rmse.png", "debug_plot", "png"
        )
        plot_topk_vs_relative_gap(
            topk_sweep_df=topk_sweep_df,
            output_path=run_dir / "topk_vs_relative_gap.png",
        )
        register_artifact(
            run_dir / "topk_vs_relative_gap.png", "debug_plot", "png"
        )

    artifacts_index_path = run_dir / "artifacts.index.json"
    register_artifact(artifacts_index_path, "artifact_index", "json")
    artifacts_index = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifacts": sorted(artifact_rows, key=lambda row: row["path"]),
    }
    save_json(artifacts_index, artifacts_index_path)
    append_run_to_manifest(
        artifacts_root=artifacts_root,
        campaign_id=cfg.artifacts.campaign_id,
        run_dir=run_dir,
        track=str(cfg.track),
        target=str(cfg.target),
        feature_set_id=str(cfg.features.set_id),
        optimization_cache_key=optimization_cache_key,
        split_signature=run_summary["split_signature"],
        feature_signature=run_summary["feature_signature"],
        purpose=run_purpose,
        metrics_path=metrics_path,
        summary_path=run_summary_path,
        predictions_path=None,
        artifacts_index_path=artifacts_index_path,
    )
