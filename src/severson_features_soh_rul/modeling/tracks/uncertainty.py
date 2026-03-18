"""Uncertainty track executor."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
import pandas as pd

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
from severson_features_soh_rul.modeling.runtime_helpers import (
    ARTIFACT_SCHEMA_VERSION,
    build_feature_signature,
    build_run_purpose,
    build_split_signature,
    target_formula,
    target_unit,
    track_family,
)
from severson_features_soh_rul.modeling.schemas import validate_required_columns
from severson_features_soh_rul.modeling.uncertainty import run_repeated_seed_uncertainty


def run_uncertainty_track(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts_root: Path,
    train_cells: list[str],
    test_cells: list[str],
    feature_columns: list[str],
) -> None:
    X_train = train_df[feature_columns]
    y_train = train_df[cfg.target]
    groups_train = train_df["cell"].astype(str)
    (
        optimization_feature_columns,
        optimization_feature_family,
        optimization_scope,
    ) = resolve_optimization_features(
        cfg=cfg,
        requested_feature_columns=feature_columns,
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
        _,
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

    configured_seeds = list(cfg.uncertainty.seeds)
    if configured_seeds:
        seeds = [int(seed) for seed in configured_seeds]
    else:
        n_repeats = int(cfg.uncertainty.n_repeats)
        seeds = [int(cfg.random_seed) + idx for idx in range(n_repeats)]

    X_test = test_df[feature_columns]
    validate_required_columns(features_df=test_df, required_columns=["SOH"])
    test_metadata_df = pd.DataFrame(
        {
            "cell": test_df["cell"].astype(str),
            "cycle": test_df["cycle"],
            "y_true": test_df[cfg.target],
            "soh_true": test_df["SOH"],
        }
    )

    (
        predictions_repeated_df,
        uncertainty_by_region_df,
        uncertainty_summary,
    ) = run_repeated_seed_uncertainty(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        test_metadata_df=test_metadata_df,
        best_params=best_params,
        seeds=seeds,
        n_jobs=int(cfg.model.n_jobs),
        target=str(cfg.target),
        region_basis=str(cfg.uncertainty.region_basis),
    )

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

    save_resolved_config(cfg=cfg, run_dir=run_dir)
    register_artifact(run_dir / "config.resolved.yaml", "config", "yaml")
    save_json(best_params, run_dir / "best_params.json")
    register_artifact(run_dir / "best_params.json", "model_selection", "json")
    save_dataframe_csv(
        optimization_history_df, run_dir / "optimization.history.csv"
    )
    register_artifact(
        run_dir / "optimization.history.csv", "optimization_history", "csv"
    )
    save_dataframe_csv(predictions_repeated_df, run_dir / "predictions.csv")
    register_artifact(run_dir / "predictions.csv", "predictions", "csv")
    save_dataframe_csv(
        uncertainty_by_region_df, run_dir / "uncertainty.by_region.csv"
    )
    register_artifact(
        run_dir / "uncertainty.by_region.csv", "uncertainty", "csv"
    )
    save_json(uncertainty_summary, run_dir / "uncertainty.summary.json")
    register_artifact(
        run_dir / "uncertainty.summary.json", "uncertainty", "json"
    )

    metrics_payload = {
        "track": str(cfg.track),
        "target": str(cfg.target),
        "feature_set_id": str(cfg.features.set_id),
        "n_features": len(feature_columns),
        "n_repeats": len(seeds),
        "region_basis": str(cfg.uncertainty.region_basis),
        **uncertainty_summary,
    }
    save_json(metrics_payload, run_dir / "metrics.json")
    register_artifact(run_dir / "metrics.json", "metrics", "json")
    save_dataframe_csv(
        pd.DataFrame([metrics_payload]), run_dir / "metrics.csv"
    )
    register_artifact(run_dir / "metrics.csv", "metrics", "csv")

    metadata = collect_run_metadata(random_seed=int(cfg.random_seed))
    metadata["feature_columns"] = feature_columns
    metadata["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    metadata["track_family"] = track_family(str(cfg.track))
    metadata["requested_track"] = str(cfg.track)
    metadata["optimization_cache_key"] = optimization_cache_key
    save_json(metadata, run_dir / "metadata.json")
    register_artifact(run_dir / "metadata.json", "run_metadata", "json")

    run_summary = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "track": str(cfg.track),
        "target": str(cfg.target),
        "target_unit": target_unit(str(cfg.target)),
        "target_formula": target_formula(str(cfg.target)),
        "feature_set_id": str(cfg.features.set_id),
        "n_features": len(feature_columns),
        "n_train_rows": int(train_df.shape[0]),
        "n_test_rows": int(test_df.shape[0]),
        "n_train_cells": int(train_df["cell"].nunique()),
        "n_test_cells": int(test_df["cell"].nunique()),
        "random_seed": int(cfg.random_seed),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "optimization_cache_key": optimization_cache_key,
        "best_params": best_params,
        "uncertainty_summary": uncertainty_summary,
        "split_signature": build_split_signature(
            random_seed=int(cfg.random_seed),
            train_cells=train_cells,
            test_cells=test_cells,
        ),
        "feature_signature": build_feature_signature(feature_columns),
    }
    summary_path = run_dir / "summary.json"
    save_json(run_summary, summary_path)
    register_artifact(summary_path, "run_summary", "json")

    run_purpose = build_run_purpose(
        cfg=cfg,
        track_name=str(cfg.track),
        feature_columns=feature_columns,
        extra={
            "n_repeats": len(seeds),
            "region_basis": str(cfg.uncertainty.region_basis),
        },
    )
    purpose_path = run_dir / "purpose.json"
    save_json(run_purpose, purpose_path)
    register_artifact(purpose_path, "run_purpose", "json")

    artifacts_index_path = run_dir / "artifacts.index.json"
    register_artifact(artifacts_index_path, "artifact_index", "json")
    save_json(
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifacts": sorted(artifact_rows, key=lambda row: row["path"]),
        },
        artifacts_index_path,
    )
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
        metrics_path=run_dir / "metrics.json",
        summary_path=summary_path,
        predictions_path=run_dir / "predictions.csv",
        artifacts_index_path=artifacts_index_path,
    )
