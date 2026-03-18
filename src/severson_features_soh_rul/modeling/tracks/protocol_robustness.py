"""Protocol-robustness track executor."""

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
from severson_features_soh_rul.modeling.protocol_robustness import (
    build_protocol_families,
    run_protocol_family_holdout,
    summarize_protocol_robustness,
)
from severson_features_soh_rul.modeling.runtime_helpers import (
    ARTIFACT_SCHEMA_VERSION,
    build_feature_signature,
    build_run_purpose,
    build_split_signature,
    target_formula,
    target_unit,
    track_family,
)


def run_protocol_robustness_track(
    cfg: DictConfig,
    features_df: pd.DataFrame,
    train_df: pd.DataFrame,
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

    family_df = build_protocol_families(
        features_df=features_df,
        cells_rated_capacity=float(
            cfg.protocol_robustness.cells_rated_capacity
        ),
        max_c_rate_bins=int(cfg.protocol_robustness.max_c_rate_bins),
        min_cells_per_family=int(cfg.protocol_robustness.min_cells_per_family),
    )
    protocol_results_df = run_protocol_family_holdout(
        features_df=features_df,
        feature_columns=feature_columns,
        target=str(cfg.target),
        best_params=best_params,
        n_jobs=int(cfg.model.n_jobs),
        family_df=family_df,
        random_seed=int(cfg.random_seed),
    )
    protocol_summary = summarize_protocol_robustness(protocol_results_df)

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
    save_dataframe_csv(
        protocol_results_df, run_dir / "robustness.by_family.csv"
    )
    register_artifact(
        run_dir / "robustness.by_family.csv", "robustness", "csv"
    )
    save_json(protocol_summary, run_dir / "robustness.summary.json")
    register_artifact(
        run_dir / "robustness.summary.json", "robustness", "json"
    )

    metrics_payload = {
        "track": str(cfg.track),
        "target": str(cfg.target),
        "feature_set_id": str(cfg.features.set_id),
        "n_features": len(feature_columns),
        **protocol_summary,
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
        "optimization_cache_key": optimization_cache_key,
        "protocol_robustness_summary": protocol_summary,
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
            "n_protocol_families": int(
                protocol_results_df["held_out_family"].nunique()
            ),
            "max_c_rate_bins": int(cfg.protocol_robustness.max_c_rate_bins),
            "min_cells_per_family": int(
                cfg.protocol_robustness.min_cells_per_family
            ),
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
        predictions_path=None,
        artifacts_index_path=artifacts_index_path,
    )
