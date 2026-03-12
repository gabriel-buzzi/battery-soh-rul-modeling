"""Final held-out evaluation track executor."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
import pandas as pd

from src.experiments.cv import regression_metrics
from src.experiments.io import (
    collect_run_metadata,
    create_run_dir,
    save_dataframe_csv,
    save_json,
    save_resolved_config,
)
from src.experiments.manifest import append_run_to_manifest
from src.experiments.models import build_extratrees
from src.experiments.optimization_helpers import (
    build_optimization_cache_key,
    get_or_run_optimization,
    resolve_optimization_features,
)
from src.experiments.optimize import OBJECTIVE_FORMULA, OBJECTIVE_NAME
from src.experiments.plotting import (
    plot_optimization_loss,
    plot_prediction_scatter,
)
from src.experiments.runtime_helpers import (
    ARTIFACT_SCHEMA_VERSION,
    build_feature_signature,
    build_run_purpose,
    build_split_signature,
    target_formula,
    target_unit,
    track_family,
)


def run_final_eval_track(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts_root: Path,
    train_cells: list[str],
    test_cells: list[str],
    feature_columns: list[str],
    track_name: str,
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
        fold_metrics_df,
        cv_aggregate_metrics,
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

    best_model = build_extratrees(
        params=best_params,
        random_seed=int(cfg.random_seed),
        n_jobs=int(cfg.model.n_jobs),
    )
    best_model.fit(X_train, y_train)
    X_test = test_df[feature_columns]
    y_test = test_df[cfg.target]
    y_pred_test = best_model.predict(X_test)

    test_predictions_df = pd.DataFrame(
        {
            "cell": test_df["cell"].astype(str),
            "cycle": test_df["cycle"],
            "y_true": y_test,
            "y_pred": y_pred_test,
        }
    )
    test_metrics = regression_metrics(y_true=y_test, y_pred=y_pred_test)

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

    metrics_cv_csv_path = run_dir / "metrics_cv.csv"
    save_dataframe_csv(fold_metrics_df, metrics_cv_csv_path)
    register_artifact(metrics_cv_csv_path, "cv_fold_metrics", "csv")

    optimization_history_csv_path = run_dir / "optimization.history.csv"
    save_dataframe_csv(optimization_history_df, optimization_history_csv_path)
    register_artifact(
        optimization_history_csv_path, "optimization_history", "csv"
    )

    predictions_csv_path = run_dir / "predictions.csv"
    save_dataframe_csv(test_predictions_df, predictions_csv_path)
    register_artifact(predictions_csv_path, "predictions", "csv")

    metrics_test_payload = {
        **test_metrics,
        **cv_aggregate_metrics,
        "target": str(cfg.target),
        "feature_set_id": str(cfg.features.set_id),
        "n_features": len(feature_columns),
        "n_train_cells": len(train_cells),
        "n_test_cells": len(test_cells),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "optimization_cache_key": optimization_cache_key,
    }
    metrics_path = run_dir / "metrics.json"
    save_json(metrics_test_payload, metrics_path)
    register_artifact(metrics_path, "metrics", "json")
    metrics_csv_path = run_dir / "metrics.csv"
    save_dataframe_csv(pd.DataFrame([metrics_test_payload]), metrics_csv_path)
    register_artifact(metrics_csv_path, "metrics", "csv")

    metadata = collect_run_metadata(random_seed=int(cfg.random_seed))
    metadata["feature_columns"] = feature_columns
    metadata["cv_metrics"] = cv_aggregate_metrics
    metadata["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    metadata["track_family"] = track_family(track_name)
    metadata["requested_track"] = track_name
    metadata["optimization_cache_key"] = optimization_cache_key
    run_metadata_path = run_dir / "metadata.json"
    save_json(metadata, run_metadata_path)
    register_artifact(run_metadata_path, "run_metadata", "json")

    split_manifest = {
        "split_seed": int(cfg.random_seed),
        "train_cells_count": len(train_cells),
        "test_cells_count": len(test_cells),
        "train_cells": train_cells,
        "test_cells": test_cells,
        **build_split_signature(
            random_seed=int(cfg.random_seed),
            train_cells=train_cells,
            test_cells=test_cells,
        ),
    }
    split_manifest_path = run_dir / "split.manifest.json"
    save_json(split_manifest, split_manifest_path)
    register_artifact(split_manifest_path, "split_manifest", "json")

    feature_manifest = {
        "feature_set_id": str(cfg.features.set_id),
        "target": str(cfg.target),
        "selected_features_count": len(feature_columns),
        "selected_features": feature_columns,
    }
    feature_manifest_path = run_dir / "feature.manifest.json"
    save_json(feature_manifest, feature_manifest_path)
    register_artifact(feature_manifest_path, "feature_manifest", "json")

    per_cell_metrics_rows = []
    for cell_id, cell_pred_df in test_predictions_df.groupby("cell"):
        cell_metrics = regression_metrics(
            y_true=cell_pred_df["y_true"],
            y_pred=cell_pred_df["y_pred"],
        )
        per_cell_metrics_rows.append(
            {
                "cell": str(cell_id),
                "n_samples": int(cell_pred_df.shape[0]),
                **cell_metrics,
            }
        )
    per_cell_metrics_df = pd.DataFrame(per_cell_metrics_rows).sort_values(
        "cell"
    )
    per_cell_metrics_csv_path = run_dir / "metrics.per_cell.csv"
    save_dataframe_csv(per_cell_metrics_df, per_cell_metrics_csv_path)
    register_artifact(
        per_cell_metrics_csv_path, "per_cell_test_metrics", "csv"
    )

    residuals = test_predictions_df["y_pred"] - test_predictions_df["y_true"]
    residual_summary = {
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std(ddof=0)),
        "residual_mae": float(residuals.abs().mean()),
        "residual_q05": float(residuals.quantile(0.05)),
        "residual_q25": float(residuals.quantile(0.25)),
        "residual_q50": float(residuals.quantile(0.50)),
        "residual_q75": float(residuals.quantile(0.75)),
        "residual_q95": float(residuals.quantile(0.95)),
    }
    residual_summary_path = run_dir / "residual.summary.json"
    save_json(residual_summary, residual_summary_path)
    register_artifact(residual_summary_path, "residual_summary", "json")

    table_main_metrics_df = pd.DataFrame(
        [
            {
                "track": str(cfg.track),
                "target": str(cfg.target),
                "target_unit": target_unit(str(cfg.target)),
                "feature_set_id": str(cfg.features.set_id),
                "n_features": len(feature_columns),
                "objective_name": OBJECTIVE_NAME,
                "objective_formula": OBJECTIVE_FORMULA,
                "objective_cv_mean": float(
                    cv_aggregate_metrics["objective_score_mean"]
                ),
                "cv_val_rmse_mean": float(
                    cv_aggregate_metrics["val_rmse_mean"]
                ),
                "cv_train_rmse_mean": float(
                    cv_aggregate_metrics["train_rmse_mean"]
                ),
                "cv_relative_gap_mean": float(
                    cv_aggregate_metrics["relative_gap_mean"]
                ),
                "test_rmse": float(test_metrics["rmse"]),
                "test_mae": float(test_metrics["mae"]),
                "test_r2": float(test_metrics["r2"]),
            }
        ]
    )
    table_main_metrics_csv_path = run_dir / "table.main_metrics.csv"
    save_dataframe_csv(table_main_metrics_df, table_main_metrics_csv_path)
    register_artifact(table_main_metrics_csv_path, "paper_table", "csv")

    run_summary = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "track": track_name,
        "target": str(cfg.target),
        "target_unit": target_unit(str(cfg.target)),
        "target_formula": target_formula(str(cfg.target)),
        "feature_set_id": str(cfg.features.set_id),
        "n_features": len(feature_columns),
        "n_train_rows": int(train_df.shape[0]),
        "n_test_rows": int(test_df.shape[0]),
        "n_train_cells": len(train_cells),
        "n_test_cells": len(test_cells),
        "random_seed": int(cfg.random_seed),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "best_params": best_params,
        "optimization_cache_key": optimization_cache_key,
        "cv_metrics": cv_aggregate_metrics,
        "test_metrics": test_metrics,
        "split_signature": build_split_signature(
            random_seed=int(cfg.random_seed),
            train_cells=train_cells,
            test_cells=test_cells,
        ),
        "feature_signature": build_feature_signature(feature_columns),
    }
    run_summary_path = run_dir / "summary.json"
    save_json(run_summary, run_summary_path)
    register_artifact(run_summary_path, "run_summary", "json")

    run_purpose = build_run_purpose(
        cfg=cfg,
        track_name=track_name,
        feature_columns=feature_columns,
        extra={
            "n_train_cells": len(train_cells),
            "n_test_cells": len(test_cells),
            "objective_name": OBJECTIVE_NAME,
        },
    )
    run_purpose_path = run_dir / "purpose.json"
    save_json(run_purpose, run_purpose_path)
    register_artifact(run_purpose_path, "run_purpose", "json")

    if bool(cfg.debug_plots.enabled):
        debug_loss_path = run_dir / "debug_optimization_loss.png"
        plot_optimization_loss(
            optimization_history_df=optimization_history_df,
            output_path=debug_loss_path,
        )
        register_artifact(debug_loss_path, "debug_plot", "png")
        debug_scatter_path = run_dir / "debug_test_scatter.png"
        plot_prediction_scatter(
            predictions_df=test_predictions_df,
            output_path=debug_scatter_path,
        )
        register_artifact(debug_scatter_path, "debug_plot", "png")

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
        track=track_name,
        target=str(cfg.target),
        feature_set_id=str(cfg.features.set_id),
        optimization_cache_key=optimization_cache_key,
        split_signature=run_summary["split_signature"],
        feature_signature=run_summary["feature_signature"],
        purpose=run_purpose,
        metrics_path=metrics_path,
        summary_path=run_summary_path,
        predictions_path=predictions_csv_path,
        artifacts_index_path=artifacts_index_path,
    )
