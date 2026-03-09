"""Run full-cycle MVP experiment pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pandas as pd

from src.experiments.ablations import (
    evaluate_feature_subset_cv,
    run_leave_one_out,
    run_topk_sweep,
)
from src.experiments.cv import regression_metrics
from src.experiments.dataset import (
    load_features_dataframe,
    resolve_feature_columns,
)
from src.experiments.diagnostics import build_error_cells_summary
from src.experiments.io import (
    collect_run_metadata,
    create_run_dir,
    save_dataframe_csv,
    save_dataframe_json,
    save_json,
    save_resolved_config,
)
from src.experiments.models import build_extratrees
from src.experiments.optimize import (
    OBJECTIVE_FORMULA,
    OBJECTIVE_NAME,
    optimize_extratrees_tpe,
)
from src.experiments.plotting import (
    plot_optimization_loss,
    plot_prediction_scatter,
    plot_topk_vs_relative_gap,
    plot_topk_vs_val_rmse,
)
from src.experiments.protocol_robustness import (
    build_protocol_families,
    run_protocol_family_holdout,
    summarize_protocol_robustness,
)
from src.experiments.ranking import compute_feature_rankings
from src.experiments.schemas import (
    CHARGE_FEATURE_COLUMNS,
    FULL_CYCLE_FEATURE_COLUMNS,
    SUPPORTED_TARGETS,
    TEMPERATURE_FEATURE_COLUMNS,
    validate_required_columns,
)
from src.experiments.split import apply_cell_split, create_or_load_cell_split
from src.experiments.uncertainty import run_repeated_seed_uncertainty

logger = logging.getLogger(__name__)
ARTIFACT_SCHEMA_VERSION = "1.0.0"


def _sha256_of_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _target_unit(target: str) -> str:
    units = {
        "SOH": "percent",
        "RUL": "cycles",
        "RUL_THROUGHPUT": "Ah",
    }
    return units.get(target, "unknown")


def _target_formula(target: str) -> str:
    formulas = {
        "SOH": "SOH_n = Q_n / Q_rated",
        "RUL": "RUL_n = EoL_cycle - cycle_n",
        "RUL_THROUGHPUT": (
            "RUL_THROUGHPUT_n = throughput_at_eol - throughput_cumulative_n"
        ),
    }
    return formulas.get(target, "N/A")


def _stable_json_hash(payload: dict[str, Any]) -> str:
    return _sha256_of_string(json.dumps(payload, sort_keys=True))


def _optimization_cache_dir(cfg: DictConfig, artifacts_root: Path) -> Path:
    if cfg.optimization_cache.dir is not None:
        return Path(to_absolute_path(str(cfg.optimization_cache.dir)))
    return artifacts_root / "_optimization_cache"


def _optimization_cache_key(
    cfg: DictConfig,
    feature_columns: list[str],
    train_cells: list[str],
) -> str:
    payload = {
        "target": str(cfg.target),
        "feature_view": str(cfg.features.view),
        "feature_columns": feature_columns,
        "train_cells": sorted([str(cell) for cell in train_cells]),
        "random_seed": int(cfg.random_seed),
        "cv_n_splits": int(cfg.cv.n_splits),
        "opt_n_trials": int(cfg.optimize.n_trials),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "model_n_jobs": int(cfg.model.n_jobs),
    }
    return _stable_json_hash(payload)


def _load_cached_optimization(cache_dir: Path) -> tuple | None:
    required_files = [
        "best_params.json",
        "optimization_history.csv",
        "best_fold_metrics.csv",
        "best_aggregate_metrics.json",
    ]
    if not all(
        (cache_dir / file_name).exists() for file_name in required_files
    ):
        return None

    with open(cache_dir / "best_params.json", "r") as fp:
        best_params = json.load(fp)
    optimization_history_df = pd.read_csv(
        cache_dir / "optimization_history.csv"
    )
    best_fold_metrics_df = pd.read_csv(cache_dir / "best_fold_metrics.csv")
    with open(cache_dir / "best_aggregate_metrics.json", "r") as fp:
        best_aggregate_metrics = json.load(fp)
    return (
        best_params,
        optimization_history_df,
        best_fold_metrics_df,
        best_aggregate_metrics,
    )


def _persist_cached_optimization(
    cache_dir: Path,
    best_params: dict,
    optimization_history_df: pd.DataFrame,
    best_fold_metrics_df: pd.DataFrame,
    best_aggregate_metrics: dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_json(best_params, cache_dir / "best_params.json")
    save_dataframe_csv(
        optimization_history_df,
        cache_dir / "optimization_history.csv",
    )
    save_dataframe_json(
        optimization_history_df,
        cache_dir / "optimization_history.json",
    )
    save_dataframe_csv(
        best_fold_metrics_df, cache_dir / "best_fold_metrics.csv"
    )
    save_dataframe_json(
        best_fold_metrics_df, cache_dir / "best_fold_metrics.json"
    )
    save_json(
        best_aggregate_metrics, cache_dir / "best_aggregate_metrics.json"
    )


def _get_or_run_optimization(
    cfg: DictConfig,
    artifacts_root: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    feature_columns: list[str],
    train_cells: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    cache_enabled = bool(cfg.optimization_cache.enabled)
    cache_key = _optimization_cache_key(
        cfg=cfg,
        feature_columns=feature_columns,
        train_cells=train_cells,
    )
    cache_root = _optimization_cache_dir(
        cfg=cfg, artifacts_root=artifacts_root
    )
    cache_dir = cache_root / cache_key

    if cache_enabled:
        cached = _load_cached_optimization(cache_dir=cache_dir)
        if cached is not None:
            logger.info("Loaded cached optimization from %s", cache_dir)
            return cached

    result = optimize_extratrees_tpe(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        n_splits=int(cfg.cv.n_splits),
        n_trials=int(cfg.optimize.n_trials),
        random_seed=int(cfg.random_seed),
        n_jobs=int(cfg.model.n_jobs),
    )
    if cache_enabled:
        _persist_cached_optimization(
            cache_dir=cache_dir,
            best_params=result[0],
            optimization_history_df=result[1],
            best_fold_metrics_df=result[2],
            best_aggregate_metrics=result[3],
        )
        logger.info("Persisted optimization cache at %s", cache_dir)
    return result


def _select_k_with_heuristics(topk_sweep_df: pd.DataFrame) -> int:
    """Select k using validation RMSE, then relative gap, then smallest k."""
    ordered = topk_sweep_df.sort_values(
        by=["val_rmse_mean", "relative_gap_mean", "k"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return int(ordered.iloc[0]["k"])


def _run_feature_analysis_track(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    artifacts_root: Path,
    train_cells: list[str],
    base_feature_columns: list[str],
    no_temp_feature_columns: list[str],
) -> None:
    validate_required_columns(
        features_df=train_df,
        required_columns=[cfg.target, *base_feature_columns],
    )

    X_train = train_df[base_feature_columns]
    y_train = train_df[cfg.target]
    groups_train = train_df["cell"].astype(str)

    (
        best_params,
        optimization_history_df,
        _,
        baseline_aggregate_metrics,
    ) = _get_or_run_optimization(
        cfg=cfg,
        artifacts_root=artifacts_root,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        feature_columns=base_feature_columns,
        train_cells=train_cells,
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

    requested_k_values = [int(k) for k in cfg.feature_analysis.k_values]
    k_values = [
        k for k in requested_k_values if 1 <= k <= len(ranked_features)
    ]
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
    if selected_k_cfg == "heuristics" or selected_k_cfg is None:
        selected_k = _select_k_with_heuristics(topk_sweep_df=topk_sweep_df)
        selection_mode = "heuristics"
    else:
        selected_k = int(selected_k_cfg)
        selection_mode = "manual"

    if selected_k not in k_values:
        raise ValueError(
            f"selected_k={selected_k} is invalid for available "
            f"k_values={k_values}"
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
    no_temp_metrics["feature_view"] = str(cfg.features.view)

    run_dir = create_run_dir(
        root_dir=artifacts_root,
        track=str(cfg.track),
        target=str(cfg.target),
        run_name=cfg.artifacts.run_name,
    )
    logger.info("Saving feature-analysis artifacts to %s", run_dir)

    save_resolved_config(cfg=cfg, run_dir=run_dir)
    save_json(best_params, run_dir / "best_params.json")
    save_dataframe_csv(
        optimization_history_df,
        run_dir / "optimization_history.csv",
    )
    save_dataframe_json(
        optimization_history_df,
        run_dir / "optimization_history.json",
    )
    save_dataframe_csv(
        permutation_df,
        run_dir / "feature_ranking_permutation.csv",
    )
    save_dataframe_json(
        permutation_df,
        run_dir / "feature_ranking_permutation.json",
    )
    save_dataframe_csv(
        intrinsic_df,
        run_dir / "feature_ranking_intrinsic.csv",
    )
    save_dataframe_json(
        intrinsic_df,
        run_dir / "feature_ranking_intrinsic.json",
    )
    save_dataframe_csv(topk_sweep_df, run_dir / "topk_sweep_metrics.csv")
    save_dataframe_json(topk_sweep_df, run_dir / "topk_sweep_metrics.json")
    save_dataframe_csv(loo_df, run_dir / "loo_metrics.csv")
    save_dataframe_json(loo_df, run_dir / "loo_metrics.json")
    save_json(no_temp_metrics, run_dir / "no_temp_metrics.json")
    save_json(
        {
            "selected_k": selected_k,
            "selection_mode": selection_mode,
            "loo_executed": True,
            "selected_features": selected_features,
            "objective_name": OBJECTIVE_NAME,
            "objective_formula": OBJECTIVE_FORMULA,
        },
        run_dir / "feature_analysis_summary.json",
    )

    if bool(cfg.debug_plots.enabled):
        plot_topk_vs_val_rmse(
            topk_sweep_df=topk_sweep_df,
            output_path=run_dir / "topk_vs_val_rmse.png",
        )
        plot_topk_vs_relative_gap(
            topk_sweep_df=topk_sweep_df,
            output_path=run_dir / "topk_vs_relative_gap.png",
        )

    logger.info(
        "Feature-analysis track complete. ranked_features=%d selected_k=%d",
        len(ranked_features),
        selected_k,
    )


def _run_uncertainty_track(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts_root: Path,
    train_cells: list[str],
    feature_columns: list[str],
) -> None:
    X_train = train_df[feature_columns]
    y_train = train_df[cfg.target]
    groups_train = train_df["cell"].astype(str)

    (
        best_params,
        optimization_history_df,
        _,
        _,
    ) = _get_or_run_optimization(
        cfg=cfg,
        artifacts_root=artifacts_root,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        feature_columns=feature_columns,
        train_cells=train_cells,
    )

    configured_seeds = list(cfg.uncertainty.seeds)
    if configured_seeds:
        seeds = [int(seed) for seed in configured_seeds]
    else:
        n_repeats = int(cfg.uncertainty.n_repeats)
        seeds = [int(cfg.random_seed) + idx for idx in range(n_repeats)]

    X_test = test_df[feature_columns]
    test_metadata_df = pd.DataFrame(
        {
            "cell": test_df["cell"].astype(str),
            "cycle": test_df["cycle"],
            "y_true": test_df[cfg.target],
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
        near_eol_quantile=float(cfg.uncertainty.near_eol_quantile),
        long_life_quantile=float(cfg.uncertainty.long_life_quantile),
    )

    run_dir = create_run_dir(
        root_dir=artifacts_root,
        track=str(cfg.track),
        target=str(cfg.target),
        run_name=cfg.artifacts.run_name,
    )
    logger.info("Saving uncertainty artifacts to %s", run_dir)

    save_resolved_config(cfg=cfg, run_dir=run_dir)
    save_json(best_params, run_dir / "best_params.json")
    save_dataframe_csv(
        optimization_history_df,
        run_dir / "optimization_history.csv",
    )
    save_dataframe_json(
        optimization_history_df,
        run_dir / "optimization_history.json",
    )
    save_dataframe_csv(
        predictions_repeated_df,
        run_dir / "predictions_repeated.csv",
    )
    save_dataframe_json(
        predictions_repeated_df,
        run_dir / "predictions_repeated.json",
    )
    save_dataframe_csv(
        uncertainty_by_region_df,
        run_dir / "uncertainty_by_region.csv",
    )
    save_dataframe_json(
        uncertainty_by_region_df,
        run_dir / "uncertainty_by_region.json",
    )
    save_json(uncertainty_summary, run_dir / "uncertainty_summary.json")

    metadata = collect_run_metadata(random_seed=int(cfg.random_seed))
    metadata["feature_columns"] = feature_columns
    metadata["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    save_json(metadata, run_dir / "run_metadata.json")

    run_summary = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "track": str(cfg.track),
        "target": str(cfg.target),
        "target_unit": _target_unit(str(cfg.target)),
        "target_formula": _target_formula(str(cfg.target)),
        "feature_view": str(cfg.features.view),
        "n_features": len(feature_columns),
        "n_train_rows": int(train_df.shape[0]),
        "n_test_rows": int(test_df.shape[0]),
        "n_train_cells": int(train_df["cell"].nunique()),
        "n_test_cells": int(test_df["cell"].nunique()),
        "random_seed": int(cfg.random_seed),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "best_params": best_params,
        "uncertainty_summary": uncertainty_summary,
    }
    save_json(run_summary, run_dir / "run_summary.json")

    artifacts_index = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifacts": [
            {
                "path": "resolved_config.yaml",
                "role": "config",
                "format": "yaml",
            },
            {
                "path": "best_params.json",
                "role": "model_selection",
                "format": "json",
            },
            {
                "path": "optimization_history.csv",
                "role": "optimization_history",
                "format": "csv",
            },
            {
                "path": "optimization_history.json",
                "role": "optimization_history",
                "format": "json",
            },
            {
                "path": "predictions_repeated.csv",
                "role": "uncertainty",
                "format": "csv",
            },
            {
                "path": "predictions_repeated.json",
                "role": "uncertainty",
                "format": "json",
            },
            {
                "path": "uncertainty_by_region.csv",
                "role": "uncertainty",
                "format": "csv",
            },
            {
                "path": "uncertainty_by_region.json",
                "role": "uncertainty",
                "format": "json",
            },
            {
                "path": "uncertainty_summary.json",
                "role": "uncertainty",
                "format": "json",
            },
            {
                "path": "run_metadata.json",
                "role": "run_metadata",
                "format": "json",
            },
            {
                "path": "run_summary.json",
                "role": "run_summary",
                "format": "json",
            },
        ],
    }
    save_json(artifacts_index, run_dir / "artifacts_index.json")

    logger.info(
        "Uncertainty track complete. repeats=%d target=%s",
        len(seeds),
        str(cfg.target),
    )


def _run_diagnostics_track(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts_root: Path,
    train_cells: list[str],
    feature_columns: list[str],
) -> None:
    X_train = train_df[feature_columns]
    y_train = train_df[cfg.target]
    groups_train = train_df["cell"].astype(str)

    (
        best_params,
        optimization_history_df,
        _,
        _,
    ) = _get_or_run_optimization(
        cfg=cfg,
        artifacts_root=artifacts_root,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        feature_columns=feature_columns,
        train_cells=train_cells,
    )

    model = build_extratrees(
        params=best_params,
        random_seed=int(cfg.random_seed),
        n_jobs=int(cfg.model.n_jobs),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(test_df[feature_columns])

    predictions_df = pd.DataFrame(
        {
            "cell": test_df["cell"].astype(str),
            "cycle": test_df["cycle"],
            "y_true": test_df[cfg.target],
            "y_pred": y_pred,
        }
    )
    error_cells_summary_df, diagnostics_summary = build_error_cells_summary(
        predictions_df=predictions_df,
        top_n_cells=int(cfg.diagnostics.top_n_cells),
    )

    run_dir = create_run_dir(
        root_dir=artifacts_root,
        track=str(cfg.track),
        target=str(cfg.target),
        run_name=cfg.artifacts.run_name,
    )
    logger.info("Saving diagnostics artifacts to %s", run_dir)

    save_resolved_config(cfg=cfg, run_dir=run_dir)
    save_json(best_params, run_dir / "best_params.json")
    save_dataframe_csv(
        optimization_history_df,
        run_dir / "optimization_history.csv",
    )
    save_dataframe_json(
        optimization_history_df,
        run_dir / "optimization_history.json",
    )
    save_dataframe_csv(predictions_df, run_dir / "predictions_test.csv")
    save_dataframe_json(predictions_df, run_dir / "predictions_test.json")
    save_dataframe_csv(
        error_cells_summary_df, run_dir / "error_cells_summary.csv"
    )
    save_dataframe_json(
        error_cells_summary_df, run_dir / "error_cells_summary.json"
    )
    save_json(diagnostics_summary, run_dir / "diagnostics_summary.json")

    metadata = collect_run_metadata(random_seed=int(cfg.random_seed))
    metadata["feature_columns"] = feature_columns
    metadata["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    save_json(metadata, run_dir / "run_metadata.json")

    run_summary = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "track": str(cfg.track),
        "target": str(cfg.target),
        "target_unit": _target_unit(str(cfg.target)),
        "target_formula": _target_formula(str(cfg.target)),
        "feature_view": str(cfg.features.view),
        "n_features": len(feature_columns),
        "diagnostics_summary": diagnostics_summary,
    }
    save_json(run_summary, run_dir / "run_summary.json")

    artifacts_index = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifacts": [
            {
                "path": "resolved_config.yaml",
                "role": "config",
                "format": "yaml",
            },
            {
                "path": "best_params.json",
                "role": "model_selection",
                "format": "json",
            },
            {
                "path": "optimization_history.csv",
                "role": "optimization_history",
                "format": "csv",
            },
            {
                "path": "optimization_history.json",
                "role": "optimization_history",
                "format": "json",
            },
            {
                "path": "predictions_test.csv",
                "role": "predictions",
                "format": "csv",
            },
            {
                "path": "predictions_test.json",
                "role": "predictions",
                "format": "json",
            },
            {
                "path": "error_cells_summary.csv",
                "role": "diagnostics",
                "format": "csv",
            },
            {
                "path": "error_cells_summary.json",
                "role": "diagnostics",
                "format": "json",
            },
            {
                "path": "diagnostics_summary.json",
                "role": "diagnostics",
                "format": "json",
            },
            {
                "path": "run_metadata.json",
                "role": "run_metadata",
                "format": "json",
            },
            {
                "path": "run_summary.json",
                "role": "run_summary",
                "format": "json",
            },
        ],
    }
    save_json(artifacts_index, run_dir / "artifacts_index.json")


def _run_protocol_robustness_track(
    cfg: DictConfig,
    features_df: pd.DataFrame,
    train_df: pd.DataFrame,
    artifacts_root: Path,
    train_cells: list[str],
    feature_columns: list[str],
) -> None:
    X_train = train_df[feature_columns]
    y_train = train_df[cfg.target]
    groups_train = train_df["cell"].astype(str)

    (
        best_params,
        optimization_history_df,
        _,
        _,
    ) = _get_or_run_optimization(
        cfg=cfg,
        artifacts_root=artifacts_root,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        feature_columns=feature_columns,
        train_cells=train_cells,
    )

    family_df = build_protocol_families(
        features_df=features_df,
        cells_rated_capacity=float(
            cfg.protocol_robustness.cells_rated_capacity
        ),
        n_families=int(cfg.protocol_robustness.n_families),
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
        run_name=cfg.artifacts.run_name,
    )
    logger.info("Saving protocol robustness artifacts to %s", run_dir)

    save_resolved_config(cfg=cfg, run_dir=run_dir)
    save_json(best_params, run_dir / "best_params.json")
    save_dataframe_csv(
        optimization_history_df,
        run_dir / "optimization_history.csv",
    )
    save_dataframe_json(
        optimization_history_df,
        run_dir / "optimization_history.json",
    )
    save_dataframe_csv(
        protocol_results_df, run_dir / "protocol_family_results.csv"
    )
    save_dataframe_json(
        protocol_results_df, run_dir / "protocol_family_results.json"
    )
    save_json(protocol_summary, run_dir / "protocol_robustness_summary.json")

    metadata = collect_run_metadata(random_seed=int(cfg.random_seed))
    metadata["feature_columns"] = feature_columns
    metadata["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    save_json(metadata, run_dir / "run_metadata.json")

    run_summary = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "track": str(cfg.track),
        "target": str(cfg.target),
        "target_unit": _target_unit(str(cfg.target)),
        "target_formula": _target_formula(str(cfg.target)),
        "feature_view": str(cfg.features.view),
        "n_features": len(feature_columns),
        "protocol_robustness_summary": protocol_summary,
    }
    save_json(run_summary, run_dir / "run_summary.json")

    artifacts_index = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifacts": [
            {
                "path": "resolved_config.yaml",
                "role": "config",
                "format": "yaml",
            },
            {
                "path": "best_params.json",
                "role": "model_selection",
                "format": "json",
            },
            {
                "path": "optimization_history.csv",
                "role": "optimization_history",
                "format": "csv",
            },
            {
                "path": "optimization_history.json",
                "role": "optimization_history",
                "format": "json",
            },
            {
                "path": "protocol_family_results.csv",
                "role": "robustness",
                "format": "csv",
            },
            {
                "path": "protocol_family_results.json",
                "role": "robustness",
                "format": "json",
            },
            {
                "path": "protocol_robustness_summary.json",
                "role": "robustness",
                "format": "json",
            },
            {
                "path": "run_metadata.json",
                "role": "run_metadata",
                "format": "json",
            },
            {
                "path": "run_summary.json",
                "role": "run_summary",
                "format": "json",
            },
        ],
    }
    save_json(artifacts_index, run_dir / "artifacts_index.json")


@hydra.main(
    version_base=None,
    config_path="../conf/experiments",
    config_name="base",
)
def run_experiment(cfg: DictConfig) -> None:
    """Execute MVP full-cycle experiment track end-to-end."""
    if cfg.target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported target={cfg.target}. Supported targets: "
            f"{sorted(SUPPORTED_TARGETS)}"
        )

    features_path = Path(to_absolute_path(cfg.data.features_data_path))
    split_dir = Path(to_absolute_path(cfg.data.split_dir))
    artifacts_root = Path(to_absolute_path(cfg.artifacts.root_dir))

    logger.info("Loading features from %s", features_path)
    features_df = load_features_dataframe(features_data_path=features_path)

    feature_columns = resolve_feature_columns(
        feature_view=cfg.features.view,
        custom_features=list(cfg.features.custom_list),
    )
    validate_required_columns(
        features_df=features_df,
        required_columns=[cfg.target, *feature_columns],
    )

    train_cells, test_cells = create_or_load_cell_split(
        features_df=features_df,
        split_dir=split_dir,
        train_cells_proportion=float(cfg.data.train_cells_proportion),
        random_seed=int(cfg.random_seed),
        force_recreate=bool(cfg.data.force_recreate_split),
    )
    train_df, test_df = apply_cell_split(
        features_df=features_df,
        train_cells=train_cells,
        test_cells=test_cells,
    )

    logger.info(
        "Split summary: train_cells=%d test_cells=%d"
        "train_rows=%d test_rows=%d",
        len(train_cells),
        len(test_cells),
        train_df.shape[0],
        test_df.shape[0],
    )

    if str(cfg.track) == "full_cycle_feature_analysis":
        full_columns = FULL_CYCLE_FEATURE_COLUMNS.copy()
        full_no_temp_columns = [
            col
            for col in full_columns
            if col not in TEMPERATURE_FEATURE_COLUMNS
        ]
        _run_feature_analysis_track(
            cfg=cfg,
            train_df=train_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            base_feature_columns=full_columns,
            no_temp_feature_columns=full_no_temp_columns,
        )
        return

    if str(cfg.track) == "charge_only_feature_analysis":
        charge_columns = CHARGE_FEATURE_COLUMNS.copy()
        charge_temp_columns = [
            f"charge_{col}" for col in TEMPERATURE_FEATURE_COLUMNS
        ]
        charge_no_temp_columns = [
            col for col in charge_columns if col not in charge_temp_columns
        ]
        _run_feature_analysis_track(
            cfg=cfg,
            train_df=train_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            base_feature_columns=charge_columns,
            no_temp_feature_columns=charge_no_temp_columns,
        )
        return

    if str(cfg.track) == "uncertainty":
        _run_uncertainty_track(
            cfg=cfg,
            train_df=train_df,
            test_df=test_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            feature_columns=feature_columns,
        )
        return

    if str(cfg.track) == "diagnostics":
        _run_diagnostics_track(
            cfg=cfg,
            train_df=train_df,
            test_df=test_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            feature_columns=feature_columns,
        )
        return

    if str(cfg.track) == "protocol_robustness":
        _run_protocol_robustness_track(
            cfg=cfg,
            features_df=features_df,
            train_df=train_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            feature_columns=feature_columns,
        )
        return

    X_train = train_df[feature_columns]
    y_train = train_df[cfg.target]
    groups_train = train_df["cell"].astype(str)

    logger.info("Running TPE optimization for ExtraTrees.")
    (
        best_params,
        optimization_history_df,
        fold_metrics_df,
        cv_aggregate_metrics,
    ) = _get_or_run_optimization(
        cfg=cfg,
        artifacts_root=artifacts_root,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        feature_columns=feature_columns,
        train_cells=train_cells,
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
        run_name=cfg.artifacts.run_name,
    )
    logger.info("Saving run artifacts to %s", run_dir)
    artifact_rows: list[dict[str, str]] = []

    def register_artifact(path: Path, role: str, fmt: str) -> None:
        artifact_rows.append(
            {
                "path": path.name,
                "role": role,
                "format": fmt,
            }
        )

    resolved_config_path = run_dir / "resolved_config.yaml"
    save_resolved_config(cfg=cfg, run_dir=run_dir)
    register_artifact(resolved_config_path, "config", "yaml")

    best_params_path = run_dir / "best_params.json"
    save_json(best_params, best_params_path)
    register_artifact(best_params_path, "model_selection", "json")

    metrics_cv_csv_path = run_dir / "metrics_cv.csv"
    save_dataframe_csv(fold_metrics_df, metrics_cv_csv_path)
    register_artifact(metrics_cv_csv_path, "cv_fold_metrics", "csv")

    metrics_cv_json_path = run_dir / "metrics_cv.json"
    save_dataframe_json(fold_metrics_df, metrics_cv_json_path)
    register_artifact(metrics_cv_json_path, "cv_fold_metrics", "json")

    optimization_history_csv_path = run_dir / "optimization_history.csv"
    save_dataframe_csv(
        optimization_history_df,
        optimization_history_csv_path,
    )
    register_artifact(
        optimization_history_csv_path, "optimization_history", "csv"
    )

    optimization_history_json_path = run_dir / "optimization_history.json"
    save_dataframe_json(
        optimization_history_df,
        optimization_history_json_path,
    )
    register_artifact(
        optimization_history_json_path, "optimization_history", "json"
    )

    predictions_test_csv_path = run_dir / "predictions_test.csv"
    save_dataframe_csv(test_predictions_df, predictions_test_csv_path)
    register_artifact(predictions_test_csv_path, "test_predictions", "csv")

    predictions_test_json_path = run_dir / "predictions_test.json"
    save_dataframe_json(test_predictions_df, predictions_test_json_path)
    register_artifact(predictions_test_json_path, "test_predictions", "json")

    metrics_test_payload = {
        **test_metrics,
        "target": str(cfg.target),
        "feature_view": str(cfg.features.view),
        "n_features": len(feature_columns),
        "n_train_cells": len(train_cells),
        "n_test_cells": len(test_cells),
    }
    metrics_test_path = run_dir / "metrics_test.json"
    save_json(metrics_test_payload, metrics_test_path)
    register_artifact(metrics_test_path, "test_metrics", "json")

    metadata = collect_run_metadata(random_seed=int(cfg.random_seed))
    metadata["feature_columns"] = feature_columns
    metadata["cv_metrics"] = cv_aggregate_metrics
    metadata["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    run_metadata_path = run_dir / "run_metadata.json"
    save_json(metadata, run_metadata_path)
    register_artifact(run_metadata_path, "run_metadata", "json")

    split_manifest = {
        "split_seed": int(cfg.random_seed),
        "train_cells_count": len(train_cells),
        "test_cells_count": len(test_cells),
        "train_cells": train_cells,
        "test_cells": test_cells,
        "train_cells_hash": _sha256_of_string(json.dumps(train_cells)),
        "test_cells_hash": _sha256_of_string(json.dumps(test_cells)),
    }
    split_manifest_path = run_dir / "split_manifest.json"
    save_json(split_manifest, split_manifest_path)
    register_artifact(split_manifest_path, "split_manifest", "json")

    feature_manifest = {
        "feature_view": str(cfg.features.view),
        "target": str(cfg.target),
        "selected_features_count": len(feature_columns),
        "selected_features": feature_columns,
    }
    feature_manifest_path = run_dir / "feature_manifest.json"
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
    per_cell_metrics_csv_path = run_dir / "per_cell_test_metrics.csv"
    save_dataframe_csv(per_cell_metrics_df, per_cell_metrics_csv_path)
    register_artifact(
        per_cell_metrics_csv_path, "per_cell_test_metrics", "csv"
    )
    per_cell_metrics_json_path = run_dir / "per_cell_test_metrics.json"
    save_dataframe_json(per_cell_metrics_df, per_cell_metrics_json_path)
    register_artifact(
        per_cell_metrics_json_path, "per_cell_test_metrics", "json"
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
    residual_summary_path = run_dir / "residual_summary.json"
    save_json(residual_summary, residual_summary_path)
    register_artifact(residual_summary_path, "residual_summary", "json")

    table_cv_metrics_df = pd.DataFrame(
        [
            {
                "track": str(cfg.track),
                "target": str(cfg.target),
                "target_unit": _target_unit(str(cfg.target)),
                "feature_view": str(cfg.features.view),
                "n_features": len(feature_columns),
                **cv_aggregate_metrics,
            }
        ]
    )
    table_cv_metrics_csv_path = run_dir / "table_cv_metrics.csv"
    save_dataframe_csv(table_cv_metrics_df, table_cv_metrics_csv_path)
    register_artifact(table_cv_metrics_csv_path, "paper_table", "csv")
    table_cv_metrics_json_path = run_dir / "table_cv_metrics.json"
    save_dataframe_json(table_cv_metrics_df, table_cv_metrics_json_path)
    register_artifact(table_cv_metrics_json_path, "paper_table", "json")

    table_test_metrics_df = pd.DataFrame(
        [
            {
                "track": str(cfg.track),
                "target": str(cfg.target),
                "target_unit": _target_unit(str(cfg.target)),
                "feature_view": str(cfg.features.view),
                "n_features": len(feature_columns),
                **test_metrics,
            }
        ]
    )
    table_test_metrics_csv_path = run_dir / "table_test_metrics.csv"
    save_dataframe_csv(table_test_metrics_df, table_test_metrics_csv_path)
    register_artifact(table_test_metrics_csv_path, "paper_table", "csv")
    table_test_metrics_json_path = run_dir / "table_test_metrics.json"
    save_dataframe_json(table_test_metrics_df, table_test_metrics_json_path)
    register_artifact(table_test_metrics_json_path, "paper_table", "json")

    table_main_metrics_df = pd.DataFrame(
        [
            {
                "track": str(cfg.track),
                "target": str(cfg.target),
                "target_unit": _target_unit(str(cfg.target)),
                "feature_view": str(cfg.features.view),
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
    table_main_metrics_csv_path = run_dir / "table_main_metrics.csv"
    save_dataframe_csv(table_main_metrics_df, table_main_metrics_csv_path)
    register_artifact(table_main_metrics_csv_path, "paper_table", "csv")
    table_main_metrics_json_path = run_dir / "table_main_metrics.json"
    save_dataframe_json(table_main_metrics_df, table_main_metrics_json_path)
    register_artifact(table_main_metrics_json_path, "paper_table", "json")

    run_summary = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "track": str(cfg.track),
        "target": str(cfg.target),
        "target_unit": _target_unit(str(cfg.target)),
        "target_formula": _target_formula(str(cfg.target)),
        "feature_view": str(cfg.features.view),
        "n_features": len(feature_columns),
        "n_train_rows": int(train_df.shape[0]),
        "n_test_rows": int(test_df.shape[0]),
        "n_train_cells": len(train_cells),
        "n_test_cells": len(test_cells),
        "random_seed": int(cfg.random_seed),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "best_params": best_params,
        "cv_metrics": cv_aggregate_metrics,
        "test_metrics": test_metrics,
    }
    run_summary_path = run_dir / "run_summary.json"
    save_json(run_summary, run_summary_path)
    register_artifact(run_summary_path, "run_summary", "json")

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

    artifacts_index_path = run_dir / "artifacts_index.json"
    register_artifact(artifacts_index_path, "artifact_index", "json")
    artifacts_index = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifacts": sorted(artifact_rows, key=lambda row: row["path"]),
    }
    save_json(artifacts_index, artifacts_index_path)

    logger.info(
        "Run complete. test_rmse=%.6f test_mae=%.6f test_r2=%.6f",
        test_metrics["rmse"],
        test_metrics["mae"],
        test_metrics["r2"],
    )


if __name__ == "__main__":
    run_experiment()
