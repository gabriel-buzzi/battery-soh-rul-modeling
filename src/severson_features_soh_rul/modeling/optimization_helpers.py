"""Shared optimization helpers with cache and feature-family scope."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pandas as pd

from severson_features_soh_rul.modeling.io import (
    save_dataframe_csv,
    save_json,
)
from severson_features_soh_rul.modeling.optimize import (
    OBJECTIVE_FORMULA,
    OBJECTIVE_NAME,
    SEARCH_SPACE_SIGNATURE,
    optimize_extratrees_tpe,
)
from severson_features_soh_rul.modeling.runtime_helpers import sha256_of_string
from severson_features_soh_rul.modeling.schemas import (
    CHARGE_FEATURE_COLUMNS,
    FULL_CYCLE_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def _stable_json_hash(payload: dict[str, Any]) -> str:
    return sha256_of_string(json.dumps(payload, sort_keys=True))


def _optimization_cache_dir(cfg: DictConfig, artifacts_root: Path) -> Path:
    if cfg.optimization_cache.dir is not None:
        return Path(to_absolute_path(str(cfg.optimization_cache.dir)))
    return artifacts_root / "_optimization_cache"


def _optimization_cache_key(
    cfg: DictConfig,
    feature_columns: list[str],
    train_cells: list[str],
    optimization_scope: str,
    optimization_feature_family: str,
) -> str:
    payload = {
        "target": str(cfg.target),
        "optimization_scope": optimization_scope,
        "optimization_feature_family": optimization_feature_family,
        "feature_columns": feature_columns,
        "train_cells": sorted([str(cell) for cell in train_cells]),
        "random_seed": int(cfg.random_seed),
        "cv_n_splits": int(cfg.cv.n_splits),
        "opt_n_trials": int(cfg.optimize.n_trials),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "search_space_signature": SEARCH_SPACE_SIGNATURE,
        "model_n_jobs": int(cfg.model.n_jobs),
    }
    return _stable_json_hash(payload)


def build_optimization_cache_key(
    cfg: DictConfig,
    feature_columns: list[str],
    train_cells: list[str],
    optimization_scope: str,
    optimization_feature_family: str,
) -> str:
    """Build deterministic optimization cache key for lineage tracking."""
    return _optimization_cache_key(
        cfg=cfg,
        feature_columns=feature_columns,
        train_cells=train_cells,
        optimization_scope=optimization_scope,
        optimization_feature_family=optimization_feature_family,
    )


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
    optimization_metadata: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_json(best_params, cache_dir / "best_params.json")
    save_dataframe_csv(
        optimization_history_df,
        cache_dir / "optimization_history.csv",
    )
    save_dataframe_csv(
        best_fold_metrics_df, cache_dir / "best_fold_metrics.csv"
    )
    save_json(
        best_aggregate_metrics, cache_dir / "best_aggregate_metrics.json"
    )
    save_json(optimization_metadata, cache_dir / "optimization_metadata.json")


def infer_feature_family(feature_columns: list[str]) -> str:
    if all(col.startswith("charge_") for col in feature_columns):
        return "charge_only"
    if all(not col.startswith("charge_") for col in feature_columns):
        return "full_cycle"
    return "mixed"


def resolve_optimization_features(
    cfg: DictConfig,
    requested_feature_columns: list[str],
) -> tuple[list[str], str, str]:
    scope = str(cfg.optimize.scope)
    feature_family = infer_feature_family(requested_feature_columns)

    if scope == "per_feature_family":
        if feature_family == "full_cycle":
            return FULL_CYCLE_FEATURE_COLUMNS.copy(), feature_family, scope
        if feature_family == "charge_only":
            return CHARGE_FEATURE_COLUMNS.copy(), feature_family, scope
        raise ValueError(
            "Optimization scope=per_feature_family does not support mixed "
            "feature families."
        )

    raise ValueError(
        "Unsupported optimize.scope="
        f"{scope}. Supported: ['per_feature_family']."
    )


def get_or_run_optimization(
    cfg: DictConfig,
    artifacts_root: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    feature_columns: list[str],
    train_cells: list[str],
    optimization_scope: str,
    optimization_feature_family: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    cache_enabled = bool(cfg.optimization_cache.enabled)
    cache_key = _optimization_cache_key(
        cfg=cfg,
        feature_columns=feature_columns,
        train_cells=train_cells,
        optimization_scope=optimization_scope,
        optimization_feature_family=optimization_feature_family,
    )
    cache_root = _optimization_cache_dir(
        cfg=cfg, artifacts_root=artifacts_root
    )
    cache_dir = cache_root / cache_key
    optimization_metadata = {
        "target": str(cfg.target),
        "requested_feature_set_id": str(cfg.features.set_id),
        "optimization_scope": optimization_scope,
        "optimization_feature_family": optimization_feature_family,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "n_train_cells": len(train_cells),
        "random_seed": int(cfg.random_seed),
        "cv_n_splits": int(cfg.cv.n_splits),
        "opt_n_trials": int(cfg.optimize.n_trials),
        "model_n_jobs": int(cfg.model.n_jobs),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "search_space_signature": SEARCH_SPACE_SIGNATURE,
    }

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
            optimization_metadata=optimization_metadata,
        )
        logger.info("Persisted optimization cache at %s", cache_dir)
    return result
