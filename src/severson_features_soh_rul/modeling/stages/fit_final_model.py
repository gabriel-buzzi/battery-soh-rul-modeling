"""Final model fitting stage."""

from __future__ import annotations

import json
import logging
from typing import Any

from severson_features_soh_rul.modeling.artifacts.resolver import (
    resolve_required_file,
    resolve_unique_stage_dir,
)
from severson_features_soh_rul.modeling.artifacts.writer import (
    prepare_stage_dir,
    write_joblib_atomic,
    write_json_atomic,
    write_resolved_config,
    write_run_info,
)
from severson_features_soh_rul.modeling.core.conformal import (
    fit_conformal_model,
)
from severson_features_soh_rul.modeling.core.models import build_model
from severson_features_soh_rul.modeling.core.weighting import (
    build_sample_weights,
)
from severson_features_soh_rul.modeling.stages.common import (
    prepare_runtime_context,
)

LOGGER = logging.getLogger(__name__)


def run_stage(cfg: Any) -> dict[str, Any]:
    """Execute fit_final_model stage."""
    LOGGER.info("[fit_final_model] running")
    selected_k: int | None = None
    topk_stage_dir: str | None = None

    context = prepare_runtime_context(
        cfg=cfg,
        stage="fit_final_model",
        k_selected=selected_k,
    )
    selected_features = context.feature_cfg.columns
    stage_dir, skipped = prepare_stage_dir(
        root_dir=context.artifacts_cfg.root_dir,
        run_key=context.run_key,
        stage="fit_final_model",
        required_files=[
            "model.best.joblib",
            "selected_features.json",
            "config.resolved.yaml",
            "run_info.json",
        ],
        overwrite=context.artifacts_cfg.overwrite,
    )
    if skipped:
        return {
            "stage": "fit_final_model",
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
    best_params = json.loads(
        resolve_required_file(
            stage_dir=optimize_stage_dir,
            file_name="best_params.json",
            stage="optimize",
        ).read_text()
    )

    X_train = context.train_df[selected_features]
    y_train = context.train_df[context.target]
    groups_train = context.train_df["cell"].astype(str)
    sample_weights = build_sample_weights(
        y_train=y_train,
        weighting_cfg=context.weighting_cfg,
        reference_series=context.train_df["RUL"],
    )

    base_model = build_model(
        model_params=best_params,
        random_seed=context.model_cfg.random_seed,
        n_jobs=context.model_cfg.n_jobs,
    )
    model_bundle = fit_conformal_model(
        base_model=base_model,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        conformal_enabled=context.conformal_cfg.enabled,
        confidence_level=context.conformal_cfg.confidence_level,
        calibration_proportion=context.conformal_cfg.calibration_proportion,
        random_seed=context.model_cfg.random_seed,
        sample_weight=sample_weights,
    )

    write_resolved_config(cfg=context.cfg, stage_dir=stage_dir)
    write_run_info(
        stage_dir=stage_dir,
        run_key=context.run_key,
        context={
            **context.stage_context,
            "run_key_components": context.run_key_components,
            "optimize_stage_dir": str(optimize_stage_dir),
            "topk_stage_dir": topk_stage_dir,
        },
    )
    write_joblib_atomic(
        output_path=stage_dir / "model.best.joblib", payload=model_bundle
    )
    write_json_atomic(
        output_path=stage_dir / "selected_features.json",
        payload={
            "selected_features": selected_features,
            "selection_mode": "base",
            "selected_k": selected_k,
        },
    )

    return {
        "stage": "fit_final_model",
        "status": "ok",
        "stage_dir": str(stage_dir),
        "run_key": context.run_key,
        "selected_features": selected_features,
    }
