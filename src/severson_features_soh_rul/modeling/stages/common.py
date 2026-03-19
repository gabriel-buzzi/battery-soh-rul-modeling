"""Shared stage-runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pandas as pd

from severson_features_soh_rul.modeling.artifacts.run_key import (
    build_run_key_components,
    serialize_run_key,
)
from severson_features_soh_rul.modeling.config.schema import (
    ArtifactsConfig,
    ConformalConfig,
    FeatureConfig,
    ModelConfig,
    OptimizeConfig,
    RankingConfig,
    RobustnessConfig,
    SplitConfig,
    TopKConfig,
    WeightingConfig,
    parse_artifacts_config,
    parse_conformal_config,
    parse_feature_config,
    parse_model_config,
    parse_optimize_config,
    parse_ranking_config,
    parse_robustness_config,
    parse_split_config,
    parse_topk_config,
    parse_weighting_config,
    stage_context_dict,
    validate_target,
)
from severson_features_soh_rul.modeling.data.features import (
    build_feature_hash,
    feature_set_id_from_config,
    load_features_dataframe,
    validate_required_columns,
)
from severson_features_soh_rul.modeling.data.split import (
    apply_cell_split,
    create_or_load_cell_split,
)


@dataclass(frozen=True)
class RuntimeContext:
    """Precomputed runtime context for stage execution."""

    cfg: DictConfig
    target: str
    feature_cfg: FeatureConfig
    split_cfg: SplitConfig
    model_cfg: ModelConfig
    optimize_cfg: OptimizeConfig
    conformal_cfg: ConformalConfig
    ranking_cfg: RankingConfig
    topk_cfg: TopKConfig
    robustness_cfg: RobustnessConfig
    weighting_cfg: WeightingConfig
    artifacts_cfg: ArtifactsConfig
    features_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_cells: list[str]
    test_cells: list[str]
    feature_hash: str
    feature_set_id: str
    run_key_components: dict[str, Any]
    run_key: str
    stage_context: dict[str, Any]


def resolve_effective_model_n_jobs(
    stage: str,
    model_n_jobs: int,
    optimize_n_jobs: int,
) -> int:
    """Resolve effective model n_jobs for the current stage."""
    normalized_stage = str(stage).strip()
    if normalized_stage == "optimize":
        return int(model_n_jobs)
    return int(model_n_jobs) * int(optimize_n_jobs)


def prepare_runtime_context(
    cfg: DictConfig,
    stage: str,
    k_selected: int | None = None,
    require_protocol_column: bool = False,
) -> RuntimeContext:
    """Prepare shared runtime context for a stage."""
    target = validate_target(str(cfg.target))
    feature_cfg = parse_feature_config(cfg)
    split_cfg = parse_split_config(cfg)
    model_cfg = parse_model_config(cfg)
    optimize_cfg = parse_optimize_config(cfg)
    effective_model_n_jobs = resolve_effective_model_n_jobs(
        stage=stage,
        model_n_jobs=model_cfg.n_jobs,
        optimize_n_jobs=optimize_cfg.n_jobs,
    )
    conformal_cfg = parse_conformal_config(cfg)
    ranking_cfg = parse_ranking_config(cfg)
    topk_cfg = parse_topk_config(cfg)
    robustness_cfg = parse_robustness_config(cfg)
    weighting_cfg = parse_weighting_config(cfg)
    artifacts_cfg = parse_artifacts_config(cfg)

    features_path = Path(to_absolute_path(str(cfg.data.features_path)))
    split_dir = Path(to_absolute_path(str(split_cfg.split_dir)))
    artifacts_root = Path(to_absolute_path(str(artifacts_cfg.root_dir)))

    features_df = load_features_dataframe(features_path)
    validate_required_columns(
        features_df=features_df,
        target=target,
        feature_columns=feature_cfg.columns,
        protocol_column=(
            robustness_cfg.protocol_column if require_protocol_column else None
        ),
    )

    train_cells, test_cells = create_or_load_cell_split(
        features_df=features_df,
        split_dir=split_dir,
        train_cells_proportion=split_cfg.train_cells_proportion,
        split_seed=split_cfg.seed,
        force_recreate=split_cfg.force_recreate,
    )
    train_df, test_df = apply_cell_split(
        features_df=features_df,
        train_cells=train_cells,
        test_cells=test_cells,
    )

    feature_hash = build_feature_hash(
        feature_columns=feature_cfg.columns,
    )
    feature_set_id = feature_set_id_from_config(
        feature_id=feature_cfg.feature_set_id,
        feature_hash=feature_hash,
    )

    run_key_components = build_run_key_components(
        target=target,
        feature_hash=feature_hash,
        split_seed=split_cfg.seed,
        model_name=model_cfg.name,
        weighting_strategy=weighting_cfg.strategy,
        k_selected=k_selected,
    )
    run_key = serialize_run_key(
        components=run_key_components,
        run_key_fields=artifacts_cfg.run_key_fields,
    )

    stage_context = stage_context_dict(
        stage=stage,
        target=target,
        feature_set_id=feature_set_id,
        feature_hash=feature_hash,
        split_seed=split_cfg.seed,
        model_name=model_cfg.name,
        weighting_strategy=weighting_cfg.strategy,
        k_selected=k_selected,
    )

    return RuntimeContext(
        cfg=cfg,
        target=target,
        feature_cfg=feature_cfg,
        split_cfg=SplitConfig(
            train_cells_proportion=split_cfg.train_cells_proportion,
            seed=split_cfg.seed,
            split_dir=split_dir,
            force_recreate=split_cfg.force_recreate,
        ),
        model_cfg=ModelConfig(
            name=model_cfg.name,
            random_seed=model_cfg.random_seed,
            n_jobs=effective_model_n_jobs,
        ),
        optimize_cfg=optimize_cfg,
        conformal_cfg=conformal_cfg,
        ranking_cfg=ranking_cfg,
        topk_cfg=topk_cfg,
        robustness_cfg=robustness_cfg,
        weighting_cfg=weighting_cfg,
        artifacts_cfg=ArtifactsConfig(
            root_dir=artifacts_root,
            naming_key=artifacts_cfg.naming_key,
            run_key_fields=artifacts_cfg.run_key_fields,
            overwrite=artifacts_cfg.overwrite,
            require_exact_match=artifacts_cfg.require_exact_match,
        ),
        features_df=features_df,
        train_df=train_df,
        test_df=test_df,
        train_cells=train_cells,
        test_cells=test_cells,
        feature_hash=feature_hash,
        feature_set_id=feature_set_id,
        run_key_components=run_key_components,
        run_key=run_key,
        stage_context=stage_context,
    )


def build_prediction_dataframe(
    base_df: pd.DataFrame,
    y_true: pd.Series | pd.DataFrame | Any,
    y_pred: Any,
    y_pred_lo: Any,
    y_pred_hi: Any,
    target: str,
    feature_columns: list[str],
    split_seed: int,
    stage: str,
    held_out_protocol: str | None = None,
) -> pd.DataFrame:
    """Build standard prediction artifact schema."""
    features_payload = json.dumps([str(col) for col in feature_columns])
    prediction_df = pd.DataFrame(
        {
            "cell": base_df["cell"].astype(str).to_numpy(),
            "cycle": base_df["cycle"].to_numpy(),
            "target": target,
            "features_used": features_payload,
            "n_features": int(len(feature_columns)),
            "split_seed": int(split_seed),
            "y_true": pd.Series(y_true).to_numpy(),
            "y_pred": pd.Series(y_pred).to_numpy(),
            "y_pred_lo": pd.Series(y_pred_lo).to_numpy(),
            "y_pred_hi": pd.Series(y_pred_hi).to_numpy(),
            "stage": stage,
        }
    )
    if held_out_protocol is not None:
        prediction_df["held_out_protocol"] = str(held_out_protocol)
    return prediction_df
