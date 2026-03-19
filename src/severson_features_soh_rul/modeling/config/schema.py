"""Typed configuration helpers for the modeling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from severson_features_soh_rul.modeling.config.defaults import (
    DEFAULT_ARTIFACTS_OVERWRITE,
    DEFAULT_ARTIFACTS_REQUIRE_EXACT_MATCH,
    DEFAULT_CONFORMAL_ALPHA,
    DEFAULT_CONFORMAL_CALIBRATION_PROPORTION,
    DEFAULT_LONG_LIFE_BOOST_FACTOR,
    DEFAULT_LONG_LIFE_QUANTILE,
    DEFAULT_OBJECTIVE_LAMBDA_GAP,
    DEFAULT_OPTIMIZE_N_JOBS,
    DEFAULT_PROTOCOL_COLUMN,
    DEFAULT_RANKING_CLIP_HIGH_Q,
    DEFAULT_RANKING_CLIP_LOW_Q,
    DEFAULT_RANKING_N_PERMUTATIONS,
    DEFAULT_RANKING_WEIGHT_RMSE,
    DEFAULT_RANKING_WEIGHT_UNCERTAINTY,
    DEFAULT_SAMPLE_WEIGHT_N_BINS,
    DEFAULT_TOPK_TAU_RMSE,
    DEFAULT_TOPK_TAU_WIDTH,
    DEFAULT_WEIGHTING_STRATEGY,
)

SUPPORTED_STAGES = {
    "optimize",
    "permutation_importance",
    "rank",
    "topk_sweep",
    "fit_final_model",
    "predict",
    "robustness_protocol_lopo",
    "baseline_flow",
}
SUPPORTED_TARGETS = {"SOH", "RUL"}
SUPPORTED_WEIGHTING_STRATEGIES = {
    "none",
    "sample_weight_inverse_life_density",
    "sample_weight_long_life_boost",
}
SUPPORTED_MODEL_NAMES = {"extratrees"}


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-configuration payload."""

    columns: list[str]
    feature_set_id: str


@dataclass(frozen=True)
class SplitConfig:
    """Cell split configuration."""

    train_cells_proportion: float
    seed: int
    split_dir: Path
    force_recreate: bool


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    name: str
    random_seed: int
    n_jobs: int


@dataclass(frozen=True)
class OptimizeConfig:
    """Optimization stage configuration."""

    enabled: bool
    n_trials: int
    n_jobs: int
    cv_folds: int
    lambda_gap: float
    save_cv_trials: bool


@dataclass(frozen=True)
class ConformalConfig:
    """Conformal inference configuration."""

    enabled: bool
    alpha: float
    calibration_proportion: float


@dataclass(frozen=True)
class RankingConfig:
    """Ranking configuration."""

    n_permutations: int
    w_rmse: float
    w_uncertainty: float
    clip_low_q: float
    clip_high_q: float


@dataclass(frozen=True)
class RobustnessConfig:
    """Robustness configuration."""

    mode: str
    protocol_column: str


@dataclass(frozen=True)
class WeightingConfig:
    """Training weighting configuration."""

    enabled: bool
    strategy: str
    n_bins: int
    long_life_quantile: float
    long_life_boost_factor: float


@dataclass(frozen=True)
class ArtifactsConfig:
    """Artifact management configuration."""

    root_dir: Path
    naming_key: str
    run_key_fields: list[str]
    overwrite: bool
    require_exact_match: bool


@dataclass(frozen=True)
class TopKConfig:
    """Top-k search configuration."""

    k_values: list[int]
    tau_rmse: float
    tau_width: float


def validate_stage(stage: str) -> str:
    """Validate stage name and return normalized value."""
    normalized = str(stage).strip()
    if normalized not in SUPPORTED_STAGES:
        raise ValueError(
            "Unsupported stage='{}'. Supported stages: {}".format(
                normalized, sorted(SUPPORTED_STAGES)
            )
        )
    return normalized


def validate_target(target: str) -> str:
    """Validate target name and return normalized value."""
    normalized = str(target).strip().upper()
    if normalized not in SUPPORTED_TARGETS:
        raise ValueError(
            "Unsupported target='{}'. Supported targets: {}".format(
                normalized, sorted(SUPPORTED_TARGETS)
            )
        )
    return normalized


def parse_feature_config(cfg: DictConfig) -> FeatureConfig:
    """Parse and validate feature configuration."""
    columns = [str(col) for col in cfg.features.columns]
    if not columns:
        raise ValueError("features.columns must be a non-empty list.")

    feature_set_id = str(cfg.features.get("id") or "")
    if not feature_set_id:
        feature_set_id = "feature_hash"

    return FeatureConfig(
        columns=columns,
        feature_set_id=feature_set_id,
    )


def parse_split_config(cfg: DictConfig) -> SplitConfig:
    """Parse split-related configuration."""
    return SplitConfig(
        train_cells_proportion=float(cfg.split.train_cells_proportion),
        seed=int(cfg.split.seed),
        split_dir=Path(str(cfg.split.split_dir)),
        force_recreate=bool(cfg.split.get("force_recreate", False)),
    )


def parse_model_config(cfg: DictConfig) -> ModelConfig:
    """Parse model configuration."""
    model_name = str(cfg.model.name).strip().lower()
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            "Unsupported model.name='{}'. Supported: {}".format(
                model_name, sorted(SUPPORTED_MODEL_NAMES)
            )
        )
    return ModelConfig(
        name=model_name,
        random_seed=int(cfg.split.seed),
        n_jobs=int(cfg.model.n_jobs),
    )


def parse_optimize_config(cfg: DictConfig) -> OptimizeConfig:
    """Parse optimization configuration."""
    objective_cfg = cfg.optimize.get("objective", {})
    n_jobs = int(cfg.optimize.get("n_jobs", DEFAULT_OPTIMIZE_N_JOBS))
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(
            "optimize.n_jobs must be -1 or a positive integer."
        )
    return OptimizeConfig(
        enabled=bool(cfg.optimize.get("enabled", True)),
        n_trials=int(cfg.optimize.n_trials),
        n_jobs=n_jobs,
        cv_folds=int(cfg.optimize.cv_folds),
        lambda_gap=float(
            objective_cfg.get("lambda_gap", DEFAULT_OBJECTIVE_LAMBDA_GAP)
        ),
        save_cv_trials=bool(cfg.optimize.get("save_cv_trials", True)),
    )


def parse_conformal_config(cfg: DictConfig) -> ConformalConfig:
    """Parse conformal configuration."""
    return ConformalConfig(
        enabled=bool(cfg.conformal.get("enabled", True)),
        alpha=float(cfg.conformal.get("alpha", DEFAULT_CONFORMAL_ALPHA)),
        calibration_proportion=float(
            cfg.conformal.get(
                "calibration_proportion",
                DEFAULT_CONFORMAL_CALIBRATION_PROPORTION,
            )
        ),
    )


def parse_ranking_config(cfg: DictConfig) -> RankingConfig:
    """Parse ranking configuration."""
    weights_cfg = cfg.ranking.get("weights", {})
    rescale_cfg = cfg.ranking.get("rescale", {})
    n_permutations = cfg.ranking.get("n_permutations", None)
    if n_permutations is None:
        n_permutations = cfg.ranking.get(
            "n_repeats",
            DEFAULT_RANKING_N_PERMUTATIONS,
        )
    n_permutations = int(n_permutations)
    if n_permutations <= 0:
        raise ValueError("ranking.n_permutations must be > 0.")
    return RankingConfig(
        n_permutations=n_permutations,
        w_rmse=float(weights_cfg.get("rmse", DEFAULT_RANKING_WEIGHT_RMSE)),
        w_uncertainty=float(
            weights_cfg.get("uncertainty", DEFAULT_RANKING_WEIGHT_UNCERTAINTY)
        ),
        clip_low_q=float(
            rescale_cfg.get("clip_low_q", DEFAULT_RANKING_CLIP_LOW_Q)
        ),
        clip_high_q=float(
            rescale_cfg.get("clip_high_q", DEFAULT_RANKING_CLIP_HIGH_Q)
        ),
    )


def parse_topk_config(cfg: DictConfig) -> TopKConfig:
    """Parse top-k selection configuration."""
    topk_cfg = cfg.get("topk", {})
    constraints_cfg = topk_cfg.get("constraints", {})
    return TopKConfig(
        k_values=[int(k) for k in topk_cfg.get("k_values", [])],
        tau_rmse=float(constraints_cfg.get("tau_rmse", DEFAULT_TOPK_TAU_RMSE)),
        tau_width=float(
            constraints_cfg.get("tau_width", DEFAULT_TOPK_TAU_WIDTH)
        ),
    )


def parse_robustness_config(cfg: DictConfig) -> RobustnessConfig:
    """Parse robustness configuration."""
    mode = str(cfg.robustness.get("mode", "protocol_lopo"))
    if mode != "protocol_lopo":
        raise ValueError(
            "Unsupported robustness.mode='{}'. "
            "Supported: ['protocol_lopo']".format(mode)
        )
    return RobustnessConfig(
        mode=mode,
        protocol_column=str(
            cfg.data.get("protocol_column", DEFAULT_PROTOCOL_COLUMN)
        ),
    )


def parse_weighting_config(cfg: DictConfig) -> WeightingConfig:
    """Parse weighting configuration."""
    enabled = bool(cfg.weighting.get("enabled", False))
    strategy = str(
        cfg.weighting.get("strategy", DEFAULT_WEIGHTING_STRATEGY)
    ).strip()
    if not enabled:
        strategy = "none"
    if strategy not in SUPPORTED_WEIGHTING_STRATEGIES:
        raise ValueError(
            "Unsupported weighting.strategy='{}'. Supported: {}".format(
                strategy,
                sorted(SUPPORTED_WEIGHTING_STRATEGIES),
            )
        )
    return WeightingConfig(
        enabled=enabled,
        strategy=strategy,
        n_bins=int(cfg.weighting.get("n_bins", DEFAULT_SAMPLE_WEIGHT_N_BINS)),
        long_life_quantile=float(
            cfg.weighting.get("long_life_quantile", DEFAULT_LONG_LIFE_QUANTILE)
        ),
        long_life_boost_factor=float(
            cfg.weighting.get(
                "long_life_boost_factor", DEFAULT_LONG_LIFE_BOOST_FACTOR
            )
        ),
    )


def parse_artifacts_config(cfg: DictConfig) -> ArtifactsConfig:
    """Parse artifact storage configuration."""
    run_key_fields = [
        str(value)
        for value in cfg.artifacts.get(
            "run_key_fields",
            [
                "target",
                "feature_hash",
                "split_seed",
                "model_name",
                "weighting_strategy",
            ],
        )
    ]
    return ArtifactsConfig(
        root_dir=Path(str(cfg.artifacts.root_dir)),
        naming_key=str(cfg.artifacts.get("naming_key", "feature_id_or_hash")),
        run_key_fields=run_key_fields,
        overwrite=bool(
            cfg.artifacts.get("overwrite", DEFAULT_ARTIFACTS_OVERWRITE)
        ),
        require_exact_match=bool(
            cfg.artifacts.get(
                "require_exact_match", DEFAULT_ARTIFACTS_REQUIRE_EXACT_MATCH
            )
        ),
    )


def stage_context_dict(
    stage: str,
    target: str,
    feature_set_id: str,
    feature_hash: str,
    split_seed: int,
    model_name: str,
    weighting_strategy: str,
    k_selected: int | None,
) -> dict[str, Any]:
    """Return a normalized metadata context dictionary for stages."""
    return {
        "stage": str(stage),
        "target": str(target),
        "feature_set_id": str(feature_set_id),
        "feature_hash": str(feature_hash),
        "split_seed": int(split_seed),
        "model_name": str(model_name),
        "weighting_strategy": str(weighting_strategy),
        "k_selected": (int(k_selected) if k_selected is not None else None),
    }
