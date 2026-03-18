"""Dataset loading and feature-view resolution for experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from severson_features_soh_rul.modeling.schemas import (
    BASE_REQUIRED_COLUMNS,
    CHARGE_FEATURE_COLUMNS,
    CHARGE_NO_TEMPERATURE_FEATURE_COLUMNS,
    CHARGE_TOPK_FEATURES_BY_TARGET,
    FULL_CYCLE_FEATURE_COLUMNS,
    FULL_NO_TEMPERATURE_FEATURE_COLUMNS,
    FULL_TOPK_FEATURES_BY_TARGET,
    SUPPORTED_FEATURE_SET_IDS,
    SUPPORTED_TARGETS,
    validate_required_columns,
)


def load_features_dataframe(features_data_path: Path) -> pd.DataFrame:
    """Load features parquet and validate baseline schema."""
    features_df = pd.read_parquet(features_data_path)
    required_cols = BASE_REQUIRED_COLUMNS + FULL_CYCLE_FEATURE_COLUMNS
    validate_required_columns(
        features_df=features_df, required_columns=required_cols
    )
    return features_df


def resolve_feature_columns(
    feature_set_id: str,
    target: str,
) -> list[str]:
    """Resolve feature list according to fixed revision feature-set IDs."""
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported target={target}. Supported targets: {sorted(SUPPORTED_TARGETS)}"
        )

    if feature_set_id not in SUPPORTED_FEATURE_SET_IDS:
        raise ValueError(
            f"Unsupported feature_set_id={feature_set_id}. "
            f"Supported: {sorted(SUPPORTED_FEATURE_SET_IDS)}"
        )

    if feature_set_id == "full_all":
        return FULL_CYCLE_FEATURE_COLUMNS.copy()

    if feature_set_id == "full_topk":
        return FULL_TOPK_FEATURES_BY_TARGET[target].copy()

    if feature_set_id == "full_no_temp":
        return FULL_NO_TEMPERATURE_FEATURE_COLUMNS.copy()

    if feature_set_id == "charge_all":
        return CHARGE_FEATURE_COLUMNS.copy()

    if feature_set_id == "charge_topk":
        return CHARGE_TOPK_FEATURES_BY_TARGET[target].copy()

    if feature_set_id == "charge_no_temp":
        return CHARGE_NO_TEMPERATURE_FEATURE_COLUMNS.copy()

    raise RuntimeError(f"Unhandled feature_set_id={feature_set_id}")
