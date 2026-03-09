"""Dataset loading and feature-view resolution for experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.schemas import (
    BASE_REQUIRED_COLUMNS,
    CHARGE_FEATURE_COLUMNS,
    FULL_CYCLE_FEATURE_COLUMNS,
    TEMPERATURE_FEATURE_COLUMNS,
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
    feature_view: str,
    custom_features: list[str] | None = None,
) -> list[str]:
    """Resolve feature list according to configured feature view."""
    if feature_view == "full_all":
        return FULL_CYCLE_FEATURE_COLUMNS.copy()

    if feature_view == "charge_all":
        return CHARGE_FEATURE_COLUMNS.copy()

    if feature_view == "full_plus_charge_all":
        return FULL_CYCLE_FEATURE_COLUMNS + CHARGE_FEATURE_COLUMNS

    if feature_view == "no_temperature":
        return [
            col
            for col in FULL_CYCLE_FEATURE_COLUMNS
            if col not in TEMPERATURE_FEATURE_COLUMNS
        ]

    if feature_view == "custom":
        if not custom_features:
            raise ValueError(
                "feature_view='custom' requires a non-empty"
                "custom_features list."
            )
        return custom_features

    raise ValueError(
        f"Unsupported feature_view={feature_view}. "
        "Use one of "
        "['full_all', 'charge_all', 'full_plus_charge_all', "
        "'no_temperature', 'custom']."
    )
