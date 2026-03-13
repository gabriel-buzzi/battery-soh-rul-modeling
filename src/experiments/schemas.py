"""Schema constants and validation for experiment inputs."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

FULL_CYCLE_FEATURE_COLUMNS = [
    "V_mean",
    "V_median",
    "V_std",
    "V_iqr",
    "V_kurtosis",
    "V_entropy",
    "I_mean",
    "I_median",
    "I_std",
    "I_iqr",
    "I_kurtosis",
    "T_mean",
    "T_median",
    "T_std",
    "T_iqr",
    "T_kurtosis",
]

TEMPERATURE_FEATURE_COLUMNS = [
    "T_mean",
    "T_median",
    "T_std",
    "T_iqr",
    "T_kurtosis",
]

CHARGE_FEATURE_COLUMNS = [
    f"charge_{feature_name}" for feature_name in FULL_CYCLE_FEATURE_COLUMNS
]

BASE_REQUIRED_COLUMNS = ["cell", "cycle", "SOH", "RUL"]

SUPPORTED_TARGETS = {"SOH", "RUL", "RUL_THROUGHPUT"}


def validate_required_columns(
    features_df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Fail fast if required columns are missing from dataframe.

    Parameters
    ----------
    features_df : pd.DataFrame
        Dataframe loaded from features parquet.
    required_columns : Iterable[str]
        Set of columns that must exist.
    """
    missing = [
        col for col in required_columns if col not in features_df.columns
    ]
    if missing:
        msg = (
            "Missing required feature columns: "
            f"{missing}. Please regenerate features or fix input path."
        )
        raise ValueError(msg)
