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

SUPPORTED_TARGETS = {"SOH", "RUL"}

FULL_TOPK_FEATURES_BY_TARGET = {
    "SOH": ["V_entropy", "V_std", "I_iqr", "V_iqr", "I_kurtosis", "V_median"],
    "RUL": ["V_iqr", "I_std", "V_entropy", "V_std", "I_mean", "I_median"],
}

CHARGE_TOPK_FEATURES_BY_TARGET = {
    "SOH": [
        "charge_V_median",
        "charge_I_median",
        "charge_V_entropy",
        "charge_V_std",
        "charge_V_iqr",
        "charge_I_std",
    ],
    "RUL": [
        "charge_V_median",
        "charge_I_median",
        "charge_V_entropy",
        "charge_I_std",
    ],
}

FULL_NO_TEMPERATURE_FEATURE_COLUMNS = [
    col
    for col in FULL_CYCLE_FEATURE_COLUMNS
    if col not in TEMPERATURE_FEATURE_COLUMNS
]

CHARGE_NO_TEMPERATURE_FEATURE_COLUMNS = [
    col
    for col in CHARGE_FEATURE_COLUMNS
    if col not in {f"charge_{x}" for x in TEMPERATURE_FEATURE_COLUMNS}
]

SUPPORTED_FEATURE_SET_IDS = {
    "full_all",
    "full_topk",
    "full_no_temp",
    "charge_all",
    "charge_topk",
    "charge_no_temp",
}


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
