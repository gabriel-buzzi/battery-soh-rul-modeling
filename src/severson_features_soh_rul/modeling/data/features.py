"""Feature-table loading, validation, and feature-signature helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

BASE_COLUMNS = ["cell", "cycle", "SOH", "RUL"]


def load_features_dataframe(features_path: Path) -> pd.DataFrame:
    """Load the features table from parquet.

    Parameters
    ----------
    features_path : Path
        Input parquet path.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features parquet not found at '{features_path}'."
        )
    return pd.read_parquet(features_path)


def validate_required_columns(
    features_df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    protocol_column: str | None = None,
) -> None:
    """Validate required columns for modeling stages.

    Parameters
    ----------
    features_df : pd.DataFrame
        Modeling input table.
    target : str
        Prediction target.
    feature_columns : list[str]
        Configured feature columns.
    protocol_column : str | None
        Optional protocol column requirement.
    """
    required = [*BASE_COLUMNS, str(target), *feature_columns]
    if protocol_column is not None:
        required.append(str(protocol_column))
    missing = [col for col in required if col not in features_df.columns]
    if missing:
        raise ValueError(
            "Missing required input columns: {}.".format(sorted(set(missing)))
        )


def build_feature_hash(
    feature_columns: list[str],
    hash_mode: str,
) -> str:
    """Build deterministic feature hash from configured columns.

    Parameters
    ----------
    feature_columns : list[str]
        Configured feature list.
    hash_mode : str
        Either ``order_invariant`` or ``order_sensitive``.

    Returns
    -------
    str
        SHA256 hash hex digest.
    """
    if hash_mode == "order_invariant":
        payload = sorted([str(col) for col in feature_columns])
    elif hash_mode == "order_sensitive":
        payload = [str(col) for col in feature_columns]
    else:
        raise ValueError(
            "Unsupported hash_mode='{}'. Expected one of ['order_invariant', "
            "'order_sensitive'].".format(hash_mode)
        )
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=False).encode(
            "utf-8"
        )
    ).hexdigest()


def feature_set_id_from_config(feature_id: str, feature_hash: str) -> str:
    """Return effective feature-set identifier.

    Parameters
    ----------
    feature_id : str
        Optional human-readable feature set id.
    feature_hash : str
        Deterministic feature hash.

    Returns
    -------
    str
        Resolved feature set identifier.
    """
    cleaned = str(feature_id).strip()
    return cleaned if cleaned and cleaned != "feature_hash" else feature_hash
