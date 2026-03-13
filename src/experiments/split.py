"""Cell-wise train/test split utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_or_load_cell_split(
    features_df: pd.DataFrame,
    split_dir: Path,
    train_cells_proportion: float,
    random_seed: int,
    force_recreate: bool = False,
) -> tuple[list[str], list[str]]:
    """Create or load deterministic train/test cell split.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature dataframe containing at least a `cell` column.
    split_dir : Path
        Directory where split JSON artifacts are persisted.
    train_cells_proportion : float
        Train proportion in (0, 1).
    random_seed : int
        Random seed used in cell split creation.
    force_recreate : bool
        If True, regenerate split files even if they already exist.

    Returns
    -------
    tuple[list[str], list[str]]
        Train cell list and test cell list.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    train_cells_path = split_dir / "train_cells.json"
    test_cells_path = split_dir / "test_cells.json"

    split_exists = train_cells_path.exists() and test_cells_path.exists()
    if split_exists and not force_recreate:
        train_cells = _read_cells_json(train_cells_path)
        test_cells = _read_cells_json(test_cells_path)
        _validate_no_overlap(train_cells, test_cells)
        return train_cells, test_cells

    unique_cells = sorted(features_df["cell"].astype(str).unique().tolist())

    train_cells, test_cells = train_test_split(
        unique_cells,
        train_size=train_cells_proportion,
        random_state=random_seed,
        shuffle=True,
    )
    train_cells = sorted(train_cells)
    test_cells = sorted(test_cells)
    _validate_no_overlap(train_cells, test_cells)

    train_cells_path.write_text(json.dumps(train_cells, indent=2))
    test_cells_path.write_text(json.dumps(test_cells, indent=2))

    return train_cells, test_cells


def apply_cell_split(
    features_df: pd.DataFrame,
    train_cells: list[str],
    test_cells: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition features dataframe into train and test by cell membership."""
    _validate_no_overlap(train_cells, test_cells)
    train_df = features_df[
        features_df["cell"].astype(str).isin(train_cells)
    ].copy()
    test_df = features_df[
        features_df["cell"].astype(str).isin(test_cells)
    ].copy()
    return train_df, test_df


def _read_cells_json(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    return [str(cell_id) for cell_id in raw]


def _validate_no_overlap(
    train_cells: list[str], test_cells: list[str]
) -> None:
    overlap = sorted(set(train_cells).intersection(test_cells))
    if overlap:
        raise ValueError(
            "Cell leakage detected. Train/test overlap is not allowed. "
            f"Overlapping cells: {overlap}"
        )
