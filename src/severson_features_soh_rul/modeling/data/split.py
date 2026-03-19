"""Deterministic cell-level split utilities for modeling stages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_or_load_cell_split(
    features_df: pd.DataFrame,
    split_dir: Path,
    train_cells_proportion: float,
    split_seed: int,
    force_recreate: bool,
) -> tuple[list[str], list[str]]:
    """Create or load deterministic train/test cell split."""
    split_seed_dir = split_dir / f"seed_{int(split_seed)}"
    split_seed_dir.mkdir(parents=True, exist_ok=True)

    train_cells_path = split_seed_dir / "train_cells.json"
    test_cells_path = split_seed_dir / "test_cells.json"
    if (
        train_cells_path.exists()
        and test_cells_path.exists()
        and not force_recreate
    ):
        train_cells = _read_cells(train_cells_path)
        test_cells = _read_cells(test_cells_path)
        _validate_no_overlap(train_cells=train_cells, test_cells=test_cells)
        return train_cells, test_cells

    unique_cells = sorted(features_df["cell"].astype(str).unique().tolist())
    train_cells, test_cells = train_test_split(
        unique_cells,
        train_size=float(train_cells_proportion),
        random_state=int(split_seed),
        shuffle=True,
    )
    train_cells = sorted([str(cell) for cell in train_cells])
    test_cells = sorted([str(cell) for cell in test_cells])
    _validate_no_overlap(train_cells=train_cells, test_cells=test_cells)

    train_cells_path.write_text(json.dumps(train_cells, indent=2))
    test_cells_path.write_text(json.dumps(test_cells, indent=2))
    return train_cells, test_cells


def apply_cell_split(
    features_df: pd.DataFrame,
    train_cells: list[str],
    test_cells: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and test partitions using cell IDs."""
    _validate_no_overlap(train_cells=train_cells, test_cells=test_cells)
    train_df = features_df[
        features_df["cell"].astype(str).isin(train_cells)
    ].copy()
    test_df = features_df[
        features_df["cell"].astype(str).isin(test_cells)
    ].copy()
    return train_df, test_df


def _read_cells(path: Path) -> list[str]:
    rows = json.loads(path.read_text())
    return [str(value) for value in rows]


def _validate_no_overlap(
    train_cells: list[str], test_cells: list[str]
) -> None:
    overlap = sorted(set(train_cells).intersection(test_cells))
    if overlap:
        raise ValueError(
            "Cell leakage detected: train/test overlap found in {}".format(
                overlap
            )
        )
