"""Integration tests for deterministic split behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from severson_features_soh_rul.modeling.data.split import (
    create_or_load_cell_split,
)


def test_split_reproducibility_with_fixed_seed(tmp_path: Path) -> None:
    """Split generation should be deterministic and reload from disk."""
    rows = []
    for cell_id in [f"c{i}" for i in range(12)]:
        for cycle in range(3):
            rows.append({"cell": cell_id, "cycle": cycle})
    features_df = pd.DataFrame(rows)

    train_cells_a, test_cells_a = create_or_load_cell_split(
        features_df=features_df,
        split_dir=tmp_path,
        train_cells_proportion=0.75,
        split_seed=42,
        force_recreate=False,
    )
    train_cells_b, test_cells_b = create_or_load_cell_split(
        features_df=features_df,
        split_dir=tmp_path,
        train_cells_proportion=0.75,
        split_seed=42,
        force_recreate=False,
    )

    assert train_cells_a == train_cells_b
    assert test_cells_a == test_cells_b
    assert set(train_cells_a).isdisjoint(set(test_cells_a))
