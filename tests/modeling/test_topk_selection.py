"""Unit tests for deterministic top-k selection rule."""

from __future__ import annotations

import pandas as pd

from severson_features_soh_rul.modeling.stages.topk_sweep import (
    select_k_from_sweep_df,
)


def test_select_smallest_feasible_k() -> None:
    """When feasible rows exist, smallest k should be selected."""
    sweep_df = pd.DataFrame(
        [
            {
                "k": 2,
                "rmse_mean": 1.1,
                "interval_width_mean": 0.9,
                "is_feasible": True,
            },
            {
                "k": 4,
                "rmse_mean": 1.0,
                "interval_width_mean": 0.8,
                "is_feasible": True,
            },
            {
                "k": 8,
                "rmse_mean": 0.9,
                "interval_width_mean": 0.7,
                "is_feasible": False,
            },
        ]
    )
    selected, mode = select_k_from_sweep_df(sweep_df)
    assert int(selected["k"]) == 2
    assert mode == "smallest_feasible"


def test_select_lexicographic_fallback() -> None:
    """Fallback should minimize rmse, then width, then k."""
    sweep_df = pd.DataFrame(
        [
            {
                "k": 2,
                "rmse_mean": 1.0,
                "interval_width_mean": 0.9,
                "is_feasible": False,
            },
            {
                "k": 4,
                "rmse_mean": 1.0,
                "interval_width_mean": 0.8,
                "is_feasible": False,
            },
            {
                "k": 6,
                "rmse_mean": 1.1,
                "interval_width_mean": 0.7,
                "is_feasible": False,
            },
        ]
    )
    selected, mode = select_k_from_sweep_df(sweep_df)
    assert int(selected["k"]) == 4
    assert mode == "lexicographic_fallback"
