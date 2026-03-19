"""Unit tests for effective model n_jobs resolution."""

from __future__ import annotations

from severson_features_soh_rul.modeling.stages.common import (
    resolve_effective_model_n_jobs,
)


def test_optimize_stage_keeps_model_n_jobs() -> None:
    """Optimize stage should use model.n_jobs as-is."""
    assert (
        resolve_effective_model_n_jobs(
            stage="optimize",
            model_n_jobs=3,
            optimize_n_jobs=5,
        )
        == 3
    )


def test_non_optimize_stage_multiplies_n_jobs() -> None:
    """Non-optimize stages should multiply model and optimize n_jobs."""
    assert (
        resolve_effective_model_n_jobs(
            stage="fit_final_model",
            model_n_jobs=2,
            optimize_n_jobs=4,
        )
        == 8
    )


def test_non_optimize_stage_supports_negative_optimize_n_jobs() -> None:
    """Negative optimize.n_jobs should propagate through multiplication."""
    assert (
        resolve_effective_model_n_jobs(
            stage="predict",
            model_n_jobs=2,
            optimize_n_jobs=-1,
        )
        == -2
    )
