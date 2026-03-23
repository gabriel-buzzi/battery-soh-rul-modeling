"""Unit tests for model builder."""

from __future__ import annotations

import pytest

from severson_features_soh_rul.modeling.core.models import build_model


def test_build_model_requires_quantile_model_name() -> None:
    """Builder should return quantile model without model name parameter."""
    pytest.importorskip("quantile_forest")
    model = build_model(
        model_params=None,
        random_seed=42,
        n_jobs=1,
    )
    assert model.__class__.__name__ == "ExtraTreesQuantileRegressor"


def test_build_model_quantile_default_quantile_is_median() -> None:
    """Quantile model should default to median for point predictions."""
    pytest.importorskip("quantile_forest")
    model = build_model(
        model_params={"n_estimators": 10},
        random_seed=42,
        n_jobs=1,
    )
    assert model.get_params()["default_quantiles"] == 0.5
