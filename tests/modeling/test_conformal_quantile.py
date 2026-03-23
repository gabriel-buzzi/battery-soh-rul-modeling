"""Unit tests for quantile conformal helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from severson_features_soh_rul.modeling.core.conformal import (
    fit_conformal_model,
    predict_with_intervals,
)
from severson_features_soh_rul.modeling.core.models import build_model


def test_quantile_conformal_interval_width_varies_by_sample() -> None:
    """Quantile conformal should produce sample-adaptive interval widths."""
    pytest.importorskip("mapie")
    pytest.importorskip("quantile_forest")

    rng = np.random.default_rng(42)
    n = 180
    x = rng.uniform(0.0, 1.0, size=n)
    noise = rng.normal(loc=0.0, scale=0.2 + x, size=n)
    y = 3.0 * x + noise
    X = pd.DataFrame({"x": x})
    y_series = pd.Series(y)
    groups = pd.Series([f"cell_{i // 6}" for i in range(n)])

    base_model = build_model(
        model_params={"n_estimators": 80, "max_depth": 8},
        random_seed=42,
        n_jobs=1,
    )
    bundle = fit_conformal_model(
        base_model=base_model,
        X_train=X,
        y_train=y_series,
        groups_train=groups,
        conformal_enabled=True,
        confidence_level=0.9,
        calibration_proportion=0.25,
        random_seed=42,
    )
    _, y_lo, y_hi = predict_with_intervals(model_bundle=bundle, X=X)

    widths = np.asarray(y_hi - y_lo, dtype=float)
    assert np.isfinite(widths).all()
    assert float(np.std(widths)) > 0.0
