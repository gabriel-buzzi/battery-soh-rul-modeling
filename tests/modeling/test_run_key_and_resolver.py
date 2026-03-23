"""Unit tests for run-key and resolver behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from severson_features_soh_rul.modeling.artifacts.resolver import (
    resolve_unique_stage_dir,
)
from severson_features_soh_rul.modeling.artifacts.run_key import (
    build_run_key_components,
    serialize_run_key,
)


def test_run_key_is_deterministic() -> None:
    """Run key should be stable for identical components and fields."""
    components = build_run_key_components(
        target="SOH",
        feature_hash="abc",
        split_seed=42,
        model_name="extratrees_quantile",
        weighting_strategy="none",
        k_selected=8,
    )
    fields = [
        "target",
        "feature_hash",
        "split_seed",
        "model_name",
        "weighting_strategy",
        "k_selected",
    ]
    key_a = serialize_run_key(components, fields)
    key_b = serialize_run_key(components, fields)
    assert key_a == key_b


def test_resolver_fails_on_ambiguous_match(tmp_path: Path) -> None:
    """Resolver should fail when exact match is required and ambiguous."""
    run_info = {
        "target": "SOH",
        "feature_hash": "abc",
        "split_seed": 42,
        "model_name": "extratrees_quantile",
        "weighting_strategy": "none",
        "k_selected": None,
    }
    for run_id in ["run_a", "run_b"]:
        stage_dir = tmp_path / run_id / "optimize"
        stage_dir.mkdir(parents=True)
        (stage_dir / "run_info.json").write_text(json.dumps(run_info))

    with pytest.raises(RuntimeError):
        resolve_unique_stage_dir(
            artifacts_root=tmp_path,
            stage="optimize",
            match_fields=run_info,
            require_exact_match=True,
        )


def test_resolver_can_match_none_key(tmp_path: Path) -> None:
    """Resolver should allow matching null k_selected values."""
    stage_a = tmp_path / "run_a" / "fit_final_model"
    stage_b = tmp_path / "run_b" / "fit_final_model"
    stage_a.mkdir(parents=True)
    stage_b.mkdir(parents=True)

    (stage_a / "run_info.json").write_text(
        json.dumps(
            {
                "target": "SOH",
                "feature_hash": "abc",
                "split_seed": 42,
                "model_name": "extratrees_quantile",
                "weighting_strategy": "none",
                "k_selected": None,
            }
        )
    )
    (stage_b / "run_info.json").write_text(
        json.dumps(
            {
                "target": "SOH",
                "feature_hash": "abc",
                "split_seed": 42,
                "model_name": "extratrees_quantile",
                "weighting_strategy": "none",
                "k_selected": 4,
            }
        )
    )

    resolved = resolve_unique_stage_dir(
        artifacts_root=tmp_path,
        stage="fit_final_model",
        match_fields={
            "target": "SOH",
            "feature_hash": "abc",
            "split_seed": 42,
            "model_name": "extratrees_quantile",
            "weighting_strategy": "none",
            "k_selected": None,
        },
        require_exact_match=True,
    )
    assert resolved == stage_a
