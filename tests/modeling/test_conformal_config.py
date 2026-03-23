"""Unit tests for conformal configuration parsing."""

from __future__ import annotations

from omegaconf import OmegaConf
import pytest

from severson_features_soh_rul.modeling.config.schema import (
    parse_conformal_config,
)


def test_parse_conformal_uses_confidence_level_when_provided() -> None:
    """confidence_level should be parsed as source-of-truth."""
    cfg = OmegaConf.create(
        {"conformal": {"enabled": True, "confidence_level": 0.95}}
    )
    parsed = parse_conformal_config(cfg)
    assert parsed.enabled is True
    assert parsed.confidence_level == pytest.approx(0.95)


def test_parse_conformal_validates_confidence_level_range() -> None:
    """confidence_level must be strictly within (0, 1)."""
    cfg = OmegaConf.create({"conformal": {"enabled": True, "confidence_level": 1.0}})
    with pytest.raises(ValueError):
        parse_conformal_config(cfg)
