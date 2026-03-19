"""Unit tests for feature signature helpers."""

from __future__ import annotations

from severson_features_soh_rul.modeling.data.features import (
    build_feature_hash,
    feature_set_id_from_config,
)


def test_feature_hash_order_invariant_is_stable() -> None:
    """Order-invariant hash should ignore column ordering."""
    cols_a = ["a", "b", "c"]
    cols_b = ["c", "a", "b"]
    assert build_feature_hash(cols_a, "order_invariant") == build_feature_hash(
        cols_b, "order_invariant"
    )


def test_feature_hash_order_sensitive_changes_with_order() -> None:
    """Order-sensitive hash should depend on sequence order."""
    cols_a = ["a", "b", "c"]
    cols_b = ["c", "a", "b"]
    assert build_feature_hash(cols_a, "order_sensitive") != build_feature_hash(
        cols_b, "order_sensitive"
    )


def test_feature_set_id_falls_back_to_hash() -> None:
    """Feature set id should resolve to hash when id is absent."""
    assert feature_set_id_from_config("", "abc123") == "abc123"
    assert feature_set_id_from_config("feature_hash", "abc123") == "abc123"
    assert feature_set_id_from_config("my_set", "abc123") == "my_set"
