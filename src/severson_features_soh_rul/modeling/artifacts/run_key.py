"""Deterministic run-key construction utilities."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def build_run_key_components(
    target: str,
    feature_hash: str,
    split_seed: int,
    model_name: str,
    weighting_strategy: str,
    k_selected: int | None = None,
) -> dict[str, Any]:
    """Build normalized run-key components."""
    payload: dict[str, Any] = {
        "target": str(target).upper(),
        "feature_hash": str(feature_hash),
        "split_seed": int(split_seed),
        "model_name": str(model_name).lower(),
        "weighting_strategy": str(weighting_strategy),
    }
    if k_selected is not None:
        payload["k_selected"] = int(k_selected)
    return payload


def serialize_run_key(
    components: dict[str, Any],
    run_key_fields: list[str],
) -> str:
    """Serialize run-key components into a deterministic directory key."""
    parts: list[str] = []
    for field in run_key_fields:
        if field not in components:
            continue
        value = _sanitize_component(str(components[field]))
        parts.append(f"{field}-{value}")
    if not parts:
        raise ValueError("run_key_fields produced an empty run key.")
    key_body = "__".join(parts)
    digest = hashlib.sha256(
        json.dumps(components, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"{key_body}__rk-{digest}"


def _sanitize_component(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "na"
