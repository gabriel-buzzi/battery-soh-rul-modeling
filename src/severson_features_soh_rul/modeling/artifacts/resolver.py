"""Metadata-driven artifact resolution utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_unique_stage_dir(
    artifacts_root: Path,
    stage: str,
    match_fields: dict[str, Any],
    require_exact_match: bool,
) -> Path:
    """Resolve a unique stage directory by metadata match."""
    matches = find_stage_dirs(
        artifacts_root=artifacts_root,
        stage=stage,
        match_fields=match_fields,
    )
    if not matches:
        raise FileNotFoundError(
            _format_resolver_error(
                stage=stage,
                message="no matching artifacts found",
                match_fields=match_fields,
            )
        )

    if require_exact_match and len(matches) != 1:
        raise RuntimeError(
            _format_resolver_error(
                stage=stage,
                message="ambiguous artifact match",
                match_fields=match_fields,
                match_count=len(matches),
            )
        )

    if len(matches) == 1:
        return matches[0]

    ordered = sorted(matches, key=lambda path: path.as_posix())
    return ordered[-1]


def find_stage_dirs(
    artifacts_root: Path,
    stage: str,
    match_fields: dict[str, Any],
) -> list[Path]:
    """Find stage directories matching metadata predicates."""
    if not artifacts_root.exists():
        return []

    results: list[Path] = []
    for run_info_path in artifacts_root.glob(f"*/{stage}/run_info.json"):
        stage_dir = run_info_path.parent
        run_info = _read_json(run_info_path)
        if _matches(run_info=run_info, match_fields=match_fields):
            results.append(stage_dir)
    return sorted(results, key=lambda path: path.as_posix())


def resolve_required_file(stage_dir: Path, file_name: str, stage: str) -> Path:
    """Resolve required file inside stage directory or fail."""
    path = stage_dir / file_name
    if not path.exists():
        raise FileNotFoundError(
            "Stage '{}' is missing required artifact '{}' in '{}'".format(
                stage, file_name, stage_dir
            )
        )
    return path


def _matches(run_info: dict[str, Any], match_fields: dict[str, Any]) -> bool:
    for key, expected in match_fields.items():
        observed = run_info.get(key)
        if expected is None:
            if observed is not None:
                return False
            continue
        if str(observed) != str(expected):
            return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format_resolver_error(
    stage: str,
    message: str,
    match_fields: dict[str, Any],
    match_count: int | None = None,
) -> str:
    expected_fields = ", ".join(
        f"{key}={value}"
        for key, value in sorted(match_fields.items())
        if value is not None
    )
    if match_count is None:
        return (
            "Resolver failure for stage '{}': {}. expected run-key fields: {}"
        ).format(stage, message, expected_fields)
    return (
        "Resolver failure for stage '{}': {} (matches={})."
        "expected run-key fields: {}"
    ).format(stage, message, match_count, expected_fields)
