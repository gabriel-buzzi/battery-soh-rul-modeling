"""Artifact writing utilities with atomic semantics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import joblib
from omegaconf import OmegaConf
import pandas as pd

RUN_INFO_SCHEMA_VERSION = "2.0.0"
PRODUCER_NAME = "severson_features_soh_rul.modeling.pipeline"
PRODUCER_VERSION = "1.0.0"


def prepare_stage_dir(
    root_dir: Path,
    run_key: str,
    stage: str,
    required_files: list[str],
    overwrite: bool,
) -> tuple[Path, bool]:
    """Prepare stage directory and evaluate idempotent skip condition."""
    stage_dir = root_dir / run_key / stage
    if (
        stage_dir.exists()
        and not overwrite
        and all((stage_dir / name).exists() for name in required_files)
    ):
        return stage_dir, True
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir, False


def write_resolved_config(cfg: Any, stage_dir: Path) -> Path:
    """Write resolved Hydra configuration file."""
    output_path = stage_dir / "config.resolved.yaml"
    yaml_payload = OmegaConf.to_yaml(cfg, resolve=True)
    _atomic_write_text(output_path, yaml_payload)
    return output_path


def write_run_info(
    stage_dir: Path,
    run_key: str,
    context: dict[str, Any],
) -> Path:
    """Write stage run metadata payload."""
    output_path = stage_dir / "run_info.json"
    payload: dict[str, Any] = {
        "schema_version": RUN_INFO_SCHEMA_VERSION,
        "producer": PRODUCER_NAME,
        "producer_version": PRODUCER_VERSION,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_key": run_key,
        "git_sha": _git_sha_or_none(),
        **context,
    }
    write_json_atomic(output_path=output_path, payload=payload)
    return output_path


def write_json_atomic(output_path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically."""
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    _atomic_write_text(output_path, serialized)
    return output_path


def write_csv_atomic(output_path: Path, df: pd.DataFrame) -> Path:
    """Write CSV atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=".tmp_",
        dir=str(output_path.parent),
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return output_path


def write_parquet_atomic(output_path: Path, df: pd.DataFrame) -> Path:
    """Write Parquet atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".tmp",
        prefix=".tmp_",
        dir=str(output_path.parent),
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return output_path


def write_joblib_atomic(output_path: Path, payload: Any) -> Path:
    """Write joblib artifact atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".tmp",
        prefix=".tmp_",
        dir=str(output_path.parent),
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        joblib.dump(payload, tmp_path)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return output_path


def _atomic_write_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=".tmp_",
        dir=str(output_path.parent),
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(text)
    try:
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _git_sha_or_none() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    value = output.strip()
    return value if value else None
