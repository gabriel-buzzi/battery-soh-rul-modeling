"""Run artifacts I/O for experiment tracks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any
import uuid

from omegaconf import OmegaConf
import pandas as pd


def create_run_dir(
    root_dir: Path,
    track: str,
    target: str,
    campaign_id: str | None = None,
    run_name: str | None = None,
) -> Path:
    """Create and return canonical run directory path."""
    run_id = run_name or _default_run_id(track=track, target=target)
    campaign = _normalize_campaign_id(campaign_id)
    run_dir = root_dir / campaign / target.lower() / track / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_resolved_config(
    cfg: Any, run_dir: Path, filename: str = "config.resolved.yaml"
) -> None:
    """Persist resolved runtime configuration as YAML."""
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    (run_dir / filename).write_text(resolved_yaml)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Persist dictionary to JSON with stable indentation."""
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True))


def save_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Persist dataframe as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def collect_run_metadata(random_seed: int) -> dict[str, Any]:
    """Collect minimal reproducibility metadata for the run."""
    return {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "random_seed": random_seed,
        "git_commit": _git_commit_or_none(),
    }


def _default_run_id(track: str, target: str) -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}_{track}_{target.lower()}_{suffix}"


def _normalize_campaign_id(campaign_id: str | None) -> str:
    cleaned = (campaign_id or "").strip()
    return cleaned or "default_campaign"


def _git_commit_or_none() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None
