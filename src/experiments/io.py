"""Run artifacts I/O for experiment tracks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from omegaconf import OmegaConf
import pandas as pd


def create_run_dir(
    root_dir: Path,
    track: str,
    target: str,
    run_name: str | None = None,
) -> Path:
    """Create and return canonical run directory path."""
    run_id = run_name or _default_run_id(track=track, target=target)
    run_dir = root_dir / track / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_resolved_config(cfg: Any, run_dir: Path) -> None:
    """Persist resolved runtime configuration as YAML."""
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    (run_dir / "resolved_config.yaml").write_text(resolved_yaml)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Persist dictionary to JSON with stable indentation."""
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True))


def save_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Persist dataframe as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_dataframe_json(df: pd.DataFrame, output_path: Path) -> None:
    """Persist dataframe as JSON records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(df.to_json(orient="records", indent=2))


def collect_run_metadata(random_seed: int) -> dict[str, Any]:
    """Collect minimal reproducibility metadata for the run."""
    return {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "random_seed": random_seed,
        "git_commit": _git_commit_or_none(),
    }


def _default_run_id(track: str, target: str) -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{track}_{target.lower()}"


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
