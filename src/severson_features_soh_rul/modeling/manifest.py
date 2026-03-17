"""Campaign manifest utilities for revision evidence mapping."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.experiments.runtime_helpers import ARTIFACT_SCHEMA_VERSION

MANIFEST_SCHEMA_VERSION = "1.0.0"

TRACK_REVIEWER_IDS = {
    "full_cycle_feature_analysis": ["RQ1_FEATURE_SELECTION"],
    "charge_only_feature_analysis": ["RQ1_FEATURE_SELECTION"],
    "final_eval": ["RQ2_HELD_OUT_PERFORMANCE"],
    "uncertainty": ["RQ3_PREDICTION_UNCERTAINTY"],
    "diagnostics": ["RQ4_ERROR_ANALYSIS"],
    "protocol_robustness": ["RQ5_PROTOCOL_ROBUSTNESS"],
}

TRACK_TABLE_ARTIFACTS = {
    "full_cycle_feature_analysis": [
        ("TBL_FEATURE_RANKING", "ranking.permutation.csv"),
        ("TBL_FEATURE_TOPK_SWEEP", "sweep.topk.csv"),
    ],
    "charge_only_feature_analysis": [
        ("TBL_FEATURE_RANKING", "ranking.permutation.csv"),
        ("TBL_FEATURE_TOPK_SWEEP", "sweep.topk.csv"),
    ],
    "final_eval": [("TBL_MAIN_RESULTS", "table.main_metrics.csv")],
    "uncertainty": [
        ("TBL_UNCERTAINTY_BY_REGION", "uncertainty.by_region.csv")
    ],
    "diagnostics": [("TBL_DIAGNOSTIC_CELLS", "diagnostics.cells.csv")],
    "protocol_robustness": [
        ("TBL_PROTOCOL_ROBUSTNESS", "robustness.by_family.csv")
    ],
}


def append_run_to_manifest(
    artifacts_root: Path,
    campaign_id: str | None,
    run_dir: Path,
    track: str,
    target: str,
    feature_set_id: str,
    optimization_cache_key: str,
    split_signature: dict[str, Any],
    feature_signature: dict[str, Any],
    purpose: dict[str, Any],
    metrics_path: Path,
    summary_path: Path,
    predictions_path: Path | None,
    artifacts_index_path: Path | None,
) -> None:
    """Append one run record to campaign manifest and refresh evidence maps."""
    normalized_campaign_id = _normalize_campaign_id(campaign_id)
    campaign_root = artifacts_root / normalized_campaign_id
    manifest_path = campaign_root / "manifest.json"
    manifest = _load_or_init_manifest(
        manifest_path=manifest_path,
        campaign_id=normalized_campaign_id,
    )

    run_rel_dir = _to_campaign_relative(
        path=run_dir, campaign_root=campaign_root
    )
    run_record = {
        "run_id": run_dir.name,
        "run_dir": run_rel_dir,
        "track": track,
        "target": target,
        "feature_set_id": feature_set_id,
        "optimization_cache_key": optimization_cache_key,
        "split_signature": split_signature,
        "feature_signature": feature_signature,
        "purpose_path": _to_campaign_relative(
            path=run_dir / "purpose.json",
            campaign_root=campaign_root,
        ),
        "metrics_path": _to_campaign_relative(
            path=metrics_path,
            campaign_root=campaign_root,
        ),
        "summary_path": _to_campaign_relative(
            path=summary_path,
            campaign_root=campaign_root,
        ),
        "predictions_path": (
            _to_campaign_relative(
                path=predictions_path, campaign_root=campaign_root
            )
            if predictions_path is not None
            else None
        ),
        "artifacts_index_path": (
            _to_campaign_relative(
                path=artifacts_index_path,
                campaign_root=campaign_root,
            )
            if artifacts_index_path is not None
            else None
        ),
        "lineage": purpose.get("lineage", {}),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    manifest["runs"] = [
        row
        for row in manifest.get("runs", [])
        if row.get("run_dir") != run_rel_dir
    ]
    manifest["runs"].append(run_record)

    _rebuild_maps(manifest=manifest)
    manifest["updated_utc"] = datetime.now(tz=timezone.utc).isoformat()
    _save_manifest(path=manifest_path, payload=manifest)


def _rebuild_maps(manifest: dict[str, Any]) -> None:
    reviewer_map: dict[str, list[dict[str, str]]] = {}
    table_map: dict[str, list[dict[str, str]]] = {}
    runs = manifest.get("runs", [])
    for run in runs:
        track = str(run.get("track"))
        run_dir = str(run.get("run_dir"))
        summary_path = str(run.get("summary_path"))

        for reviewer_id in TRACK_REVIEWER_IDS.get(track, []):
            reviewer_map.setdefault(reviewer_id, [])
            reviewer_map[reviewer_id].append(
                {"run_dir": run_dir, "artifact": summary_path}
            )

        for table_id, file_name in TRACK_TABLE_ARTIFACTS.get(track, []):
            table_map.setdefault(table_id, [])
            table_map[table_id].append(
                {"run_dir": run_dir, "artifact": f"{run_dir}/{file_name}"}
            )

    manifest["reviewer_question_map"] = reviewer_map
    manifest["table_figure_map"] = table_map


def _load_or_init_manifest(
    manifest_path: Path,
    campaign_id: str,
) -> dict[str, Any]:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    now = datetime.now(tz=timezone.utc).isoformat()
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "created_utc": now,
        "updated_utc": now,
        "runs": [],
        "reviewer_question_map": {},
        "table_figure_map": {},
    }


def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _to_campaign_relative(path: Path, campaign_root: Path) -> str:
    return str(path.relative_to(campaign_root))


def _normalize_campaign_id(value: str | None) -> str:
    cleaned = (value or "").strip()
    return cleaned or "default_campaign"
