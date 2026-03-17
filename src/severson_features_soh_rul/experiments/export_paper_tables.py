"""Export manuscript-ready tables by merging track artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import time

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pandas as pd

from src.experiments.io import (
    save_dataframe_csv,
    save_json,
)


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Campaign manifest not found: {manifest_path}"
        )
    return json.loads(manifest_path.read_text())


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _runs_for_target(manifest: dict, target: str) -> list[dict]:
    return [
        run
        for run in manifest.get("runs", [])
        if str(run.get("target")) == str(target)
    ]


def _latest_run_for_track(runs: list[dict], track: str) -> dict | None:
    candidates = [run for run in runs if str(run.get("track")) == track]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: str(row.get("created_utc", "")))[
        -1
    ]


def _require_table_mapping(manifest: dict, table_id: str, target: str) -> None:
    entries = manifest.get("table_figure_map", {}).get(table_id, [])
    target_prefix = f"{str(target).lower()}/"
    has_target_entry = any(
        str(entry.get("run_dir", "")).startswith(target_prefix)
        for entry in entries
    )
    if not has_target_entry:
        raise ValueError(
            "Manifest incomplete: missing table mapping for "
            f"{table_id} and target={target}."
        )


def _require_reviewer_mapping(
    manifest: dict, reviewer_id: str, target: str
) -> None:
    entries = manifest.get("reviewer_question_map", {}).get(reviewer_id, [])
    target_prefix = f"{str(target).lower()}/"
    has_target_entry = any(
        str(entry.get("run_dir", "")).startswith(target_prefix)
        for entry in entries
    )
    if not has_target_entry:
        raise ValueError(
            "Manifest incomplete: missing reviewer mapping for "
            f"{reviewer_id} and target={target}."
        )


@hydra.main(
    version_base=None,
    config_path="../conf/experiments",
    config_name="base",
)
def export_paper_tables(cfg: DictConfig) -> None:
    """Aggregate manuscript-ready tables from campaign manifest only."""
    artifacts_root = Path(to_absolute_path(cfg.artifacts.root_dir))
    out_root = artifacts_root.parent / "paper_tables"
    export_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = out_root / export_id
    out_dir.mkdir(parents=True, exist_ok=False)

    campaign_id = (
        str(cfg.artifacts.campaign_id or "").strip() or "default_campaign"
    )
    campaign_root = artifacts_root / campaign_id
    manifest_path = campaign_root / "manifest.json"
    manifest = _load_manifest(manifest_path=manifest_path)

    target_runs = _runs_for_target(manifest=manifest, target=str(cfg.target))
    if not target_runs:
        raise ValueError(
            "Manifest incomplete: no runs found for target="
            f"{cfg.target} in campaign={campaign_id}."
        )

    for table_id in [
        "TBL_FEATURE_TOPK_SWEEP",
        "TBL_FEATURE_RANKING",
        "TBL_UNCERTAINTY_BY_REGION",
        "TBL_PROTOCOL_ROBUSTNESS",
    ]:
        _require_table_mapping(
            manifest=manifest,
            table_id=table_id,
            target=str(cfg.target),
        )
    for reviewer_id in [
        "RQ1_FEATURE_SELECTION",
        "RQ2_HELD_OUT_PERFORMANCE",
        "RQ3_PREDICTION_UNCERTAINTY",
        "RQ4_ERROR_ANALYSIS",
        "RQ5_PROTOCOL_ROBUSTNESS",
    ]:
        _require_reviewer_mapping(
            manifest=manifest,
            reviewer_id=reviewer_id,
            target=str(cfg.target),
        )

    full_cycle_run = _latest_run_for_track(
        runs=target_runs,
        track="full_cycle_feature_analysis",
    )
    charge_run = _latest_run_for_track(
        runs=target_runs,
        track="charge_only_feature_analysis",
    )
    uncertainty_run = _latest_run_for_track(
        runs=target_runs, track="uncertainty"
    )
    diagnostics_run = _latest_run_for_track(
        runs=target_runs, track="diagnostics"
    )
    robustness_run = _latest_run_for_track(
        runs=target_runs,
        track="protocol_robustness",
    )

    if full_cycle_run is None or charge_run is None:
        raise ValueError(
            "Manifest incomplete: feature-analysis runs are required for "
            "both full_cycle_feature_analysis and charge_only_feature_analysis."
        )
    if uncertainty_run is None:
        raise ValueError("Manifest incomplete: uncertainty run is required.")
    if robustness_run is None:
        raise ValueError(
            "Manifest incomplete: protocol_robustness run is required."
        )

    full_cycle_dir = campaign_root / str(full_cycle_run["run_dir"])
    charge_dir = campaign_root / str(charge_run["run_dir"])
    uncertainty_dir = campaign_root / str(uncertainty_run["run_dir"])
    diagnostics_dir = (
        campaign_root / str(diagnostics_run["run_dir"])
        if diagnostics_run is not None
        else None
    )
    robustness_dir = campaign_root / str(robustness_run["run_dir"])

    # Main comparison table (full-cycle + charge-only top-k sweep best rows).
    main_rows = []
    for label, run_dir in [
        ("full_cycle", full_cycle_dir),
        ("charge_only", charge_dir),
    ]:
        if run_dir is None:
            continue
        topk_df = _safe_read_csv(run_dir / "sweep.topk.csv")
        if topk_df.empty:
            continue
        best_row = topk_df.sort_values("val_rmse_mean").iloc[0].to_dict()
        best_row["track"] = label
        main_rows.append(best_row)
    table_main_df = pd.DataFrame(main_rows)

    # Feature-analysis table.
    feature_rows = []
    for label, run_dir in [
        ("full_cycle", full_cycle_dir),
        ("charge_only", charge_dir),
    ]:
        if run_dir is None:
            continue
        ranking_df = _safe_read_csv(run_dir / "ranking.permutation.csv")
        if ranking_df.empty:
            continue
        top5_df = ranking_df.head(5).copy()
        top5_df["track"] = label
        feature_rows.append(top5_df)
    table_feature_df = (
        pd.concat(feature_rows, ignore_index=True)
        if feature_rows
        else pd.DataFrame()
    )

    # Uncertainty table.
    uncertainty_df = (
        _safe_read_csv(uncertainty_dir / "uncertainty.by_region.csv")
        if uncertainty_dir is not None
        else pd.DataFrame()
    )

    # Robustness table.
    robustness_df = (
        _safe_read_csv(robustness_dir / "robustness.by_family.csv")
        if robustness_dir is not None
        else pd.DataFrame()
    )

    save_dataframe_csv(table_main_df, out_dir / "table_main_comparison.csv")
    save_dataframe_csv(
        table_feature_df, out_dir / "table_feature_analysis.csv"
    )
    save_dataframe_csv(uncertainty_df, out_dir / "table_uncertainty.csv")
    save_dataframe_csv(robustness_df, out_dir / "table_robustness.csv")

    export_summary = {
        "export_id": export_id,
        "artifacts_root": str(artifacts_root),
        "campaign_id": campaign_id,
        "target": str(cfg.target),
        "manifest_path": str(manifest_path),
        "source_runs": {
            "full_cycle_feature_analysis": str(full_cycle_dir)
            if full_cycle_dir
            else None,
            "charge_only_feature_analysis": str(charge_dir)
            if charge_dir
            else None,
            "uncertainty": str(uncertainty_dir) if uncertainty_dir else None,
            "diagnostics": str(diagnostics_dir) if diagnostics_dir else None,
            "protocol_robustness": str(robustness_dir)
            if robustness_dir
            else None,
        },
    }
    save_json(export_summary, out_dir / "export_summary.json")


if __name__ == "__main__":
    export_paper_tables()
