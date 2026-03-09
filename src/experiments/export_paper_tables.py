"""Export manuscript-ready tables by merging track artifacts."""

from __future__ import annotations

from pathlib import Path
import time

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pandas as pd

from src.experiments.io import (
    save_dataframe_csv,
    save_dataframe_json,
    save_json,
)


def _latest_run_dir(track_root: Path) -> Path | None:
    if not track_root.exists():
        return None
    candidates = [path for path in track_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@hydra.main(
    version_base=None,
    config_path="../conf/experiments",
    config_name="base",
)
def export_paper_tables(cfg: DictConfig) -> None:
    """Aggregate latest track outputs into manuscript-ready tables."""
    artifacts_root = Path(to_absolute_path(cfg.artifacts.root_dir))
    out_root = artifacts_root.parent / "paper_tables"
    export_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = out_root / export_id
    out_dir.mkdir(parents=True, exist_ok=False)

    full_cycle_dir = _latest_run_dir(
        artifacts_root / "full_cycle_feature_analysis"
    )
    charge_dir = _latest_run_dir(
        artifacts_root / "charge_only_feature_analysis"
    )
    uncertainty_dir = _latest_run_dir(artifacts_root / "uncertainty")
    diagnostics_dir = _latest_run_dir(artifacts_root / "diagnostics")
    robustness_dir = _latest_run_dir(artifacts_root / "protocol_robustness")

    # Main comparison table (full-cycle + charge-only top-k sweep best rows).
    main_rows = []
    for label, run_dir in [
        ("full_cycle", full_cycle_dir),
        ("charge_only", charge_dir),
    ]:
        if run_dir is None:
            continue
        topk_df = _safe_read_csv(run_dir / "topk_sweep_metrics.csv")
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
        ranking_df = _safe_read_csv(
            run_dir / "feature_ranking_permutation.csv"
        )
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
        _safe_read_csv(uncertainty_dir / "uncertainty_by_region.csv")
        if uncertainty_dir is not None
        else pd.DataFrame()
    )

    # Robustness table.
    robustness_df = (
        _safe_read_csv(robustness_dir / "protocol_family_results.csv")
        if robustness_dir is not None
        else pd.DataFrame()
    )

    save_dataframe_csv(table_main_df, out_dir / "table_main_comparison.csv")
    save_dataframe_json(table_main_df, out_dir / "table_main_comparison.json")
    save_dataframe_csv(
        table_feature_df, out_dir / "table_feature_analysis.csv"
    )
    save_dataframe_json(
        table_feature_df, out_dir / "table_feature_analysis.json"
    )
    save_dataframe_csv(uncertainty_df, out_dir / "table_uncertainty.csv")
    save_dataframe_json(uncertainty_df, out_dir / "table_uncertainty.json")
    save_dataframe_csv(robustness_df, out_dir / "table_robustness.csv")
    save_dataframe_json(robustness_df, out_dir / "table_robustness.json")

    export_summary = {
        "export_id": export_id,
        "artifacts_root": str(artifacts_root),
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
