#!/usr/bin/env python3
"""Compute per-cell RMSE and R2 from revision-round held-out predictions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path().absolute().parent.parent
ARTIFACT_ROOT = ROOT / "results" / "results" / "experiments" / "revision_round1"
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
OUTPUT_DIR = ROOT / "paper" / "review_process" / "tables_revision_round1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with MANIFEST_PATH.open() as f:
    MANIFEST = json.load(f)


def resolve_run(track: str, target: str, feature_set_id: str) -> Path:
    """Return the artifact directory for a unique run selection."""
    runs = [
        run
        for run in MANIFEST["runs"]
        if run["track"] == track
        and run["target"] == target
        and run["feature_set_id"] == feature_set_id
    ]
    if len(runs) != 1:
        raise ValueError(
            f"Expected one run for {track=}, {target=}, {feature_set_id=}; "
            f"found {len(runs)}."
        )
    return ARTIFACT_ROOT / runs[0]["run_dir"]


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    """Compute RMSE and R2 for paired true and predicted values."""
    residual = y_true.astype(float) - y_pred.astype(float)
    rmse = float((residual.pow(2).mean()) ** 0.5)
    ss_res = float((residual.pow(2)).sum())
    centered = y_true.astype(float) - float(y_true.astype(float).mean())
    ss_tot = float((centered.pow(2)).sum())
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))
    return rmse, r2


def build_per_cell_table(target: str) -> pd.DataFrame:
    """Build per-cell metrics and mark the worst RMSE quartile cells."""
    pred_df = pd.read_csv(
        resolve_run("diagnostics", target, "full_all") / "predictions.csv"
    )
    rows: list[dict[str, float | str | bool | int]] = []
    for cell, cell_df in pred_df.groupby("cell", sort=True):
        rmse, r2 = compute_metrics(cell_df["y_true"], cell_df["y_pred"])
        rows.append(
            {
                "cell": str(cell),
                "n_samples": int(len(cell_df)),
                "rmse": rmse,
                "r2": r2,
            }
        )
    per_cell_df = pd.DataFrame(rows).sort_values("rmse", ascending=False)
    n_focus = math.ceil(len(per_cell_df) * 0.25)
    focus_cells = set(per_cell_df.head(n_focus)["cell"].tolist())
    per_cell_df["is_worst_quartile_cell"] = per_cell_df["cell"].isin(focus_cells)
    return per_cell_df.reset_index(drop=True)


def main() -> None:
    """Write per-cell metrics tables and a summary JSON for SOH and RUL."""
    summary: dict[str, dict[str, float | int | list[str]]] = {}
    for target in ["SOH", "RUL"]:
        per_cell_df = build_per_cell_table(target)
        out_path = OUTPUT_DIR / f"{target.lower()}_per_cell_metrics.csv"
        per_cell_df.to_csv(out_path, index=False)

        focus_df = per_cell_df[per_cell_df["is_worst_quartile_cell"]]
        keep_df = per_cell_df[~per_cell_df["is_worst_quartile_cell"]]
        summary[target] = {
            "n_cells": int(len(per_cell_df)),
            "worst_quartile_count": int(len(focus_df)),
            "worst_quartile_cells": focus_df["cell"].tolist(),
            "mean_rmse_worst_quartile": float(focus_df["rmse"].mean()),
            "mean_rmse_remaining": float(keep_df["rmse"].mean()),
        }

    with (OUTPUT_DIR / "per_cell_metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
