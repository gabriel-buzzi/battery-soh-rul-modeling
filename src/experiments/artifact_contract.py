"""Artifact contract checks for integration validation."""

from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_FILES = {
    "full_cycle_feature_analysis": [
        "feature_ranking_permutation.csv",
        "feature_ranking_intrinsic.csv",
        "topk_sweep_metrics.csv",
        "loo_metrics.csv",
        "no_temp_metrics.json",
    ],
    "charge_only_feature_analysis": [
        "feature_ranking_permutation.csv",
        "feature_ranking_intrinsic.csv",
        "topk_sweep_metrics.csv",
        "loo_metrics.csv",
        "no_temp_metrics.json",
    ],
    "uncertainty": [
        "predictions_repeated.csv",
        "uncertainty_by_region.csv",
        "uncertainty_summary.json",
    ],
    "diagnostics": [
        "error_cells_summary.csv",
        "diagnostics_summary.json",
    ],
    "protocol_robustness": [
        "protocol_family_results.csv",
        "protocol_robustness_summary.json",
    ],
}

REQUIRED_COLUMNS = {
    "feature_ranking_permutation.csv": [
        "feature",
        "permutation_rmse_increase_mean",
    ],
    "topk_sweep_metrics.csv": [
        "k",
        "val_rmse_mean",
        "relative_gap_mean",
        "val_rmse_delta_from_baseline",
    ],
    "loo_metrics.csv": [
        "dropped_feature",
        "val_rmse_mean",
    ],
    "predictions_repeated.csv": [
        "seed",
        "cell",
        "cycle",
        "y_true",
        "y_pred",
    ],
    "uncertainty_by_region.csv": [
        "region",
        "rmse_mean_prediction",
    ],
    "error_cells_summary.csv": [
        "cell",
        "rmse",
        "mae",
        "dominant_error_region",
    ],
    "protocol_family_results.csv": [
        "held_out_family",
        "rmse",
        "mae",
        "r2",
    ],
}

REQUIRED_CACHE_FILES = [
    "best_params.json",
    "optimization_history.csv",
    "best_fold_metrics.csv",
    "best_aggregate_metrics.json",
]


def validate_track_run_dir(run_dir: Path, track: str) -> list[str]:
    """Return a list of artifact contract violations for a track run dir."""
    errors: list[str] = []
    required_files = REQUIRED_FILES.get(track, [])

    for file_name in required_files:
        file_path = run_dir / file_name
        if not file_path.exists():
            errors.append(f"Missing required file: {file_name}")
            continue

        required_cols = REQUIRED_COLUMNS.get(file_name)
        if required_cols is None:
            continue
        if file_path.suffix.lower() != ".csv":
            continue
        with open(file_path, "r", newline="") as fp:
            reader = csv.reader(fp)
            header = next(reader, [])
        missing_cols = [col for col in required_cols if col not in header]
        if missing_cols:
            errors.append(f"Missing columns in {file_name}: {missing_cols}")

    return errors


def validate_optimization_cache_dir(cache_dir: Path) -> list[str]:
    """Return a list of cache artifact violations."""
    errors: list[str] = []
    for file_name in REQUIRED_CACHE_FILES:
        if not (cache_dir / file_name).exists():
            errors.append(f"Missing cache file: {file_name}")
    return errors
