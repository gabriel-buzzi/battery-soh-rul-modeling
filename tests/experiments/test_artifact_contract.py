"""Integration-style checks for artifact contract utilities."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.artifact_contract import (
    validate_optimization_cache_dir,
    validate_track_run_dir,
)


def _write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        writer.writerows(rows)


class TestArtifactContract(unittest.TestCase):
    """Validate artifact contract checks on synthetic directories."""

    def test_uncertainty_artifacts_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            _write_csv(
                run_dir / "predictions_repeated.csv",
                headers=["seed", "cell", "cycle", "y_true", "y_pred"],
                rows=[[1, "c1", 1, 10.0, 9.8]],
            )
            _write_csv(
                run_dir / "uncertainty_by_region.csv",
                headers=["region", "rmse_mean_prediction"],
                rows=[["near_eol", 1.2]],
            )
            (run_dir / "uncertainty_summary.json").write_text(json.dumps({}))

            errors = validate_track_run_dir(run_dir=run_dir, track="uncertainty")
            self.assertEqual(errors, [])

    def test_feature_analysis_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            _write_csv(
                run_dir / "feature_ranking_permutation.csv",
                headers=["feature", "permutation_rmse_increase_mean"],
                rows=[["V_mean", 0.1]],
            )
            # intentionally missing other required files
            errors = validate_track_run_dir(
                run_dir=run_dir, track="full_cycle_feature_analysis"
            )
            self.assertTrue(any("Missing required file" in error for error in errors))

    def test_cache_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            for file_name in [
                "best_params.json",
                "optimization_history.csv",
                "best_fold_metrics.csv",
                "best_aggregate_metrics.json",
            ]:
                (cache_dir / file_name).write_text("{}")

            errors = validate_optimization_cache_dir(cache_dir=cache_dir)
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
