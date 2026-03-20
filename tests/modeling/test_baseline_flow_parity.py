"""Integration test for baseline_flow parity with manual stage chain."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd
import pytest

from severson_features_soh_rul.modeling.stages.baseline_flow import (
    run_stage as run_baseline_flow,
)
from severson_features_soh_rul.modeling.stages.fit_final_model import (
    run_stage as run_fit_final_model,
)
from severson_features_soh_rul.modeling.stages.optimize import (
    run_stage as run_optimize,
)
from severson_features_soh_rul.modeling.stages.predict import (
    run_stage as run_predict,
)


def _build_cfg(tmp_path: Path, features_path: Path, root_name: str) -> object:
    return OmegaConf.create(
        {
            "stage": "baseline_flow",
            "target": "RUL",
            "data": {
                "features_path": str(features_path),
                "protocol_column": "charge_policy",
            },
            "split": {
                "train_cells_proportion": 0.75,
                "seed": 42,
                "split_dir": str(tmp_path / "splits_shared"),
                "force_recreate": False,
            },
            "features": {
                "id": "small_set",
                "columns": ["V_mean", "I_mean"],
            },
            "topk": {
                "k_values": [1, 2],
                "constraints": {"tau_rmse": 0.05, "tau_width": 0.1},
            },
            "model": {"name": "extratrees", "n_jobs": 1},
            "optimize": {
                "enabled": True,
                "n_trials": 1,
                "cv_folds": 2,
                "save_cv_trials": True,
                "objective": {"lambda_gap": 0.5},
                "search_space": {
                    "n_estimators": {"type": "fixed", "value": 20},
                    "criterion": {"type": "fixed", "value": "squared_error"},
                    "max_depth": {"type": "fixed", "value": 5},
                    "min_samples_split": {"type": "fixed", "value": 2},
                    "min_samples_leaf": {"type": "fixed", "value": 1},
                    "max_features": {"type": "fixed", "value": "sqrt"},
                },
            },
            "conformal": {
                "enabled": True,
                "alpha": 0.1,
                "calibration_proportion": 0.2,
            },
            "ranking": {
                "n_permutations": 2,
                "weights": {"rmse": 0.7, "uncertainty": 0.3},
                "rescale": {"clip_low_q": 0.05, "clip_high_q": 0.95},
            },
            "robustness": {"mode": "protocol_lopo"},
            "weighting": {
                "enabled": False,
                "strategy": "none",
                "n_bins": 10,
                "long_life_quantile": 0.75,
                "long_life_boost_factor": 2.0,
            },
            "predict": {"split": "test"},
            "artifacts": {
                "root_dir": str(tmp_path / root_name),
                "naming_key": "feature_id_or_hash",
                "run_key_fields": [
                    "target",
                    "feature_hash",
                    "split_seed",
                    "model_name",
                    "weighting_strategy",
                    "k_selected",
                ],
                "overwrite": False,
                "require_exact_match": True,
            },
        }
    )


def _write_features(path: Path) -> None:
    rows = []
    for cell_index in range(8):
        protocol = "p0" if cell_index < 4 else "p1"
        for cycle in range(5):
            rul = 120 - cycle - cell_index * 2
            rows.append(
                {
                    "cell": f"c{cell_index}",
                    "cycle": cycle,
                    "SOH": 1.0 - 0.01 * cycle,
                    "RUL": float(rul),
                    "V_mean": float(cell_index) + 0.1 * cycle,
                    "I_mean": float(cell_index) * 0.25 + 0.05 * cycle,
                    "charge_policy": protocol,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.mark.integration
def test_baseline_flow_matches_manual_chain(tmp_path: Path) -> None:
    """baseline_flow should match optimize->fit->predict artifacts."""
    pytest.importorskip("mapie")

    features_path = tmp_path / "features.parquet"
    _write_features(features_path)

    manual_cfg = _build_cfg(
        tmp_path, features_path, root_name="artifacts_manual"
    )
    flow_cfg = _build_cfg(tmp_path, features_path, root_name="artifacts_flow")

    run_optimize(manual_cfg)
    run_fit_final_model(manual_cfg)
    manual_predict_result = run_predict(manual_cfg)

    flow_result = run_baseline_flow(flow_cfg)

    manual_predictions = (
        pd.read_parquet(
            Path(manual_predict_result["stage_dir"])
            / "predictions_test.parquet"
        )
        .sort_values(["cell", "cycle"])
        .reset_index(drop=True)
    )
    flow_predictions = (
        pd.read_parquet(
            Path(flow_result["predict"]["stage_dir"])
            / "predictions_test.parquet"
        )
        .sort_values(["cell", "cycle"])
        .reset_index(drop=True)
    )

    pd.testing.assert_series_equal(
        manual_predictions["y_pred"],
        flow_predictions["y_pred"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        manual_predictions["y_pred_lo"],
        flow_predictions["y_pred_lo"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        manual_predictions["y_pred_hi"],
        flow_predictions["y_pred_hi"],
        check_names=False,
    )
