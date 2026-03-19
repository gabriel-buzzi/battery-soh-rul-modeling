"""Unit tests for stage dependency failure behavior."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd
import pytest

from severson_features_soh_rul.modeling.stages.fit_final_model import (
    run_stage as run_fit_final_model,
)
from severson_features_soh_rul.modeling.stages.predict import (
    run_stage as run_predict,
)


def _build_cfg(tmp_path: Path, features_path: Path) -> object:
    return OmegaConf.create(
        {
            "stage": "fit_final_model",
            "target": "SOH",
            "data": {
                "features_path": str(features_path),
                "protocol_column": "charge_policy",
            },
            "split": {
                "train_cells_proportion": 0.8,
                "seed": 42,
                "split_dir": str(tmp_path / "splits"),
                "force_recreate": False,
            },
            "features": {
                "id": None,
                "columns": ["V_mean", "I_mean"],
                "hash_mode": "order_invariant",
                "selection_mode": "base",
                "topk": {
                    "k_values": [1, 2],
                    "selection_rule": "smallest_feasible",
                    "constraints": {
                        "tau_rmse": 0.05,
                        "tau_width": 0.1,
                    },
                },
            },
            "model": {"name": "extratrees", "n_jobs": 1},
            "optimize": {
                "enabled": True,
                "n_trials": 2,
                "cv_folds": 2,
                "save_cv_trials": True,
                "objective": {"tau_gap": 0.05, "lambda_gap": 0.5},
                "search_space": {
                    "n_estimators": {
                        "type": "fixed",
                        "value": 10,
                    },
                    "max_depth": {"type": "fixed", "value": 3},
                    "min_samples_split": {
                        "type": "fixed",
                        "value": 2,
                    },
                    "min_samples_leaf": {
                        "type": "fixed",
                        "value": 1,
                    },
                    "max_features": {
                        "type": "fixed",
                        "value": "sqrt",
                    },
                },
            },
            "conformal": {
                "enabled": True,
                "alpha": 0.1,
                "calibration_proportion": 0.2,
            },
            "ranking": {
                "n_repeats": 1,
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
                "root_dir": str(tmp_path / "artifacts"),
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
    for cell_index in range(6):
        cell_id = f"c{cell_index}"
        for cycle in range(4):
            rows.append(
                {
                    "cell": cell_id,
                    "cycle": cycle,
                    "SOH": 1.0 - 0.01 * cycle,
                    "RUL": 100 - cycle,
                    "V_mean": float(cell_index) + cycle * 0.1,
                    "I_mean": float(cell_index) * 0.2 + cycle * 0.05,
                    "charge_policy": "p0" if cell_index < 3 else "p1",
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_fit_final_model_requires_optimize_artifact(tmp_path: Path) -> None:
    """fit_final_model should fail fast without optimize artifacts."""
    features_path = tmp_path / "features.parquet"
    _write_features(features_path)
    cfg = _build_cfg(tmp_path=tmp_path, features_path=features_path)
    with pytest.raises(FileNotFoundError):
        run_fit_final_model(cfg)


def test_predict_requires_fitted_model_artifact(tmp_path: Path) -> None:
    """Predict should fail fast without fitted model artifacts."""
    features_path = tmp_path / "features.parquet"
    _write_features(features_path)
    cfg = _build_cfg(tmp_path=tmp_path, features_path=features_path)
    cfg.stage = "predict"
    with pytest.raises(FileNotFoundError):
        run_predict(cfg)
