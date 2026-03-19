"""Baseline orchestration stage."""

from __future__ import annotations

from typing import Any

from severson_features_soh_rul.modeling.stages.fit_final_model import (
    run_stage as run_fit_final_model,
)
from severson_features_soh_rul.modeling.stages.optimize import (
    run_stage as run_optimize,
)
from severson_features_soh_rul.modeling.stages.predict import (
    run_stage as run_predict,
)


def run_stage(cfg: Any) -> dict[str, Any]:
    """Run optimize -> fit_final_model -> predict."""
    print("[baseline_flow] running")
    optimize_result = run_optimize(cfg)
    fit_result = run_fit_final_model(cfg)
    predict_result = run_predict(cfg)
    return {
        "stage": "baseline_flow",
        "status": "ok",
        "optimize": optimize_result,
        "fit_final_model": fit_result,
        "predict": predict_result,
    }
