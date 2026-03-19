"""Full orchestration stage running all modeling stages."""

from __future__ import annotations

import logging
from typing import Any

from severson_features_soh_rul.modeling.stages.fit_final_model import (
    run_stage as run_fit_final_model,
)
from severson_features_soh_rul.modeling.stages.optimize import (
    run_stage as run_optimize,
)
from severson_features_soh_rul.modeling.stages.permutation_importance import (
    run_stage as run_permutation_importance,
)
from severson_features_soh_rul.modeling.stages.predict import (
    run_stage as run_predict,
)
from severson_features_soh_rul.modeling.stages.rank import (
    run_stage as run_rank,
)
from severson_features_soh_rul.modeling.stages.robustness_protocol_lopo import (
    run_stage as run_robustness_protocol_lopo,
)
from severson_features_soh_rul.modeling.stages.topk_sweep import (
    run_stage as run_topk_sweep,
)

LOGGER = logging.getLogger(__name__)


def run_stage(cfg: Any) -> dict[str, Any]:
    """Run optimize -> fit -> predict -> permutation -> rank -> topk -> lopo."""
    LOGGER.info("[all_stages] running")
    optimize_result = run_optimize(cfg)
    fit_result = run_fit_final_model(cfg)
    predict_result = run_predict(cfg)
    permutation_result = run_permutation_importance(cfg)
    rank_result = run_rank(cfg)
    topk_result = run_topk_sweep(cfg)
    robustness_result = run_robustness_protocol_lopo(cfg)
    return {
        "stage": "all_stages",
        "status": "ok",
        "optimize": optimize_result,
        "fit_final_model": fit_result,
        "predict": predict_result,
        "permutation_importance": permutation_result,
        "rank": rank_result,
        "topk_sweep": topk_result,
        "robustness_protocol_lopo": robustness_result,
    }
