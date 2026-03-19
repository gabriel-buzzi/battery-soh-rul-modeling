"""Single-entrypoint stage dispatcher for the modeling pipeline."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from severson_features_soh_rul.modeling.config.schema import validate_stage
from severson_features_soh_rul.modeling.stages.all_stages import (
    run_stage as run_all_stages,
)
from severson_features_soh_rul.modeling.stages.baseline_flow import (
    run_stage as run_baseline_flow,
)
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
from severson_features_soh_rul.modeling.stages.robustness_protocol_lopo import (  # noqa: E501
    run_stage as run_robustness_protocol_lopo,
)
from severson_features_soh_rul.modeling.stages.topk_sweep import (
    run_stage as run_topk_sweep,
)


@hydra.main(
    version_base=None, config_path="../../../config", config_name="modeling"
)
def run_pipeline(cfg: DictConfig) -> None:
    """Dispatch execution to selected stage."""
    stage = validate_stage(str(cfg.stage))

    if stage == "all_stages":
        result = run_all_stages(cfg)
    elif stage == "optimize":
        result = run_optimize(cfg)
    elif stage == "permutation_importance":
        result = run_permutation_importance(cfg)
    elif stage == "rank":
        result = run_rank(cfg)
    elif stage == "topk_sweep":
        result = run_topk_sweep(cfg)
    elif stage == "fit_final_model":
        result = run_fit_final_model(cfg)
    elif stage == "predict":
        result = run_predict(cfg)
    elif stage == "robustness_protocol_lopo":
        result = run_robustness_protocol_lopo(cfg)
    elif stage == "baseline_flow":
        result = run_baseline_flow(cfg)
    else:
        raise ValueError(
            "Unsupported stage='{}'. Supported: {}".format(
                stage,
                [
                    "all_stages",
                    "optimize",
                    "permutation_importance",
                    "rank",
                    "topk_sweep",
                    "fit_final_model",
                    "predict",
                    "robustness_protocol_lopo",
                    "baseline_flow",
                ],
            )
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_pipeline()
