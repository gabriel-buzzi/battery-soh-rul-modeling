"""Shared runtime helpers for experiment track execution."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from omegaconf import DictConfig
import pandas as pd

logger = logging.getLogger(__name__)
ARTIFACT_SCHEMA_VERSION = "1.0.0"


def sha256_of_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_split_signature(
    random_seed: int,
    train_cells: list[str],
    test_cells: list[str],
) -> dict[str, Any]:
    return {
        "split_seed": int(random_seed),
        "n_train_cells": int(len(train_cells)),
        "n_test_cells": int(len(test_cells)),
        "train_cells_hash": sha256_of_string(
            json.dumps(list(train_cells), sort_keys=False)
        ),
        "test_cells_hash": sha256_of_string(
            json.dumps(list(test_cells), sort_keys=False)
        ),
    }


def build_feature_signature(feature_columns: list[str]) -> dict[str, Any]:
    return {
        "n_features": int(len(feature_columns)),
        "feature_columns_hash": sha256_of_string(
            json.dumps(list(feature_columns), sort_keys=False)
        ),
        "feature_columns": list(feature_columns),
    }


def target_unit(target: str) -> str:
    units = {
        "SOH": "percent",
        "RUL": "cycles",
    }
    return units.get(target, "unknown")


def target_formula(target: str) -> str:
    formulas = {
        "SOH": "SOH_n = Q_n / Q_rated",
        "RUL": "RUL_n = EoL_cycle - cycle_n",
    }
    return formulas.get(target, "N/A")


def track_family(track_name: str) -> str:
    if track_name == "final_eval":
        return "baseline_family"
    if track_name in {
        "full_cycle_feature_analysis",
        "charge_only_feature_analysis",
    }:
        return "feature_analysis_family"
    if track_name == "uncertainty":
        return "uncertainty_family"
    if track_name == "diagnostics":
        return "diagnostics_family"
    if track_name == "protocol_robustness":
        return "protocol_robustness_family"
    return "other"


def _normalize_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _infer_run_role(track_name: str, feature_set_id: str) -> str:
    if track_name == "final_eval":
        return f"final_eval_{feature_set_id}"
    if track_name == "full_cycle_feature_analysis":
        return "full_cycle_cv_feature_analysis"
    if track_name == "charge_only_feature_analysis":
        return "charge_only_cv_feature_analysis"
    if track_name == "uncertainty":
        return "uncertainty_analysis"
    if track_name == "diagnostics":
        return "diagnostics_analysis"
    if track_name == "protocol_robustness":
        return "protocol_family_holdout"
    return "unclassified"


def build_run_purpose(
    cfg: DictConfig,
    track_name: str,
    feature_columns: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feature_set_id = str(cfg.features.set_id)
    artifacts_cfg = cfg.get("artifacts", {})
    lineage_cfg = artifacts_cfg.get("lineage", {})
    explicit_role = _normalize_optional(artifacts_cfg.get("run_role"))
    role = explicit_role or _infer_run_role(
        track_name=track_name,
        feature_set_id=feature_set_id,
    )
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "track": track_name,
        "track_family": track_family(track_name),
        "target": str(cfg.target),
        "feature_set_id": feature_set_id,
        "n_features": len(feature_columns),
        "run_role": role,
        "paper_section": _normalize_optional(
            artifacts_cfg.get("paper_section")
        ),
        "comparison_group": _normalize_optional(
            artifacts_cfg.get("comparison_group")
        ),
        "notes": _normalize_optional(artifacts_cfg.get("notes")),
        "lineage": {
            "source_track": _normalize_optional(
                lineage_cfg.get("source_track")
            ),
            "source_run": _normalize_optional(lineage_cfg.get("source_run")),
            "source_artifact": _normalize_optional(
                lineage_cfg.get("source_artifact")
            ),
        },
        "requested_track": str(cfg.track),
    }
    if extra:
        payload["details"] = extra
    return payload


def select_k_with_heuristics(
    topk_sweep_df: pd.DataFrame,
    allowed_k_values: list[int],
    max_val_rmse_increase_pct: float = 0.10,
) -> int:
    """Select the smallest allowed k within a max validation-RMSE increase."""
    if not allowed_k_values:
        raise ValueError("allowed_k_values is empty.")

    baseline_rows = topk_sweep_df[
        topk_sweep_df["val_rmse_delta_from_baseline"].abs() < 1e-12
    ]
    if baseline_rows.empty:
        baseline_val_rmse = float(
            topk_sweep_df.sort_values("k", ascending=False).iloc[0][
                "val_rmse_mean"
            ]
        )
        logger.warning(
            "Baseline row not found in top-k sweep; using largest-k row as baseline (val_rmse=%.6f).",
            baseline_val_rmse,
        )
    else:
        baseline_val_rmse = float(baseline_rows.iloc[0]["val_rmse_mean"])

    max_allowed_val_rmse = baseline_val_rmse * (
        1.0 + float(max_val_rmse_increase_pct)
    )
    allowed_df = topk_sweep_df[
        topk_sweep_df["k"].isin(allowed_k_values)
    ].copy()
    eligible_df = allowed_df[
        allowed_df["val_rmse_mean"] <= max_allowed_val_rmse
    ].copy()

    if not eligible_df.empty:
        selected_k = int(
            eligible_df.sort_values("k", ascending=True).iloc[0]["k"]
        )
        logger.info(
            "Heuristic k-selection: baseline_val_rmse=%.6f threshold=%.6f eligible=%d selected_k=%d",
            baseline_val_rmse,
            max_allowed_val_rmse,
            int(eligible_df.shape[0]),
            selected_k,
        )
        return selected_k

    fallback_row = allowed_df.sort_values(
        by=["val_rmse_mean", "k"],
        ascending=[True, True],
    ).iloc[0]
    fallback_k = int(fallback_row["k"])
    logger.warning(
        "No k within %.1f%% baseline RMSE increase; fallback to best val_rmse k=%d (val_rmse=%.6f baseline=%.6f).",
        float(max_val_rmse_increase_pct) * 100.0,
        fallback_k,
        float(fallback_row["val_rmse_mean"]),
        baseline_val_rmse,
    )
    return fallback_k
