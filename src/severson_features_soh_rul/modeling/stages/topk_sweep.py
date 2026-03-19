"""Top-k sweep stage using ranking composite order."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from sklearn.model_selection import GroupKFold

from severson_features_soh_rul.modeling.artifacts.resolver import (
    resolve_required_file,
    resolve_unique_stage_dir,
)
from severson_features_soh_rul.modeling.artifacts.writer import (
    prepare_stage_dir,
    write_csv_atomic,
    write_json_atomic,
    write_resolved_config,
    write_run_info,
)
from severson_features_soh_rul.modeling.core.conformal import (
    fit_conformal_model,
    predict_with_intervals,
)
from severson_features_soh_rul.modeling.core.models import build_model
from severson_features_soh_rul.modeling.core.weighting import (
    build_sample_weights,
)
from severson_features_soh_rul.modeling.metrics.regression import rmse
from severson_features_soh_rul.modeling.stages.rank import (
    run_stage as run_rank,
)
from severson_features_soh_rul.modeling.stages.common import (
    prepare_runtime_context,
)


def run_stage(cfg: Any) -> dict[str, Any]:
    """Execute top-k sweep stage."""
    rank_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    rank_cfg.stage = "rank"
    rank_cfg.artifacts.overwrite = True
    rank_result = run_rank(rank_cfg)

    context = prepare_runtime_context(cfg=cfg, stage="topk_sweep")
    stage_dir, _ = prepare_stage_dir(
        root_dir=context.artifacts_cfg.root_dir,
        run_key=context.run_key,
        stage="topk_sweep",
        required_files=[
            "topk_sweep_cv.csv",
            "topk_selection.json",
            "config.resolved.yaml",
            "run_info.json",
        ],
        overwrite=True,
    )

    optimize_stage_dir = resolve_unique_stage_dir(
        artifacts_root=context.artifacts_cfg.root_dir,
        stage="optimize",
        match_fields={
            "target": context.target,
            "feature_hash": context.feature_hash,
            "split_seed": context.split_cfg.seed,
            "model_name": context.model_cfg.name,
            "weighting_strategy": context.weighting_cfg.strategy,
        },
        require_exact_match=context.artifacts_cfg.require_exact_match,
    )
    rank_stage_dir = resolve_unique_stage_dir(
        artifacts_root=context.artifacts_cfg.root_dir,
        stage="rank",
        match_fields={
            "target": context.target,
            "feature_hash": context.feature_hash,
            "split_seed": context.split_cfg.seed,
            "model_name": context.model_cfg.name,
            "weighting_strategy": context.weighting_cfg.strategy,
        },
        require_exact_match=context.artifacts_cfg.require_exact_match,
    )

    best_params = json.loads(
        resolve_required_file(
            stage_dir=optimize_stage_dir,
            file_name="best_params.json",
            stage="optimize",
        ).read_text()
    )
    ranking_composite_df = pd.read_csv(
        resolve_required_file(
            stage_dir=rank_stage_dir,
            file_name="ranking_composite.csv",
            stage="rank",
        )
    )
    ranked_features = ranking_composite_df["feature"].astype(str).tolist()

    total_features = len(ranked_features)
    requested_k = [
        k for k in context.topk_cfg.k_values if 1 <= k <= total_features
    ]
    k_values = sorted(set([*requested_k, total_features]))
    if not k_values:
        raise ValueError(
            "topk_sweep requires features.topk.k_values with values within "
            "[1, n_features]."
        )

    X_train = context.train_df[context.feature_cfg.columns]
    y_train = context.train_df[context.target]
    groups_train = context.train_df["cell"].astype(str)

    sweep_rows: list[dict[str, Any]] = []
    all_features: dict[str, Any] | None = None

    for k in k_values:
        selected_features = ranked_features[:k]
        fold_metrics = _evaluate_k(
            train_df=context.train_df,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            selected_features=selected_features,
            best_params=best_params,
            context=context,
        )
        row = {
            "k": int(k),
            "selected_features": ",".join(selected_features),
            **fold_metrics,
        }
        sweep_rows.append(row)
        if k == total_features:
            all_features = row

    if all_features is None:
        raise RuntimeError(
            "All features baseline row was not computed in topk_sweep."
        )

    sweep_df = pd.DataFrame(sweep_rows).sort_values("k").reset_index(drop=True)
    rmse_threshold = float(all_features["rmse_mean"]) * (
        1.0 + context.topk_cfg.tau_rmse
    )
    width_threshold = float(all_features["interval_width_mean"]) * (
        1.0 + context.topk_cfg.tau_width
    )
    sweep_df["is_feasible"] = (sweep_df["rmse_mean"] <= rmse_threshold) & (
        sweep_df["interval_width_mean"] <= width_threshold
    )

    selection, selection_mode = select_k_from_sweep_df(sweep_df=sweep_df)

    selected_k = int(selection["k"])
    selected_features = ranked_features[:selected_k]
    selection_payload = {
        "selected_k": selected_k,
        "selected_features": selected_features,
        "selection_mode": selection_mode,
        "tau_rmse": context.topk_cfg.tau_rmse,
        "tau_width": context.topk_cfg.tau_width,
        "all_features_rmse_mean": float(all_features["rmse_mean"]),
        "all_features_interval_width_mean": float(
            all_features["interval_width_mean"]
        ),
        "rmse_threshold": rmse_threshold,
        "width_threshold": width_threshold,
    }

    write_resolved_config(cfg=context.cfg, stage_dir=stage_dir)
    write_run_info(
        stage_dir=stage_dir,
        run_key=context.run_key,
        context={
            **context.stage_context,
            "run_key_components": context.run_key_components,
            "optimize_stage_dir": str(optimize_stage_dir),
            "rank_stage_dir": str(rank_stage_dir),
            "rank_refresh_status": rank_result.get("status"),
        },
    )
    write_csv_atomic(output_path=stage_dir / "topk_sweep_cv.csv", df=sweep_df)
    write_json_atomic(
        output_path=stage_dir / "topk_selection.json",
        payload=selection_payload,
    )

    return {
        "stage": "topk_sweep",
        "status": "ok",
        "stage_dir": str(stage_dir),
        "run_key": context.run_key,
        "selected_k": selected_k,
        "rank": rank_result,
    }


def select_k_from_sweep_df(
    sweep_df: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """Select top-k row using feasible-set and lexicographic fallback."""
    feasible_df = sweep_df[sweep_df["is_feasible"]].copy()
    if not feasible_df.empty:
        return feasible_df.sort_values("k", ascending=True).iloc[
            0
        ], "smallest_feasible"
    return (
        sweep_df.sort_values(
            by=["rmse_mean", "interval_width_mean", "k"],
            ascending=[True, True, True],
        ).iloc[0],
        "lexicographic_fallback",
    )


def _evaluate_k(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    selected_features: list[str],
    best_params: dict[str, Any],
    context: Any,
) -> dict[str, float]:
    """Evaluate one top-k subset with grouped CV."""
    gkf = GroupKFold(n_splits=context.optimize_cfg.cv_folds)
    rmse_values: list[float] = []
    width_values: list[float] = []

    for fold_id, (train_idx, val_idx) in enumerate(
        gkf.split(X=X_train, y=y_train, groups=groups_train),
        start=1,
    ):
        X_tr = X_train.iloc[train_idx][selected_features]
        X_val = X_train.iloc[val_idx][selected_features]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        groups_tr = groups_train.iloc[train_idx]

        fold_weights = build_sample_weights(
            y_train=y_tr,
            weighting_cfg=context.weighting_cfg,
            reference_series=train_df.iloc[train_idx]["RUL"],
        )
        base_model = build_model(
            model_name=context.model_cfg.name,
            model_params=best_params,
            random_seed=context.model_cfg.random_seed + fold_id,
            n_jobs=context.model_cfg.n_jobs,
        )
        model_bundle = fit_conformal_model(
            base_model=base_model,
            X_train=X_tr,
            y_train=y_tr,
            groups_train=groups_tr,
            conformal_enabled=context.conformal_cfg.enabled,
            alpha=context.conformal_cfg.alpha,
            calibration_proportion=context.conformal_cfg.calibration_proportion,
            random_seed=context.model_cfg.random_seed + fold_id,
            sample_weight=fold_weights,
        )
        y_pred, y_lo, y_hi = predict_with_intervals(
            model_bundle=model_bundle,
            X=X_val,
        )

        rmse_values.append(rmse(y_true=y_val, y_pred=y_pred))
        width_values.append(float(np.nanmean(y_hi - y_lo)))

    return {
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_std": float(np.std(rmse_values, ddof=0)),
        "interval_width_mean": float(np.mean(width_values)),
        "interval_width_std": float(np.std(width_values, ddof=0)),
    }
