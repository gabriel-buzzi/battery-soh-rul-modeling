"""Permutation-importance stage that saves shuffled predictions only."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from severson_features_soh_rul.modeling.artifacts.resolver import (
    resolve_required_file,
    resolve_unique_stage_dir,
)
from severson_features_soh_rul.modeling.artifacts.writer import (
    prepare_stage_dir,
    write_parquet_atomic,
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
from severson_features_soh_rul.modeling.stages.common import (
    build_prediction_dataframe,
    prepare_runtime_context,
)

LOGGER = logging.getLogger(__name__)


def run_stage(cfg: Any) -> dict[str, Any]:
    """Execute permutation_importance stage."""
    LOGGER.info("[permutation_importance] running")
    context = prepare_runtime_context(cfg=cfg, stage="permutation_importance")
    stage_dir, skipped = prepare_stage_dir(
        root_dir=context.artifacts_cfg.root_dir,
        run_key=context.run_key,
        stage="permutation_importance",
        required_files=[
            "predictions_permutation_importance.parquet",
            "config.resolved.yaml",
            "run_info.json",
        ],
        overwrite=context.artifacts_cfg.overwrite,
    )
    if skipped:
        return {
            "stage": "permutation_importance",
            "status": "skipped",
            "stage_dir": str(stage_dir),
            "run_key": context.run_key,
        }

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
    best_params_path = resolve_required_file(
        stage_dir=optimize_stage_dir,
        file_name="best_params.json",
        stage="optimize",
    )
    best_params = json.loads(best_params_path.read_text())

    X_train = context.train_df[context.feature_cfg.columns]
    y_train = context.train_df[context.target]
    groups_train = context.train_df["cell"].astype(str)

    gkf = GroupKFold(n_splits=context.optimize_cfg.cv_folds)
    fold_splits = list(gkf.split(X=X_train, y=y_train, groups=groups_train))
    prediction_rows: list[pd.DataFrame] = []
    total_permutations = (
        len(fold_splits)
        * len(context.feature_cfg.columns)
        * context.ranking_cfg.n_permutations
    )
    permutation_pbar = tqdm(
        total=total_permutations,
        desc="permutation_importance",
        unit="perm",
    )

    try:
        for fold_id, (train_idx, val_idx) in enumerate(fold_splits, start=1):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]
            groups_tr = groups_train.iloc[train_idx]

            fold_seed = context.model_cfg.random_seed + fold_id
            fold_weights = build_sample_weights(
                y_train=y_tr,
                weighting_cfg=context.weighting_cfg,
                reference_series=context.train_df.iloc[train_idx]["RUL"],
            )

            base_model = build_model(
                model_name=context.model_cfg.name,
                model_params=best_params,
                random_seed=fold_seed,
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
                random_seed=fold_seed,
                sample_weight=fold_weights,
            )

            for feature_idx, feature in enumerate(context.feature_cfg.columns):
                for permutation_idx in range(
                    context.ranking_cfg.n_permutations
                ):
                    X_val_perm = X_val.copy()
                    seed = _build_permutation_seed(
                        split_seed=context.split_cfg.seed,
                        fold_id=fold_id,
                        feature_idx=feature_idx,
                        feature_name=feature,
                        permutation_idx=permutation_idx,
                    )
                    rng = np.random.default_rng(seed)
                    X_val_perm.loc[:, feature] = rng.permutation(
                        X_val_perm[feature].to_numpy()
                    )
                    y_pred, y_lo, y_hi = predict_with_intervals(
                        model_bundle=model_bundle,
                        X=X_val_perm,
                    )
                    shuffled_predictions = build_prediction_dataframe(
                        base_df=context.train_df.iloc[val_idx],
                        y_true=y_val,
                        y_pred=y_pred,
                        y_pred_lo=y_lo,
                        y_pred_hi=y_hi,
                        target=context.target,
                        feature_columns=context.feature_cfg.columns,
                        split_seed=context.split_cfg.seed,
                        stage="permutation_importance_val",
                    )
                    shuffled_predictions["fold"] = fold_id
                    shuffled_predictions["feature"] = feature
                    shuffled_predictions["permutation"] = permutation_idx
                    prediction_rows.append(shuffled_predictions)
                    permutation_pbar.update(1)
    finally:
        permutation_pbar.close()

    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    write_resolved_config(cfg=context.cfg, stage_dir=stage_dir)
    write_run_info(
        stage_dir=stage_dir,
        run_key=context.run_key,
        context={
            **context.stage_context,
            "run_key_components": context.run_key_components,
            "n_permutations": context.ranking_cfg.n_permutations,
            "optimize_stage_dir": str(optimize_stage_dir),
        },
    )
    write_parquet_atomic(
        output_path=stage_dir / "predictions_permutation_importance.parquet",
        df=predictions_df,
    )

    return {
        "stage": "permutation_importance",
        "status": "ok",
        "stage_dir": str(stage_dir),
        "run_key": context.run_key,
    }


def _build_permutation_seed(
    split_seed: int,
    fold_id: int,
    feature_idx: int,
    feature_name: str,
    permutation_idx: int,
) -> int:
    """Build deterministic seed for one feature-permutation execution."""
    digest = hashlib.sha256(feature_name.encode("utf-8")).hexdigest()[:8]
    feature_hash = int(digest, 16)
    seed = (
        int(split_seed) * 1_000_000
        + int(fold_id) * 10_000
        + int(feature_idx) * 100
        + int(permutation_idx)
        + feature_hash
    )
    return int(seed % (2**32 - 1))
