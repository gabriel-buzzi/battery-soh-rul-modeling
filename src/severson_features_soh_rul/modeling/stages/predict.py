"""Prediction stage from persisted final model artifact."""

from __future__ import annotations

import json
from typing import Any

import joblib

from severson_features_soh_rul.modeling.artifacts.resolver import (
    resolve_required_file,
    resolve_unique_stage_dir,
)
from severson_features_soh_rul.modeling.artifacts.writer import (
    prepare_stage_dir,
    write_csv_atomic,
    write_resolved_config,
    write_run_info,
)
from severson_features_soh_rul.modeling.core.conformal import (
    predict_with_intervals,
)
from severson_features_soh_rul.modeling.stages.common import (
    build_prediction_dataframe,
    prepare_runtime_context,
)


def run_stage(cfg: Any) -> dict[str, Any]:
    """Execute predict stage."""
    print("[predict] running")
    base_context = prepare_runtime_context(cfg=cfg, stage="predict")

    selected_k: int | None = None
    if base_context.feature_cfg.selection_mode == "topk":
        topk_stage_dir = resolve_unique_stage_dir(
            artifacts_root=base_context.artifacts_cfg.root_dir,
            stage="topk_sweep",
            match_fields={
                "target": base_context.target,
                "feature_hash": base_context.feature_hash,
                "split_seed": base_context.split_cfg.seed,
                "model_name": base_context.model_cfg.name,
                "weighting_strategy": base_context.weighting_cfg.strategy,
            },
            require_exact_match=base_context.artifacts_cfg.require_exact_match,
        )
        topk_selection = json.loads(
            resolve_required_file(
                stage_dir=topk_stage_dir,
                file_name="topk_selection.json",
                stage="topk_sweep",
            ).read_text()
        )
        selected_k = int(topk_selection["selected_k"])

    context = prepare_runtime_context(
        cfg=cfg,
        stage="predict",
        k_selected=selected_k,
    )
    stage_dir, skipped = prepare_stage_dir(
        root_dir=context.artifacts_cfg.root_dir,
        run_key=context.run_key,
        stage="predict",
        required_files=[
            "predictions_test.csv",
            "config.resolved.yaml",
            "run_info.json",
        ],
        overwrite=context.artifacts_cfg.overwrite,
    )
    if skipped:
        return {
            "stage": "predict",
            "status": "skipped",
            "stage_dir": str(stage_dir),
            "run_key": context.run_key,
        }

    fit_stage_dir = resolve_unique_stage_dir(
        artifacts_root=context.artifacts_cfg.root_dir,
        stage="fit_final_model",
        match_fields={
            "target": context.target,
            "feature_hash": context.feature_hash,
            "split_seed": context.split_cfg.seed,
            "model_name": context.model_cfg.name,
            "weighting_strategy": context.weighting_cfg.strategy,
            "k_selected": selected_k,
        },
        require_exact_match=context.artifacts_cfg.require_exact_match,
    )
    model_path = resolve_required_file(
        stage_dir=fit_stage_dir,
        file_name="model.best.joblib",
        stage="fit_final_model",
    )
    selected_features_payload = json.loads(
        resolve_required_file(
            stage_dir=fit_stage_dir,
            file_name="selected_features.json",
            stage="fit_final_model",
        ).read_text()
    )
    selected_features = [
        str(value) for value in selected_features_payload["selected_features"]
    ]

    prediction_split = str(cfg.predict.get("split", "test")).lower().strip()
    if prediction_split == "test":
        prediction_df = context.test_df
    elif prediction_split == "train":
        prediction_df = context.train_df
    elif prediction_split == "all":
        prediction_df = context.features_df
    else:
        raise ValueError(
            "Unsupported predict.split='{}'. Supported: "
            "['test', 'train', 'all']".format(prediction_split)
        )

    model_bundle = joblib.load(model_path)
    y_pred, y_lo, y_hi = predict_with_intervals(
        model_bundle=model_bundle,
        X=prediction_df[selected_features],
    )
    output_df = build_prediction_dataframe(
        base_df=prediction_df,
        y_true=prediction_df[context.target],
        y_pred=y_pred,
        y_pred_lo=y_lo,
        y_pred_hi=y_hi,
        target=context.target,
        feature_columns=selected_features,
        split_seed=context.split_cfg.seed,
        stage=("test" if prediction_split == "test" else prediction_split),
    )

    write_resolved_config(cfg=context.cfg, stage_dir=stage_dir)
    write_run_info(
        stage_dir=stage_dir,
        run_key=context.run_key,
        context={
            **context.stage_context,
            "run_key_components": context.run_key_components,
            "fit_stage_dir": str(fit_stage_dir),
            "predict_split": prediction_split,
        },
    )
    write_csv_atomic(
        output_path=stage_dir / "predictions_test.csv", df=output_df
    )

    return {
        "stage": "predict",
        "status": "ok",
        "stage_dir": str(stage_dir),
        "run_key": context.run_key,
        "n_predictions": int(output_df.shape[0]),
    }
