"""Protocol robustness stage using strict leave-one-protocol-out evaluation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

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


def run_stage(cfg: Any) -> dict[str, Any]:
    """Execute strict protocol LOPO robustness stage."""
    context = prepare_runtime_context(
        cfg=cfg,
        stage="robustness_protocol_lopo",
        require_protocol_column=True,
    )
    stage_dir, skipped = prepare_stage_dir(
        root_dir=context.artifacts_cfg.root_dir,
        run_key=context.run_key,
        stage="robustness_protocol_lopo",
        required_files=[
            "predictions_protocol_lopo.csv",
            "config.resolved.yaml",
            "run_info.json",
        ],
        overwrite=context.artifacts_cfg.overwrite,
    )
    if skipped:
        return {
            "stage": "robustness_protocol_lopo",
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
    best_params = json.loads(
        resolve_required_file(
            stage_dir=optimize_stage_dir,
            file_name="best_params.json",
            stage="optimize",
        ).read_text()
    )

    protocol_column = context.robustness_cfg.protocol_column
    protocols = sorted(
        context.features_df[protocol_column].astype(str).unique().tolist()
    )
    output_rows: list[pd.DataFrame] = []

    for protocol_value in protocols:
        train_df = context.features_df[
            context.features_df[protocol_column].astype(str) != protocol_value
        ].copy()
        test_df = context.features_df[
            context.features_df[protocol_column].astype(str) == protocol_value
        ].copy()

        if train_df.empty or test_df.empty:
            continue

        selected_features = context.feature_cfg.columns
        X_train = train_df[selected_features]
        y_train = train_df[context.target]
        groups_train = train_df["cell"].astype(str)

        sample_weights = build_sample_weights(
            y_train=y_train,
            weighting_cfg=context.weighting_cfg,
            reference_series=train_df["RUL"],
        )
        base_model = build_model(
            model_name=context.model_cfg.name,
            model_params=best_params,
            random_seed=context.model_cfg.random_seed,
            n_jobs=context.model_cfg.n_jobs,
        )
        model_bundle = fit_conformal_model(
            base_model=base_model,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            conformal_enabled=context.conformal_cfg.enabled,
            alpha=context.conformal_cfg.alpha,
            calibration_proportion=context.conformal_cfg.calibration_proportion,
            random_seed=context.model_cfg.random_seed,
            sample_weight=sample_weights,
        )

        y_pred, y_lo, y_hi = predict_with_intervals(
            model_bundle=model_bundle,
            X=test_df[selected_features],
        )
        prediction_df = build_prediction_dataframe(
            base_df=test_df,
            y_true=test_df[context.target],
            y_pred=y_pred,
            y_pred_lo=y_lo,
            y_pred_hi=y_hi,
            target=context.target,
            feature_columns=selected_features,
            split_seed=context.split_cfg.seed,
            stage="protocol_lopo",
            held_out_protocol=protocol_value,
        )
        prediction_df["protocol_column"] = protocol_column
        prediction_df["n_train_cells"] = int(train_df["cell"].nunique())
        prediction_df["n_test_cells"] = int(test_df["cell"].nunique())
        output_rows.append(prediction_df)

    if not output_rows:
        raise RuntimeError(
            "robustness_protocol_lopo produced no predictions; check protocol "
            "column and data coverage."
        )

    output_df = pd.concat(output_rows, ignore_index=True)

    write_resolved_config(cfg=context.cfg, stage_dir=stage_dir)
    write_run_info(
        stage_dir=stage_dir,
        run_key=context.run_key,
        context={
            **context.stage_context,
            "run_key_components": context.run_key_components,
            "optimize_stage_dir": str(optimize_stage_dir),
            "protocol_column": protocol_column,
            "n_protocols": len(protocols),
        },
    )
    write_csv_atomic(
        output_path=stage_dir / "predictions_protocol_lopo.csv",
        df=output_df,
    )

    return {
        "stage": "robustness_protocol_lopo",
        "status": "ok",
        "stage_dir": str(stage_dir),
        "run_key": context.run_key,
        "n_predictions": int(output_df.shape[0]),
    }
