"""Thin dispatcher for experiment track execution."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from severson_features_soh_rul.modeling.dataset import (
    load_features_dataframe,
    resolve_feature_columns,
)
from severson_features_soh_rul.modeling.schemas import (
    CHARGE_FEATURE_COLUMNS,
    FULL_CYCLE_FEATURE_COLUMNS,
    SUPPORTED_TARGETS,
    TEMPERATURE_FEATURE_COLUMNS,
    validate_required_columns,
)
from severson_features_soh_rul.modeling.split import apply_cell_split, create_or_load_cell_split
from severson_features_soh_rul.modeling.tracks.diagnostics import run_diagnostics_track
from severson_features_soh_rul.modeling.tracks.feature_analysis import run_feature_analysis_track
from severson_features_soh_rul.modeling.tracks.final_eval import run_final_eval_track
from severson_features_soh_rul.modeling.tracks.protocol_robustness import (
    run_protocol_robustness_track,
)
from severson_features_soh_rul.modeling.tracks.uncertainty import run_uncertainty_track


@hydra.main(
    version_base=None,
    config_path="../../../config/experiments",
    config_name="base",
)
def run_experiment(cfg: DictConfig) -> None:
    """Route execution to one experiment track."""
    if cfg.target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported target={cfg.target}. Supported targets: "
            f"{sorted(SUPPORTED_TARGETS)}"
        )

    features_path = Path(to_absolute_path(cfg.data.features_data_path))
    split_dir = Path(to_absolute_path(cfg.data.split_dir))
    artifacts_root = Path(to_absolute_path(cfg.artifacts.root_dir))

    features_df = load_features_dataframe(features_data_path=features_path)

    feature_columns = resolve_feature_columns(
        feature_set_id=str(cfg.features.set_id),
        target=str(cfg.target),
    )
    validate_required_columns(
        features_df=features_df,
        required_columns=[cfg.target, *feature_columns],
    )

    train_cells, test_cells = create_or_load_cell_split(
        features_df=features_df,
        split_dir=split_dir,
        train_cells_proportion=float(cfg.data.train_cells_proportion),
        random_seed=int(cfg.random_seed),
        force_recreate=bool(cfg.data.force_recreate_split),
    )
    train_df, test_df = apply_cell_split(
        features_df=features_df,
        train_cells=train_cells,
        test_cells=test_cells,
    )

    track_name = str(cfg.track)
    if track_name == "full_cycle_feature_analysis":
        full_no_temp_columns = [
            col
            for col in FULL_CYCLE_FEATURE_COLUMNS
            if col not in TEMPERATURE_FEATURE_COLUMNS
        ]
        run_feature_analysis_track(
            cfg=cfg,
            train_df=train_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            test_cells=test_cells,
            base_feature_columns=FULL_CYCLE_FEATURE_COLUMNS.copy(),
            no_temp_feature_columns=full_no_temp_columns,
        )
        return

    if track_name == "charge_only_feature_analysis":
        charge_temp_columns = [
            f"charge_{col}" for col in TEMPERATURE_FEATURE_COLUMNS
        ]
        charge_no_temp_columns = [
            col
            for col in CHARGE_FEATURE_COLUMNS
            if col not in charge_temp_columns
        ]
        run_feature_analysis_track(
            cfg=cfg,
            train_df=train_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            test_cells=test_cells,
            base_feature_columns=CHARGE_FEATURE_COLUMNS.copy(),
            no_temp_feature_columns=charge_no_temp_columns,
        )
        return

    if track_name == "uncertainty":
        run_uncertainty_track(
            cfg=cfg,
            train_df=train_df,
            test_df=test_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            test_cells=test_cells,
            feature_columns=feature_columns,
        )
        return

    if track_name == "diagnostics":
        run_diagnostics_track(
            cfg=cfg,
            train_df=train_df,
            test_df=test_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            test_cells=test_cells,
            feature_columns=feature_columns,
        )
        return

    if track_name == "protocol_robustness":
        run_protocol_robustness_track(
            cfg=cfg,
            features_df=features_df,
            train_df=train_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            test_cells=test_cells,
            feature_columns=feature_columns,
        )
        return

    if track_name == "final_eval":
        run_final_eval_track(
            cfg=cfg,
            train_df=train_df,
            test_df=test_df,
            artifacts_root=artifacts_root,
            train_cells=train_cells,
            test_cells=test_cells,
            feature_columns=feature_columns,
            track_name=track_name,
        )
        return

    raise ValueError(
        "Unsupported track="
        f"{track_name}. Supported tracks are "
        "['final_eval', 'full_cycle_feature_analysis', "
        "'charge_only_feature_analysis', 'uncertainty', "
        "'diagnostics', 'protocol_robustness']."
    )


if __name__ == "__main__":
    run_experiment()
