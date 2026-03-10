"""Uncertainty estimation utilities for repeated-seed analysis."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.experiments.cv import regression_metrics
from src.experiments.models import build_extratrees

SOH_REGION_DEFINITION = {
    "units": "percent",
    "order": ["Early-Life", "Mid-Life", "Aged"],
    "stages": {
        "Early-Life": {"soh_min": 95.0, "soh_max": 100.0},
        "Mid-Life": {"soh_min": 85.0, "soh_max": 95.0},
        "Aged": {"soh_min": 80.0, "soh_max": 85.0},
    },
    "out_of_range_policy": (
        "values_below_80_or_above_100_mapped_to_nearest_stage"
    ),
}


def run_repeated_seed_uncertainty(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_metadata_df: pd.DataFrame,
    best_params: dict,
    seeds: list[int],
    n_jobs: int,
    target: str,
    region_basis: str = "soh_true",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run repeated-seed retraining and summarize prediction uncertainty.

    Regioning is defined by fixed SOH stages in SOH_REGION_DEFINITION.
    """
    if str(region_basis) != "soh_true":
        raise ValueError(
            "Unsupported uncertainty region_basis. "
            "Currently supported: ['soh_true']"
        )

    repeated_rows: list[dict] = []

    for seed in seeds:
        model = build_extratrees(
            params=best_params,
            random_seed=int(seed),
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        run_df = pd.DataFrame(
            {
                "seed": int(seed),
                "cell": test_metadata_df["cell"].astype(str).to_numpy(),
                "cycle": test_metadata_df["cycle"].to_numpy(),
                "y_true": test_metadata_df["y_true"].to_numpy(),
                "soh_true": test_metadata_df["soh_true"].to_numpy(),
                "y_pred": y_pred,
            }
        )
        repeated_rows.append(run_df)

    repeated_predictions_df = pd.concat(repeated_rows, ignore_index=True)

    grouped = (
        repeated_predictions_df.groupby(
            ["cell", "cycle", "y_true", "soh_true"], as_index=False
        )
        .agg(
            y_pred_mean=("y_pred", "mean"),
            y_pred_std=("y_pred", "std"),
            y_pred_q05=("y_pred", lambda x: float(np.quantile(x, 0.05))),
            y_pred_q50=("y_pred", lambda x: float(np.quantile(x, 0.50))),
            y_pred_q95=("y_pred", lambda x: float(np.quantile(x, 0.95))),
        )
        .fillna(0.0)
    )

    overall_metrics = regression_metrics(
        y_true=grouped["y_true"],
        y_pred=grouped["y_pred_mean"],
    )
    uncertainty_summary = {
        "target": target,
        "n_repeats": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "overall_rmse_mean_prediction": float(overall_metrics["rmse"]),
        "overall_mae_mean_prediction": float(overall_metrics["mae"]),
        "overall_r2_mean_prediction": float(overall_metrics["r2"]),
        "mean_prediction_std": float(grouped["y_pred_std"].mean()),
        "prediction_std_q50": float(grouped["y_pred_std"].quantile(0.50)),
        "prediction_std_q90": float(grouped["y_pred_std"].quantile(0.90)),
        "prediction_std_q95": float(grouped["y_pred_std"].quantile(0.95)),
    }

    region_df = grouped.copy()
    soh_pct = region_df["soh_true"].astype(float)
    if float(soh_pct.max()) <= 1.5:
        soh_pct = soh_pct * 100.0
    region_df["soh_pct"] = soh_pct
    early_life = SOH_REGION_DEFINITION["stages"]["Early-Life"]
    mid_life = SOH_REGION_DEFINITION["stages"]["Mid-Life"]
    region_df["region"] = "Aged"
    region_df.loc[
        region_df["soh_pct"] >= float(early_life["soh_min"]), "region"
    ] = "Early-Life"
    region_df.loc[
        (region_df["soh_pct"] >= float(mid_life["soh_min"]))
        & (region_df["soh_pct"] < float(mid_life["soh_max"])),
        "region",
    ] = "Mid-Life"

    region_rows: list[dict] = []
    for region_name, region_data in region_df.groupby("region"):
        metrics = regression_metrics(
            y_true=region_data["y_true"],
            y_pred=region_data["y_pred_mean"],
        )
        region_rows.append(
            {
                "region": region_name,
                "n_samples": int(region_data.shape[0]),
                "soh_min_pct": float(region_data["soh_pct"].min()),
                "soh_max_pct": float(region_data["soh_pct"].max()),
                "rmse_mean_prediction": float(metrics["rmse"]),
                "mae_mean_prediction": float(metrics["mae"]),
                "r2_mean_prediction": float(metrics["r2"]),
                "mean_prediction_std": float(region_data["y_pred_std"].mean()),
                "prediction_std_q90": float(
                    region_data["y_pred_std"].quantile(0.90)
                ),
            }
        )
    uncertainty_by_region_df = pd.DataFrame(region_rows)
    stage_order = list(SOH_REGION_DEFINITION["order"])
    uncertainty_by_region_df["region"] = pd.Categorical(
        uncertainty_by_region_df["region"],
        categories=stage_order,
        ordered=True,
    )
    uncertainty_by_region_df = uncertainty_by_region_df.sort_values(
        "region"
    ).reset_index(drop=True)
    uncertainty_by_region_df["region"] = uncertainty_by_region_df[
        "region"
    ].astype(str)

    uncertainty_summary["region_basis"] = str(region_basis)
    uncertainty_summary["region_definition"] = deepcopy(
        SOH_REGION_DEFINITION
    )

    return (
        repeated_predictions_df,
        uncertainty_by_region_df,
        uncertainty_summary,
    )
