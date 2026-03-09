"""Uncertainty estimation utilities for repeated-seed analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.cv import regression_metrics
from src.experiments.models import build_extratrees


def run_repeated_seed_uncertainty(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_metadata_df: pd.DataFrame,
    best_params: dict,
    seeds: list[int],
    n_jobs: int,
    target: str,
    near_eol_quantile: float = 0.20,
    long_life_quantile: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run repeated-seed retraining and summarize prediction uncertainty."""
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
                "y_pred": y_pred,
            }
        )
        repeated_rows.append(run_df)

    repeated_predictions_df = pd.concat(repeated_rows, ignore_index=True)

    grouped = (
        repeated_predictions_df.groupby(["cell", "cycle", "y_true"], as_index=False)
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

    near_threshold = float(grouped["y_true"].quantile(near_eol_quantile))
    long_threshold = float(grouped["y_true"].quantile(long_life_quantile))

    region_df = grouped.copy()
    region_df["region"] = "mid_life"
    region_df.loc[region_df["y_true"] <= near_threshold, "region"] = "near_eol"
    region_df.loc[region_df["y_true"] >= long_threshold, "region"] = "long_life"

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
                "rmse_mean_prediction": float(metrics["rmse"]),
                "mae_mean_prediction": float(metrics["mae"]),
                "r2_mean_prediction": float(metrics["r2"]),
                "mean_prediction_std": float(region_data["y_pred_std"].mean()),
                "prediction_std_q90": float(region_data["y_pred_std"].quantile(0.90)),
            }
        )
    uncertainty_by_region_df = pd.DataFrame(region_rows).sort_values(
        "region"
    ).reset_index(drop=True)

    uncertainty_summary["near_eol_threshold"] = near_threshold
    uncertainty_summary["long_life_threshold"] = long_threshold
    uncertainty_summary["near_eol_quantile"] = near_eol_quantile
    uncertainty_summary["long_life_quantile"] = long_life_quantile

    return repeated_predictions_df, uncertainty_by_region_df, uncertainty_summary
