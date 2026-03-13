"""Diagnostics utilities for difficult-cell analysis."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def build_error_cells_summary(
    predictions_df: pd.DataFrame,
    top_n_cells: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Build per-cell error diagnostics and a compact summary payload."""
    work_df = predictions_df.copy()
    work_df["abs_error"] = (work_df["y_pred"] - work_df["y_true"]).abs()
    work_df["error"] = work_df["y_pred"] - work_df["y_true"]

    cell_rows: list[dict] = []
    for cell_id, cell_df in work_df.groupby("cell"):
        rmse = float(
            root_mean_squared_error(cell_df["y_true"], cell_df["y_pred"])
        )
        mae = float(mean_absolute_error(cell_df["y_true"], cell_df["y_pred"]))
        bias = float(cell_df["error"].mean())
        max_abs_error = float(cell_df["abs_error"].max())
        p90_abs_error = float(cell_df["abs_error"].quantile(0.90))

        # Region concentration along cell life (cycle position).
        if cell_df["cycle"].max() > cell_df["cycle"].min():
            cycle_pos = (cell_df["cycle"] - cell_df["cycle"].min()) / (
                cell_df["cycle"].max() - cell_df["cycle"].min()
            )
        else:
            cycle_pos = pd.Series(
                [0.0] * cell_df.shape[0], index=cell_df.index
            )

        region = pd.Series("mid_life", index=cell_df.index)
        region.loc[cycle_pos <= 0.33] = "early_life"
        region.loc[cycle_pos >= 0.67] = "late_life"
        region_mae = (
            pd.DataFrame({"region": region, "abs_error": cell_df["abs_error"]})
            .groupby("region")["abs_error"]
            .mean()
            .to_dict()
        )
        dominant_region = max(region_mae, key=region_mae.get)

        cell_rows.append(
            {
                "cell": str(cell_id),
                "n_samples": int(cell_df.shape[0]),
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "max_abs_error": max_abs_error,
                "p90_abs_error": p90_abs_error,
                "dominant_error_region": dominant_region,
                "mae_early_life": float(region_mae.get("early_life", 0.0)),
                "mae_mid_life": float(region_mae.get("mid_life", 0.0)),
                "mae_late_life": float(region_mae.get("late_life", 0.0)),
                "y_true_max": float(cell_df["y_true"].max()),
            }
        )

    summary_df = pd.DataFrame(cell_rows).sort_values("rmse", ascending=False)
    difficult_cells = summary_df.head(top_n_cells)["cell"].astype(str).tolist()
    summary_df["is_difficult_cell"] = summary_df["cell"].isin(difficult_cells)
    summary_df = summary_df.reset_index(drop=True)

    difficult_df = summary_df[summary_df["is_difficult_cell"]]
    rest_df = summary_df[~summary_df["is_difficult_cell"]]
    diagnostics_summary = {
        "top_n_cells": int(top_n_cells),
        "difficult_cells": difficult_cells,
        "mean_rmse_difficult": float(difficult_df["rmse"].mean())
        if not difficult_df.empty
        else 0.0,
        "mean_rmse_rest": float(rest_df["rmse"].mean())
        if not rest_df.empty
        else 0.0,
        "mean_mae_difficult": float(difficult_df["mae"].mean())
        if not difficult_df.empty
        else 0.0,
        "mean_mae_rest": float(rest_df["mae"].mean())
        if not rest_df.empty
        else 0.0,
    }
    return summary_df, diagnostics_summary
