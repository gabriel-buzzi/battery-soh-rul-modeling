"""Debug plotting helpers for experiment runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_optimization_loss(
    optimization_history_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot trial objective value evolution over optimization trials."""
    x_col = (
        "number" if "number" in optimization_history_df.columns else "trial"
    )
    y_col = (
        "value"
        if "value" in optimization_history_df.columns
        else "objective_score"
    )
    if x_col not in optimization_history_df.columns:
        return
    if y_col not in optimization_history_df.columns:
        return

    df = optimization_history_df.sort_values(x_col).copy()
    if df.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df[x_col], df[y_col], marker="o", linewidth=1.5)
    ax.set_title("Optimization Loss Evolution")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prediction_scatter(
    predictions_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot predicted vs true scatter with ideal y=x reference line."""
    if predictions_df.empty:
        return

    if not {"y_true", "y_pred"}.issubset(predictions_df.columns):
        return

    min_val = min(
        predictions_df["y_true"].min(), predictions_df["y_pred"].min()
    )
    max_val = max(
        predictions_df["y_true"].max(), predictions_df["y_pred"].max()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(
        predictions_df["y_true"],
        predictions_df["y_pred"],
        alpha=0.6,
        s=18,
    )
    ax.plot(
        [min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.2
    )
    ax.set_title("Test Predictions: True vs Predicted")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_topk_vs_val_rmse(
    topk_sweep_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot top-k subset size versus validation RMSE."""
    required_cols = {"k", "val_rmse_mean"}
    if topk_sweep_df.empty:
        return
    if not required_cols.issubset(topk_sweep_df.columns):
        return

    df = topk_sweep_df.sort_values("k").copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["k"], df["val_rmse_mean"], marker="o", linewidth=1.8)
    ax.set_title("Top-k vs Validation RMSE")
    ax.set_xlabel("k (number of selected features)")
    ax.set_ylabel("Validation RMSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_topk_vs_relative_gap(
    topk_sweep_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot top-k subset size versus relative train/validation gap."""
    required_cols = {"k", "relative_gap_mean"}
    if topk_sweep_df.empty:
        return
    if not required_cols.issubset(topk_sweep_df.columns):
        return

    df = topk_sweep_df.sort_values("k").copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["k"], df["relative_gap_mean"], marker="o", linewidth=1.8)
    ax.set_title("Top-k vs Relative Gap")
    ax.set_xlabel("k (number of selected features)")
    ax.set_ylabel("Relative gap |RMSE_train - RMSE_val| / RMSE_val")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
