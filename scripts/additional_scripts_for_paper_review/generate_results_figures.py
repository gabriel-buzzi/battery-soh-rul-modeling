"""Additional figures for the first round of paper revisions (March 2026)."""

# %%[markdown]
# # Revision-round figure generation
#
# This script regenerates the figures used in `content/results.tex`.
# Each figure family is generated in a single code cell.

# %%
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
ROOT = Path().absolute().parent
ARTIFACT_ROOT = ROOT / "results" / "results" / "experiments" / "revision_round1"
MANIFEST = json.loads((ARTIFACT_ROOT / "manifest.json").read_text())
OUTPUT_DIR = ROOT / "paper" / "review_process" / "figures_revision_round1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SET_FULL = "full_all"
FEATURE_SET_CHARGE = "charge_all"
FIG_DPI = 300
FIG_WIDTH_MM = 140
FIG_HEIGHT_MM = 90
TARGETS = ["SOH", "RUL"]
REGION_ORDER = ["Early-Life", "Mid-Life", "Aged"]
TRACK_COLORS = {"Full cycle": "#0f766e", "Charge only": "#b45309"}
GROUP_COLORS = {"Remaining cells": "#94a3b8", "Worst-quartile cells": "#be123c"}

sns.set_theme(
    style="white",
    context="paper",
    rc={
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
    },
)


# %%[markdown]
# ## Figure 00: Baseline held-out scatters

# %%
for target in TARGETS:
    for feature_set_id, stem_prefix, label in [
        (FEATURE_SET_FULL, "full_cycle", "Full cycle"),
        (FEATURE_SET_CHARGE, "charge_only", "Charge only"),
    ]:
        run = [
            r
            for r in MANIFEST["runs"]
            if r["track"] == "final_eval"
            and r["target"] == target
            and r["feature_set_id"] == feature_set_id
        ][0]
        pred_df = pd.read_csv(ARTIFACT_ROOT / run["run_dir"] / "predictions.csv")

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4), constrained_layout=True)
        low = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
        high = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
        ax.plot([low, high], [low, high], "--", color="#475569", lw=1.5)
        ax.scatter(
            pred_df["y_true"],
            pred_df["y_pred"],
            s=10,
            alpha=0.35,
            color=TRACK_COLORS[label],
            edgecolors="none",
            rasterized=True,
        )

        residual = pred_df["y_true"].to_numpy() - pred_df["y_pred"].to_numpy()
        rmse = float(np.sqrt(np.mean(residual**2)))
        ss_res = float(np.sum((pred_df["y_true"] - pred_df["y_pred"]) ** 2))
        ss_tot = float(
            np.sum((pred_df["y_true"] - pred_df["y_true"].mean()) ** 2)
        )
        r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

        ax.set_title(f"{target}: {label}\nRMSE={rmse:.3f}, $R^2$={r2:.3f}")
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.grid(False)
        fig.savefig(
            OUTPUT_DIR / f"figure_00_{stem_prefix}_scatter_{target.lower()}.png",
            dpi=FIG_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


# %%[markdown]
# ## Figure 01: Full-cycle top-k curves

# %%
selected_k_full = {"SOH": 6, "RUL": 6}

for target in TARGETS:
    run = [
        r
        for r in MANIFEST["runs"]
        if r["track"] == "full_cycle_feature_analysis" and r["target"] == target
    ][0]
    df = pd.read_csv(ARTIFACT_ROOT / run["run_dir"] / "sweep.topk.csv").sort_values(
        "k"
    )
    df["k"] = df["k"].astype(int)
    df["val_rmse_mean"] = df["val_rmse_mean"].astype(float)
    baseline_rmse = float(df.loc[df["k"] == 16, "val_rmse_mean"].iloc[0])
    tolerance_rmse = baseline_rmse * 1.10
    selected_k = selected_k_full[target]
    best_row = df.loc[df["val_rmse_mean"].idxmin()]
    selected_row = df.loc[df["k"] == selected_k].iloc[0]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4), constrained_layout=True)
    ax.plot(
        df["k"],
        df["val_rmse_mean"],
        marker="o",
        linewidth=2.5,
        color=TRACK_COLORS["Full cycle"],
    )
    ax.axhline(
        tolerance_rmse,
        linestyle="--",
        linewidth=1.6,
        color="#64748b",
        label="10% tolerance",
    )
    ax.scatter([16], [baseline_rmse], s=100, color="#1d4ed8", zorder=4, label="16-feature baseline")
    ax.scatter(
        [selected_k],
        [selected_row["val_rmse_mean"]],
        s=140,
        color="#dc2626",
        zorder=5,
        label=f"Selected k={selected_k}",
    )
    ax.scatter(
        [best_row["k"]],
        [best_row["val_rmse_mean"]],
        s=110,
        color="#16a34a",
        zorder=5,
        label="Best RMSE",
    )
    ax.set_title(f"{target}: full-cycle top-k sweep")
    ax.set_xlabel("Number of retained features (k)")
    ax.set_ylabel("Validation RMSE")
    ax.set_xticks(df["k"])
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(False)
    fig.savefig(
        OUTPUT_DIR / f"figure_01_full_cycle_topk_{target.lower()}.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# %%[markdown]
# ## Figure 02: Full-cycle vs charge-only compactness

# %%
selected_k_compact = {
    "SOH": {"Full cycle": 6, "Charge only": 6},
    "RUL": {"Full cycle": 6, "Charge only": 4},
}

for target in TARGETS:
    full_run = [
        r
        for r in MANIFEST["runs"]
        if r["track"] == "full_cycle_feature_analysis" and r["target"] == target
    ][0]
    charge_run = [
        r
        for r in MANIFEST["runs"]
        if r["track"] == "charge_only_feature_analysis" and r["target"] == target
    ][0]
    full_df = pd.read_csv(ARTIFACT_ROOT / full_run["run_dir"] / "sweep.topk.csv").sort_values("k")
    charge_df = pd.read_csv(ARTIFACT_ROOT / charge_run["run_dir"] / "sweep.topk.csv").sort_values("k")
    for df in (full_df, charge_df):
        df["k"] = df["k"].astype(int)
        df["val_rmse_mean"] = df["val_rmse_mean"].astype(float)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4), constrained_layout=True)
    ax.plot(
        full_df["k"],
        full_df["val_rmse_mean"],
        marker="o",
        linewidth=2.5,
        color=TRACK_COLORS["Full cycle"],
        label="Full cycle",
    )
    ax.plot(
        charge_df["k"],
        charge_df["val_rmse_mean"],
        marker="o",
        linewidth=2.5,
        color=TRACK_COLORS["Charge only"],
        label="Charge only",
    )
    for label, df in [("Full cycle", full_df), ("Charge only", charge_df)]:
        k = selected_k_compact[target][label]
        row = df.loc[df["k"] == k].iloc[0]
        ax.scatter(
            [k],
            [row["val_rmse_mean"]],
            s=130,
            color=TRACK_COLORS[label],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
    ax.set_title(f"{target}: compactness comparison (full-cycle vs charge-only)")
    ax.set_xlabel("Number of retained features (k)")
    ax.set_ylabel("Validation RMSE")
    ax.set_xticks(sorted(full_df["k"].unique()))
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(False)
    fig.savefig(
        OUTPUT_DIR / f"figure_02_compactness_comparison_{target.lower()}.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# %%[markdown]
# ## Figure 03A: Uncertainty by region

# %%
for target in TARGETS:
    run = [
        r
        for r in MANIFEST["runs"]
        if r["track"] == "uncertainty"
        and r["target"] == target
        and r["feature_set_id"] == FEATURE_SET_FULL
    ][0]
    uncertainty_df = pd.read_csv(
        ARTIFACT_ROOT / run["run_dir"] / "uncertainty.by_region.csv"
    )
    uncertainty_df["region"] = pd.Categorical(
        uncertainty_df["region"], categories=REGION_ORDER, ordered=True
    )
    uncertainty_df = uncertainty_df.sort_values("region")
    uncertainty_df["mean_prediction_std"] = uncertainty_df[
        "mean_prediction_std"
    ].astype(float)
    uncertainty_df["rmse_mean_prediction"] = uncertainty_df[
        "rmse_mean_prediction"
    ].astype(float)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4), constrained_layout=True)
    x = np.arange(len(REGION_ORDER))
    ax.bar(
        x,
        uncertainty_df["mean_prediction_std"],
        width=0.65,
        color="#0f766e" if target == "SOH" else "#b45309",
        alpha=0.85,
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        uncertainty_df["rmse_mean_prediction"],
        color="#1f2937",
        marker="o",
        linewidth=2.0,
    )
    ax.set_xticks(x, REGION_ORDER)
    ax.set_title(f"{target}: uncertainty by life region")
    ax.set_ylabel("Mean prediction std")
    ax2.set_ylabel("RMSE of mean prediction")
    ax.grid(False)
    ax2.grid(False)
    fig.savefig(
        OUTPUT_DIR / f"figure_03_uncertainty_by_region_{target.lower()}.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# %%[markdown]
# ## Figure 03B: Repeated-seed scatter

# %%
for target in TARGETS:
    run = [
        r
        for r in MANIFEST["runs"]
        if r["track"] == "uncertainty"
        and r["target"] == target
        and r["feature_set_id"] == FEATURE_SET_FULL
    ][0]
    repeated_df = pd.read_csv(ARTIFACT_ROOT / run["run_dir"] / "predictions.csv")
    repeated_df["region"] = pd.cut(
        repeated_df["soh_true"].astype(float),
        bins=[-np.inf, 85.0, 95.0, np.inf],
        labels=["Aged", "Mid-Life", "Early-Life"],
        right=False,
    ).astype("object")
    repeated_df.loc[repeated_df["soh_true"].astype(float) >= 95.0, "region"] = "Early-Life"
    repeated_df.loc[
        (repeated_df["soh_true"].astype(float) >= 85.0)
        & (repeated_df["soh_true"].astype(float) < 95.0),
        "region",
    ] = "Mid-Life"
    repeated_df.loc[repeated_df["soh_true"].astype(float) < 85.0, "region"] = "Aged"

    summary_df = repeated_df.groupby(
        ["cell", "cycle", "y_true", "soh_true", "region"],
        observed=False,
        as_index=False,
    ).agg(y_pred_mean=("y_pred", "mean"))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4), constrained_layout=True)
    low = min(summary_df["y_true"].min(), summary_df["y_pred_mean"].min())
    high = max(summary_df["y_true"].max(), summary_df["y_pred_mean"].max())
    ax.plot([low, high], [low, high], "--", color="#475569", lw=1.5)
    region_palette = dict(zip(REGION_ORDER, sns.color_palette("viridis", len(REGION_ORDER))))
    for region in REGION_ORDER:
        region_df = summary_df[summary_df["region"] == region]
        ax.scatter(
            region_df["y_true"],
            region_df["y_pred_mean"],
            s=10,
            alpha=0.4,
            color=region_palette[region],
            edgecolors="none",
            rasterized=True,
            label=region,
        )
    ax.set_title(f"{target}: repeated-seed prediction scatter")
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Mean predicted {target}")
    ax.legend(title="Region", loc="best", fontsize=9, frameon=True)
    ax.grid(False)
    fig.savefig(
        OUTPUT_DIR / f"figure_03_uncertainty_scatter_{target.lower()}.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# %%[markdown]
# ## Figure 04A: Diagnostics scatter

# %%
for target in TARGETS:
    run = [
        r
        for r in MANIFEST["runs"]
        if r["track"] == "diagnostics"
        and r["target"] == target
        and r["feature_set_id"] == FEATURE_SET_FULL
    ][0]
    diag_df = pd.read_csv(ARTIFACT_ROOT / run["run_dir"] / "diagnostics.cells.csv")
    diag_df["rmse"] = diag_df["rmse"].astype(float)
    n_focus = math.ceil(len(diag_df) * 0.25)
    focus_cells = (
        diag_df.sort_values("rmse", ascending=False).head(n_focus)["cell"].tolist()
    )
    diag_df["is_focus"] = diag_df["cell"].isin(focus_cells)

    pred_df = pd.read_csv(ARTIFACT_ROOT / run["run_dir"] / "predictions.csv").merge(
        diag_df[["cell", "is_focus"]], on="cell", how="left"
    )
    pred_df["group"] = np.where(
        pred_df["is_focus"], "Worst-quartile cells", "Remaining cells"
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4), constrained_layout=True)
    low = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
    high = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
    ax.plot([low, high], [low, high], "--", color="#475569", lw=1.5)
    for group in ["Worst-quartile cells", "Remaining cells"]:
        plot_df = pred_df[pred_df["group"] == group]
        ax.scatter(
            plot_df["y_true"],
            plot_df["y_pred"],
            s=14,
            alpha=0.22 if group == "Remaining cells" else 0.16,
            color=GROUP_COLORS[group],
            edgecolors="none",
            rasterized=True,
            label=group,
            zorder=2 if group == "Worst-quartile cells" else 3,
        )
    ax.set_title(f"{target}: held-out prediction scatter by error group")
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.legend(title="", loc="best", fontsize=9, frameon=True)
    ax.grid(False)
    fig.savefig(
        OUTPUT_DIR / f"figure_04_scatter_{target.lower()}.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# %%[markdown]
# ## Figure 04B: Diagnostics RMSE boxplots

# %%
for target in TARGETS:
    run = [
        r
        for r in MANIFEST["runs"]
        if r["track"] == "diagnostics"
        and r["target"] == target
        and r["feature_set_id"] == FEATURE_SET_FULL
    ][0]
    diag_df = pd.read_csv(ARTIFACT_ROOT / run["run_dir"] / "diagnostics.cells.csv")
    diag_df["rmse"] = diag_df["rmse"].astype(float)
    n_focus = math.ceil(len(diag_df) * 0.25)
    focus_cells = (
        diag_df.sort_values("rmse", ascending=False).head(n_focus)["cell"].tolist()
    )
    diag_df["group"] = np.where(
        diag_df["cell"].isin(focus_cells),
        "Worst-quartile cells",
        "Remaining cells",
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4), constrained_layout=True)
    sns.boxplot(
        data=diag_df,
        x="group",
        y="rmse",
        hue="group",
        order=["Remaining cells", "Worst-quartile cells"],
        palette=GROUP_COLORS,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=diag_df,
        x="group",
        y="rmse",
        hue="group",
        order=["Remaining cells", "Worst-quartile cells"],
        palette=GROUP_COLORS,
        alpha=0.35,
        size=3,
        edgecolor="none",
        ax=ax,
    )
    ax.set_title(f"{target}: per-cell RMSE by error group")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE")
    ax.grid(False)
    fig.savefig(
        OUTPUT_DIR / f"figure_04_rmse_boxplot_{target.lower()}.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
