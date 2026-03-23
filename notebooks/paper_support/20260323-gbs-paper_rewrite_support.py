"""Paper-support notebook for methodology/results rewrite figures."""

# %% [markdown]
# # Paper Rewrite Support Figures
#
# This notebook gathers the figures and table-figures that are most useful for
# the current manuscript rewrite. It keeps the figures aligned with the current
# paper scope:
#
# - full-cycle features only,
# - quantile-conformal uncertainty,
# - grouped optimization on training cells,
# - permutation-based explainability,
# - strict protocol robustness.
#
# The notebook saves new outputs directly into `paper/overleaf_project`, but it
# never overwrites files that already exist there.

# %%
import json
from pathlib import Path
import re
import sys

from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from severson_features_soh_rul.utils.plots import create_figure


# %%
def resolve_repo_root() -> Path:
    """Resolve repository root in both notebook and script execution."""
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    return Path().resolve().parent.parent


REPO_ROOT = resolve_repo_root()
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


plt.style.use(REPO_ROOT / ".matplotlibrc")

SAVE_TO_OVERLEAF = True
EXAMPLE_CELL = "b1c20"
OVERLEAF_DIR = REPO_ROOT / "paper" / "overleaf_project"
PROCESSED_CELLS_DIR = REPO_ROOT / "data" / "processed" / "cells"
FEATURES_PATH = REPO_ROOT / "data" / "interim" / "features.parquet"
RESULTS_ROOT = REPO_ROOT / "results_refactored" / "modeling"

FULL_CYCLE_FEATURES = [
    "V_mean",
    "V_median",
    "V_std",
    "V_iqr",
    "V_kurtosis",
    "V_entropy",
    "I_mean",
    "I_median",
    "I_std",
    "I_iqr",
    "I_kurtosis",
    "T_mean",
    "T_median",
    "T_std",
    "T_iqr",
    "T_kurtosis",
]

FEATURE_LABELS = {
    "V_mean": "Voltage Mean",
    "V_median": "Voltage Median",
    "V_std": "Voltage Std. Dev.",
    "V_iqr": "Voltage IQR",
    "V_kurtosis": "Voltage Kurtosis",
    "V_entropy": "Voltage Entropy",
    "I_mean": "Current Mean",
    "I_median": "Current Median",
    "I_std": "Current Std. Dev.",
    "I_iqr": "Current IQR",
    "I_kurtosis": "Current Kurtosis",
    "T_mean": "Temperature Mean",
    "T_median": "Temperature Median",
    "T_std": "Temperature Std. Dev.",
    "T_iqr": "Temperature IQR",
    "T_kurtosis": "Temperature Kurtosis",
}


# %%
def save_figure(fig: plt.Figure, filename: str, save_pdf=True) -> None:
    """Save a figure into the Overleaf project without overwriting files."""
    output_path = OVERLEAF_DIR / filename
    if not SAVE_TO_OVERLEAF:
        print(f"SAVE_TO_OVERLEAF=False -> skipped {output_path.name}")
        return
    if output_path.exists():
        print(f"Skipping existing file: {output_path.name}")
        return
    if save_pdf:
        fig.savefig(
            f"{output_path}.pdf",
            dpi=500,
            bbox_inches="tight",
            format="pdf",
        )
    else:
        fig.savefig(
            f"{output_path}",
            dpi=500,
            bbox_inches="tight",
        )
    print(f"Saved {output_path}")


# %%
def add_panel_labels(axes) -> None:
    """Add (a), (b), ... labels to a matplotlib axes collection."""
    axes_list = list(np.ravel(axes))
    for idx, axis in enumerate(axes_list):
        axis.text(
            -0.12,
            1.08,
            f"({chr(97 + idx)})",
            transform=axis.transAxes,
            va="top",
        )


# %%
def resolve_artifact_dir(target: str) -> Path:
    """Resolve the single artifact root used in the paper analyses."""
    matches = sorted(
        RESULTS_ROOT.glob(
            f"target-{target.lower()}__feature_hash-*__"
            "split_seed-42__model_name-extratrees_quantile__"
            "weighting_strategy-none__*"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one artifact directory for {target}, found "
            f"{len(matches)}."
        )
    return matches[0]


# %%
def extract_charge_policy_metadata(features_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-cell metadata with protocol-derived summaries."""
    metadata = (
        features_df.groupby("cell", as_index=False)
        .agg(
            charge_policy=("charge_policy", "first"),
            cycle_life=("RUL", "max"),
        )
        .copy()
    )

    pattern = r"(\d+(?:\.\d+)?)C\((\d+)%\)-(\d+(?:\.\d+)?)(?:C)?"

    def parse_policy(policy: str) -> pd.Series:
        match = re.search(pattern, str(policy))
        if match is None:
            return pd.Series([np.nan, np.nan, np.nan])
        return pd.Series(
            [
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            ]
        )

    metadata[["charge_step1", "change_soc", "charge_step2"]] = metadata[
        "charge_policy"
    ].apply(parse_policy)
    metadata["mean_charge_c_rate"] = (
        metadata["charge_step1"] * metadata["change_soc"]
        + metadata["charge_step2"] * (80.0 - metadata["change_soc"])
    ) / 80.0
    return metadata.dropna(subset=["mean_charge_c_rate"]).reset_index(
        drop=True
    )


# %%
def build_table_figure(
    df: pd.DataFrame, title: str, layout_type: str
) -> plt.Figure:
    """Render a dataframe as a simple matplotlib table figure."""
    fig, ax = create_figure(layout_type=layout_type)
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    fig.suptitle(title)
    return fig


# %% [markdown]
# ## Shared data
#
# Each figure cell below either loads its own data or uses these small shared
# objects. This keeps the notebook easy to rerun while avoiding too much
# repeated path handling.

# %%
features_df = pd.read_parquet(FEATURES_PATH)
example_cycle_df = pd.read_parquet(
    PROCESSED_CELLS_DIR / f"{EXAMPLE_CELL}.parquet"
)
artifact_dirs = {
    target: resolve_artifact_dir(target) for target in ["SOH", "RUL"]
}
metadata_df = extract_charge_policy_metadata(features_df)


# %% [markdown]
# ## Figure 1: Pipeline overview
#
# This is a new figure for the methodology rewrite. It summarizes the current
# adopted workflow using the same three blocks that will structure the Methods
# section.

# %%
fig, ax = create_figure(layout_type="double_column")
ax.axis("off")
fig.suptitle("Overview of the adopted modeling workflow")

rows = {
    "Data Processing and Feature Extraction": [
        (0.08, 0.76, "Raw cycle signals\n(V, I, T)"),
        (0.31, 0.76, "Cycle validation\nand feature extraction"),
        (0.54, 0.76, "Full-cycle\n16-feature table"),
    ],
    "Modeling and Optimization": [
        (0.08, 0.48, "Cell-wise\ntrain/test split"),
        (0.31, 0.48, "Grouped CV\nOptuna optimization"),
        (0.54, 0.48, "Final quantile forest\n+ conformal calibration"),
        (0.77, 0.48, "Held-out prediction\nand strict protocol LOPO"),
    ],
    "Model Explainability": [
        (0.20, 0.20, "Permutation\npredictions"),
        (0.47, 0.20, "Composite\nranking"),
        (0.74, 0.20, "Top-k compactness\nsweep"),
    ],
}

for row_title, boxes in rows.items():
    first_x = min(box[0] for box in boxes)
    y = boxes[0][1]
    ax.text(
        first_x - 0.02,
        y + 0.12,
        row_title,
        transform=ax.transAxes,
        fontweight="bold",
        ha="left",
    )
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "black",
            },
        )
    for (x0, y0, _), (x1, y1, _) in zip(boxes[:-1], boxes[1:]):
        arrow = FancyArrowPatch(
            (x0 + 0.09, y0),
            (x1 - 0.09, y1),
            transform=ax.transAxes,
            arrowstyle="->",
            mutation_scale=10,
            linewidth=1.0,
            color="black",
        )
        ax.add_patch(arrow)

save_figure(fig, "paper_support_pipeline_overview")
plt.show()


# %% [markdown]
# ## Figure 2: Example full-cycle signals
#
# This reproduces the most relevant signal-level figure for the Methods section:
# one representative diagnostic cycle with voltage, current, and temperature.

# %%
cycle_id = int(example_cycle_df["cycle"].min())
cycle_data = example_cycle_df[example_cycle_df["cycle"] == cycle_id].copy()

fig, axes = create_figure(layout_type="one_and_half_column", nrows=3, ncols=1)
fig.suptitle(
    f"Measured signals on the first diagnostic cycle of cell {EXAMPLE_CELL}"
)
add_panel_labels(axes)

signals = [
    ("V", "Voltage (V)", "Voltage"),
    ("I", "Current (A)", "Current"),
    ("T", "Temperature (°C)", "Temperature"),
]
for axis, (column, ylabel, title) in zip(axes, signals):
    axis.plot(cycle_data["t"], cycle_data[column], color="C0")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
axes[-1].set_xlabel("Time (s)")

save_figure(fig, "paper_support_example_cycle_signals")
plt.show()


# %% [markdown]
# ## Figure 3: Example full-cycle feature trajectories
#
# This repeats the existing example-cell feature view, keeping the message close
# to the current paper: the handcrafted descriptors evolve smoothly over life
# and expose different degradation signatures.

# %%
example_features_df = (
    features_df[features_df["cell"].astype(str) == EXAMPLE_CELL]
    .sort_values("cycle")
    .copy()
)

fig, axes = create_figure(layout_type="double_column", nrows=6, ncols=3)
fig.suptitle(
    f"Full-cycle feature trajectories for example cell {EXAMPLE_CELL}"
)
add_panel_labels(axes)
axes_flat = np.ravel(axes)

for axis, feature in zip(axes_flat, FULL_CYCLE_FEATURES):
    axis.plot(
        example_features_df["cycle"], example_features_df[feature], color="C0"
    )
    axis.set_title(FEATURE_LABELS[feature])
    axis.set_xlabel("Cycle")
    axis.set_ylabel("Value")
for axis in axes_flat[len(FULL_CYCLE_FEATURES) :]:
    axis.axis("off")

save_figure(fig, "paper_support_example_feature_trajectories")
plt.show()


# %% [markdown]
# ## Figure 4: Feature-target correlation heatmap
#
# This figure remains useful as a descriptive summary of the full-cycle feature
# space, provided the manuscript makes clear that correlation is exploratory and
# not the formal ranking rule used later.

# %%
corr_df = (
    features_df[FULL_CYCLE_FEATURES + ["SOH", "RUL"]]
    .corr()
    .loc[FULL_CYCLE_FEATURES, ["SOH", "RUL"]]
    .abs()
)

fig, ax = create_figure(layout_type="one_and_half_column")
fig.suptitle("Absolute correlation between full-cycle features and targets")
sns.heatmap(
    corr_df,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=False,
    ax=ax,
)
ax.set_xlabel("Target")
ax.set_ylabel("Feature")

save_figure(fig, "paper_support_feature_target_correlation")
plt.show()


# %% [markdown]
# ## Figure 5: Protocol context summary
#
# This combines the existing cycle-life versus charge-severity idea with a small
# distribution view so that the protocol axis used later in strict LOPO is less
# abstract.

# %%
fig, axes = create_figure(layout_type="one_and_half_column", nrows=1, ncols=2)
fig.suptitle("Protocol context in the Severson dataset")
add_panel_labels(axes)

axes[0].hist(metadata_df["mean_charge_c_rate"], bins=12, color="C0")
axes[0].set_title("Distribution of mean charge C-rate")
axes[0].set_xlabel("Mean charge C-rate")
axes[0].set_ylabel("Number of cells")

x = metadata_df["mean_charge_c_rate"].to_numpy()
y = metadata_df["cycle_life"].to_numpy()
axes[1].scatter(x, y, color="C0")
if x.size >= 2:
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[1].plot(
        x_line, slope * x_line + intercept, color="C1", linestyle="--"
    )
# TODO: Add the value of the pearson CC at the title of the axes[1] plot
axes[1].set_title("Cycle life versus average charge C-rate")
axes[1].set_xlabel("Average charge C-rate")
axes[1].set_ylabel("Cycle life (cycles)")

save_figure(fig, "paper_support_protocol_context")
plt.show()


# %% [markdown]
# ## Table-Figure 1: Optimization summary
#
# This is a new compact table-figure for the Methods and Results rewrite. It
# condenses the selected hyperparameters together with the main grouped-CV
# optimization metrics of the best trial for each target.

# %%
summary_rows = []
for target, artifact_dir in artifact_dirs.items():
    cv_trials = pd.read_csv(artifact_dir / "optimize" / "cv_trials.csv")
    best_trial = cv_trials.loc[cv_trials["objective"].idxmin()]
    summary_rows.append(
        {
            "Target": target,
            "n_estimators": int(best_trial["n_estimators"]),
            "max_depth": int(best_trial["max_depth"]),
            "min_samples_split": int(best_trial["min_samples_split"]),
            "min_samples_leaf": int(best_trial["min_samples_leaf"]),
            "max_features": str(best_trial["max_features"]),
            "Val. RMSE": round(float(best_trial["rmse_val_mean"]), 3),
            "Gap": round(float(best_trial["overfit_gap_mean"]), 3),
            "Objective": round(float(best_trial["objective"]), 3),
        }
    )
optimization_summary_df = pd.DataFrame(summary_rows)
optimization_summary_df

# %%
# TODO: Do not generate an image with the table, instead print it in latex so I can copy and paste later
fig = build_table_figure(
    df=optimization_summary_df,
    title="Best grouped-CV optimization summary for SOH and RUL",
    layout_type="double_column",
)
save_figure(fig, "paper_support_optimization_summary_table")
plt.show()


# %% [markdown]
# ## Figure 6: Feature-family contribution in the ranking and selected compact subsets
#
# This is a new explainability figure. It summarizes how much of the ranking and
# top-k selections is dominated by voltage, current, or temperature descriptors.

# TODO: I don't think it's necessary to have both top-10 and selected-k, only selected-k would be sufficient, additionally I think it would be better to discuss this on the paper only by looking at a table that contains the ranking of features and the topk sweep errorbar with the plot from the other notebook.


# %%
def feature_family(feature_name: str) -> str:
    if str(feature_name).startswith("V_"):
        return "Voltage"
    if str(feature_name).startswith("I_"):
        return "Current"
    if str(feature_name).startswith("T_"):
        return "Temperature"
    return "Other"


family_rows = []
for target, artifact_dir in artifact_dirs.items():
    ranking_df = pd.read_csv(artifact_dir / "rank" / "ranking_composite.csv")
    with open(artifact_dir / "topk_sweep" / "topk_selection.json") as handle:
        topk_selection = json.load(handle)

    top10 = ranking_df.head(10)["feature"].astype(str).tolist()
    selected_features = [
        str(feature) for feature in topk_selection["selected_features"]
    ]

    for subset_name, features_list in [
        ("Top 10 ranking", top10),
        (
            f"Selected top-k (k={topk_selection['selected_k']})",
            selected_features,
        ),
    ]:
        counts = pd.Series(
            [feature_family(name) for name in features_list]
        ).value_counts()
        for family in ["Voltage", "Current", "Temperature"]:
            family_rows.append(
                {
                    "target": target,
                    "subset": subset_name,
                    "family": family,
                    "count": int(counts.get(family, 0)),
                }
            )
family_df = pd.DataFrame(family_rows)
family_df

# %%
fig, axes = create_figure(layout_type="one_and_half_column", nrows=1, ncols=2)
fig.suptitle(
    "Feature-family composition of the ranking and selected compact subsets"
)
add_panel_labels(axes)

for axis, target in zip(axes, ["SOH", "RUL"]):
    target_df = family_df[family_df["target"] == target].copy()
    pivot = target_df.pivot(
        index="subset", columns="family", values="count"
    ).fillna(0)
    x = np.arange(pivot.shape[0])
    width = 0.22
    colors = {"Voltage": "C0", "Current": "C1", "Temperature": "C2"}
    for offset, family in enumerate(["Voltage", "Current", "Temperature"]):
        axis.bar(
            x + (offset - 1) * width,
            pivot[family].to_numpy(),
            width=width,
            label=family,
            color=colors[family],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(pivot.index, rotation=15, ha="right")
    axis.set_ylabel("Number of features")
    axis.set_title(target)
axes[1].legend(loc="upper right")

save_figure(fig, "paper_support_feature_family_contribution")
plt.show()


# %% [markdown]
# ## Table-Figure 2: Compact subset summary
#
# This compact table records the selected subset sizes and the count of voltage,
# current, and temperature features retained for each target.

# %%
compact_rows = []
for target, artifact_dir in artifact_dirs.items():
    with open(artifact_dir / "topk_sweep" / "topk_selection.json") as handle:
        topk_selection = json.load(handle)
    selected_features = [
        str(feature) for feature in topk_selection["selected_features"]
    ]
    families = pd.Series([feature_family(name) for name in selected_features])
    compact_rows.append(
        {
            "Target": target,
            "Selected k": int(topk_selection["selected_k"]),
            "Voltage": int((families == "Voltage").sum()),
            "Current": int((families == "Current").sum()),
            "Temperature": int((families == "Temperature").sum()),
        }
    )
compact_summary_df = pd.DataFrame(compact_rows)
compact_summary_df

# %%
fig = build_table_figure(
    df=compact_summary_df,
    title="Signal-family composition of the selected compact subsets",
    layout_type="one_and_half_column",
)
save_figure(fig, "paper_support_compact_subset_summary_table")
plt.show()


# %% [markdown]
# ## Generated filenames
#
# These are the new non-conflicting filenames used by this notebook inside the
# Overleaf project folder:
#
# - `paper_support_pipeline_overview`
# - `paper_support_example_cycle_signals`
# - `paper_support_example_feature_trajectories`
# - `paper_support_feature_target_correlation`
# - `paper_support_protocol_context`
# - `paper_support_optimization_summary_table`
# - `paper_support_feature_family_contribution`
# - `paper_support_compact_subset_summary_table`
