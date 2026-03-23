"""Analyse results after pipeline refactor for paper revision."""

# %%
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error

from severson_features_soh_rul.utils.plots import create_figure

plt.style.use("../../.matplotlibrc")

# %%
BASE_PATH = Path().absolute().parent.parent

RUL_ARTIFACTS = (
    BASE_PATH
    / "results_refactored"
    / "modeling"
    / "target-rul__feature_hash-d3c068bcf5e7a4c6ed2306b1952bf61d5ebe1e1691881fc154114f0df9bdf07c__split_seed-42__model_name-extratrees__weighting_strategy-none__rk-db1500afda"
)
SOH_ARTIFACTS = (
    BASE_PATH
    / "results_refactored"
    / "modeling"
    / "target-soh__feature_hash-d3c068bcf5e7a4c6ed2306b1952bf61d5ebe1e1691881fc154114f0df9bdf07c__split_seed-42__model_name-extratrees__weighting_strategy-none__rk-e7ebb21c64"
)
# RUL_NO_TEMP_ARTIFACTS = (
#     BASE_PATH
#     / "results_refactored"
#     / "modeling"
#     / "target-rul__feature_hash-82854e4bcdd504169be5162d100888a4a50cce5db7d665692a29ab5d23a5b70d__split_seed-42__model_name-extratrees__weighting_strategy-none__rk-4074892c0e"
# )
# SOH_NO_TEMP_ARTIFACTS = (
#     BASE_PATH
#     / "results_refactored"
#     / "modeling"
#     / "target-soh__feature_hash-82854e4bcdd504169be5162d100888a4a50cce5db7d665692a29ab5d23a5b70d__split_seed-42__model_name-extratrees__weighting_strategy-none__rk-357a381862"
# )

# %% [markdown]
# ## Optimization results

# %%
soh_optimize_path = SOH_ARTIFACTS / "optimize"
rul_optimize_path = RUL_ARTIFACTS / "optimize"

# %%
soh_optimize_history = pd.read_csv(soh_optimize_path / "cv_trials.csv")
rul_optimize_history = pd.read_csv(rul_optimize_path / "cv_trials.csv")

# %%
fig, ax = create_figure(layout_type="one_and_a_half")
soh_best_trial = soh_optimize_history.loc[
    soh_optimize_history["objective"].idxmin()
]
sns.scatterplot(
    x=soh_optimize_history["rmse_val_mean"],
    y=soh_optimize_history["overfit_gap_mean"],
    hue=soh_optimize_history["objective"],
    ax=ax,
)
ax.scatter(
    [soh_best_trial["rmse_val_mean"]],
    [soh_best_trial["overfit_gap_mean"]],
    label="Best Trial",
    c="red",
)
ax.set_xlabel("Validation RMSE (Accuracy)")
ax.set_ylabel("Validation Gap (Generalization)")
ax.set_title("Trade-off Between Accuracy and Generalization (SOH)")
plt.legend(title="Objective")

# %%
fig, ax = create_figure(layout_type="one_and_a_half")
rul_best_trial = rul_optimize_history.loc[
    rul_optimize_history["objective"].idxmin()
]
sns.scatterplot(
    x=rul_optimize_history["rmse_val_mean"],
    y=rul_optimize_history["overfit_gap_mean"],
    hue=rul_optimize_history["objective"],
    ax=ax,
)
ax.scatter(
    [rul_best_trial["rmse_val_mean"]],
    [rul_best_trial["overfit_gap_mean"]],
    label="Best Trial",
    c="red",
)
ax.set_xlabel("Validation RMSE (Accuracy)")
ax.set_ylabel("Validation Gap (Generalization)")
ax.set_title("Trade-off Between Accuracy and Generalization (RUL)")
plt.legend(title="Objective")

# %%
soh_best_trial = soh_optimize_history.loc[
    soh_optimize_history["objective"].idxmin()
]
rul_best_trial = rul_optimize_history.loc[
    rul_optimize_history["objective"].idxmin()
]

# %%
param_names = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "max_features",
    "criterion",
]
print(f"Best SOH params:\n{soh_best_trial[param_names].to_dict()}")
print(f"Best RUL params:\n{rul_best_trial[param_names].to_dict()}")

# %% [markdown]
# ## Final Model

# %%
soh_final_model_path = SOH_ARTIFACTS / "predict" / "predictions_test.parquet"
rul_final_model_path = RUL_ARTIFACTS / "predict" / "predictions_test.parquet"

# %%
soh_final_model_preds = pd.read_parquet(soh_final_model_path)
rul_final_model_preds = pd.read_parquet(rul_final_model_path)

soh_test_cells_rmse = (
    soh_final_model_preds.groupby("cell")[["y_true", "y_pred"]]
    .apply(lambda g: root_mean_squared_error(g["y_true"], g["y_pred"]))
    .sort_values()
)
rul_test_cells_rmse = (
    rul_final_model_preds.groupby("cell")[["y_true", "y_pred"]]
    .apply(lambda g: root_mean_squared_error(g["y_true"], g["y_pred"]))
    .sort_values()
)
soh_test_cells_r2 = (
    soh_final_model_preds.groupby("cell")[["y_true", "y_pred"]]
    .apply(lambda g: r2_score(g["y_true"], g["y_pred"]))
    .sort_values(ascending=False)
)
rul_test_cells_r2 = (
    rul_final_model_preds.groupby("cell")[["y_true", "y_pred"]]
    .apply(lambda g: r2_score(g["y_true"], g["y_pred"]))
    .sort_values(ascending=False)
)

# %%
fig, ax = create_figure(
    "DOUBLE", aspect_ratio=0.75, ncols=5, nrows=5, sharey=True
)
axes = ax.flatten()
fig.suptitle("Battery SOH Model Fidelity Assessment")
for i, cell in enumerate(soh_test_cells_rmse.index):
    example_cell = soh_final_model_preds[
        soh_final_model_preds["cell"] == cell
    ].sort_values("cycle")

    cycles = example_cell["cycle"]
    y_true = example_cell["y_true"]
    y_pred = example_cell["y_pred"]
    y_pred_hi = example_cell["y_pred_hi"]
    y_pred_lo = example_cell["y_pred_lo"]

    # 1. Calculate RMSE
    rmse = root_mean_squared_error(y_true, y_pred)

    # 2. Plotting (keep labels for the global legend)
    fill = axes[i].fill_between(
        cycles, y_pred_hi, y_pred_lo, color="gray", alpha=0.35, label="95% CI"
    )
    (line_pred,) = axes[i].plot(cycles, y_pred, label="Estimate", c="blue")
    (line_true,) = axes[i].plot(cycles, y_true, label="Reference", c="red")

    # 3. Local Legend: Only show RMSE
    # We pass an empty handle or a "proxy" to show just the text
    axes[i].legend(
        [line_pred],
        [f"RMSE: {rmse:.2f} %"],
        handlelength=0,
        handletextpad=0,
        loc="lower left",
    )

    # axes[i].set_ylabel("SOH (%)")
    # axes[i].set_xlabel("Cycles")

    axes[i].set_title(f"Cell {cell}")

# 4. Global Legend
# We take the handles/labels from the very last subplot processed
handles = [fill, line_pred, line_true]
labels = ["95% CI", "Estimate", "Reference"]

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.05),  # Adjust this to sit below the plots
    frameon=True,
)

plt.tight_layout()
plt.show()

# %%
fig, ax = create_figure("DOUBLE", aspect_ratio=0.75, ncols=5, nrows=5)
axes = ax.flatten()
fig.suptitle("Battery RUL Model Fidelity Assessment")
for i, cell in enumerate(rul_test_cells_rmse.index):
    example_cell = rul_final_model_preds[
        rul_final_model_preds["cell"] == cell
    ].sort_values("cycle")

    cycles = example_cell["cycle"]
    y_true = example_cell["y_true"]
    y_pred = example_cell["y_pred"]
    y_pred_hi = example_cell["y_pred_hi"]
    y_pred_lo = example_cell["y_pred_lo"]

    # 1. Calculate RMSE
    rmse = root_mean_squared_error(y_true, y_pred)

    # 2. Plotting (keep labels for the global legend)
    fill = axes[i].fill_between(
        cycles, y_pred_hi, y_pred_lo, color="gray", alpha=0.35, label="95% CI"
    )
    (line_pred,) = axes[i].plot(cycles, y_pred, label="Estimate", c="blue")
    (line_true,) = axes[i].plot(cycles, y_true, label="Reference", c="red")

    # 3. Local Legend: Only show RMSE
    # We pass an empty handle or a "proxy" to show just the text
    axes[i].legend(
        [line_pred],
        [f"RMSE: {rmse:.2f} %"],
        handlelength=0,
        handletextpad=0,
        loc="lower left",
    )

    # axes[i].set_ylabel("RUL (Cycles)")
    # axes[i].set_xlabel("Cycles")

    axes[i].set_title(f"Cell {cell}")

# 4. Global Legend
# We take the handles/labels from the very last subplot processed
handles = [fill, line_pred, line_true]
labels = ["95% CI", "Estimate", "Reference"]

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.05),  # Adjust this to sit below the plots
    frameon=True,
)

plt.tight_layout()
plt.show()

# %%
soh_test_cells_rmse_mean = soh_test_cells_rmse.mean()
soh_test_cells_rmse_median = soh_test_cells_rmse.median()
fig, ax = create_figure("SINGLE")

ax.hist(soh_test_cells_rmse, label="Test Cells' RMSE", color="C0")
ax.axvline(
    soh_test_cells_rmse.mean(),
    label=f"Mean: {soh_test_cells_rmse_mean:.2f}",
    color="C1",
)
ax.axvline(
    soh_test_cells_rmse.median(),
    label=f"Median: {soh_test_cells_rmse_median:.2f}",
    color="C2",
    linestyle="--",
)
ax.set_title("Test cells RMSE distribution (SOH)")
ax.legend()
# %%
soh_test_cells_r2_mean = soh_test_cells_r2.mean()
soh_test_cells_r2_median = soh_test_cells_r2.median()
fig, ax = create_figure("SINGLE")

ax.hist(soh_test_cells_r2, label="Test Cells' R2", color="C0")
ax.axvline(
    soh_test_cells_r2.mean(),
    label=f"Mean: {soh_test_cells_r2_mean:.2f}",
    color="C1",
)
ax.axvline(
    soh_test_cells_r2.median(),
    label=f"Median: {soh_test_cells_r2_median:.2f}",
    color="C2",
    linestyle="--",
)
ax.set_title("Test cells R2 distribution (SOH)")
ax.legend()

# %%
rul_test_cells_rmse_mean = rul_test_cells_rmse.mean()
rul_test_cells_rmse_median = rul_test_cells_rmse.median()
fig, ax = create_figure("SINGLE")

ax.hist(rul_test_cells_rmse, label="Test Cells' RMSE", color="C0")
ax.axvline(
    rul_test_cells_rmse.mean(),
    label=f"Mean: {rul_test_cells_rmse_mean:.2f}",
    color="C1",
)
ax.axvline(
    rul_test_cells_rmse.median(),
    label=f"Median: {rul_test_cells_rmse_median:.2f}",
    color="C2",
    linestyle="--",
)
ax.set_title("Test cells RMSE distribution (RUL)")
ax.legend()
# %%
rul_test_cells_r2_mean = rul_test_cells_r2.mean()
rul_test_cells_r2_median = rul_test_cells_r2.median()
fig, ax = create_figure("SINGLE")

ax.hist(rul_test_cells_r2, label="Test Cells' R2", color="C0")
ax.axvline(
    rul_test_cells_r2.mean(),
    label=f"Mean: {rul_test_cells_r2_mean:.2f}",
    color="C1",
)
ax.axvline(
    rul_test_cells_r2.median(),
    label=f"Median: {rul_test_cells_r2_median:.2f}",
    color="C2",
    linestyle="--",
)
ax.set_title("Test cells R2 distribution (RUL)")
ax.legend()

# %%
soh_final_model_preds["ci_size"] = (
    soh_final_model_preds["y_pred_hi"] - soh_final_model_preds["y_pred_lo"]
)

fig, ax = create_figure("ONE_AND_A_HALF")
ax.plot(soh_final_model_preds["y_true"], soh_final_model_preds["ci_size"])

# %%
