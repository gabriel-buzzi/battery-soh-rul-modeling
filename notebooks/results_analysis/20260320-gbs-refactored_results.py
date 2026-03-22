"""Analyse results after pipeline refactor for paper revision."""

# %%
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from severson_features_soh_rul.utils.plots import create_figure

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
RUL_NO_TEMP_ARTIFACTS = (
    BASE_PATH
    / "results_refactored"
    / "modeling"
    / "target-rul__feature_hash-82854e4bcdd504169be5162d100888a4a50cce5db7d665692a29ab5d23a5b70d__split_seed-42__model_name-extratrees__weighting_strategy-none__rk-4074892c0e"
)
SOH_NO_TEMP_ARTIFACTS = (
    BASE_PATH
    / "results_refactored"
    / "modeling"
    / "target-soh__feature_hash-82854e4bcdd504169be5162d100888a4a50cce5db7d665692a29ab5d23a5b70d__split_seed-42__model_name-extratrees__weighting_strategy-none__rk-357a381862"
)

# %% [markdown]
# ## Optimization results

# %%
soh_optimize_path = SOH_ARTIFACTS / "optimize"
rul_optimize_path = RUL_ARTIFACTS / "optimize"

# %%
soh_optimize_history = pd.read_csv(soh_optimize_path / "cv_trials.csv")
rul_optimize_history = pd.read_csv(rul_optimize_path / "cv_trials.csv")

# %%
w = 1
rul_optimize_history["new_objective"] = (
    rul_optimize_history["rmse_val_mean"]
    + w * rul_optimize_history["overfit_gap_mean"]
)

soh_optimize_history["new_objective"] = (
    soh_optimize_history["rmse_val_mean"]
    + w * soh_optimize_history["overfit_gap_mean"]
)

# %%
fig, ax = create_figure(layout_type="one_and_a_half")
soh_best_trial = soh_optimize_history.loc[
    soh_optimize_history["new_objective"].idxmin()
]
sns.scatterplot(
    x=soh_optimize_history["rmse_val_mean"],
    y=soh_optimize_history["overfit_gap_mean"],
    hue=soh_optimize_history["new_objective"],
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
    rul_optimize_history["new_objective"].idxmin()
]
sns.scatterplot(
    x=rul_optimize_history["rmse_val_mean"],
    y=rul_optimize_history["overfit_gap_mean"],
    hue=rul_optimize_history["new_objective"],
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
    soh_optimize_history["new_objective"].idxmin()
]
rul_best_trial = rul_optimize_history.loc[
    rul_optimize_history["new_objective"].idxmin()
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
