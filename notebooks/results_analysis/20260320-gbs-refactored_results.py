"""Analyse results after pipeline refactor for paper revision."""

# %%
import json
from pathlib import Path

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

# %% [markdown]
# ## Optimization results

# %%
soh_optimize_path = SOH_ARTIFACTS / "optimize"
with open(soh_optimize_path / "best_params.json") as f:
    soh_best_params = json.load(f)

print(soh_best_params)

rul_optimize_path = RUL_ARTIFACTS / "optimize"
with open(rul_optimize_path / "best_params.json") as f:
    rul_best_params = json.load(f)

print(rul_best_params)

# %%
