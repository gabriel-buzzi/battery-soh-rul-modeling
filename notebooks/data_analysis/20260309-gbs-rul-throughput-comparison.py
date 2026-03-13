"""Quick notebook showing the difference between RUL in cycles vs. Ah."""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# %%
# Resolve repository root and input path.
repo_root = Path.cwd()
if not (repo_root / "notebooks").exists():
    repo_root = Path(__file__).resolve().parents[2]

features_path = repo_root / "data" / "interim" / "features.parquet"
print(f"Loading features from: {features_path}")

if not features_path.exists():
    raise FileNotFoundError(
        f"Features file not found at {features_path}. "
        "Run feature extraction first (src/data/make_features.py)."
    )

features_df = pd.read_parquet(features_path)
features_df.head()

# %%
required_cols = {"cell", "cycle", "RUL", "RUL_THROUGHPUT"}
missing_cols = required_cols - set(features_df.columns)
if missing_cols:
    raise ValueError(
        f"Missing required columns for comparison plot: {sorted(missing_cols)}"
    )

# Set to a specific cell id if desired, e.g. "b2c32".
example_cell = "b1c23"

available_cells = sorted(features_df["cell"].astype(str).unique().tolist())
if not available_cells:
    raise ValueError("No cells found in features dataset.")

if example_cell is None:
    example_cell = available_cells[0]

cell_df = (
    features_df[features_df["cell"].astype(str) == str(example_cell)]
    .sort_values("cycle")
    .copy()
)

if cell_df.empty:
    raise ValueError(f"Cell {example_cell} not found in features dataset.")

print(f"Example cell: {example_cell}")
print(f"Number of cycles: {cell_df.shape[0]}")
cell_df[["cell", "cycle", "RUL", "RUL_THROUGHPUT"]].head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax1 = axes[0]
ax2 = ax1.twinx()

# Raw targets (different units).
(l1,) = ax1.plot(
    cell_df["cycle"],
    cell_df["RUL"],
    label="RUL (cycles)",
    linewidth=2,
    c="C0",
)
(l2,) = ax2.plot(
    cell_df["cycle"],
    cell_df["RUL_THROUGHPUT"],
    label="RUL throughput (Ah)",
    linewidth=2,
    c="C1",
)
ax1.set_title(f"Raw Targets - Cell {example_cell}")
ax1.set_xlabel("Cycle")
ax1.set_ylabel("RUL (Cycles)")
ax2.set_ylabel("RUL throughput (Ah)")
ax1.grid(alpha=0.3)
ax1.legend(handles=[l1, l2])

# Normalized comparison (shape only).
rul_norm = cell_df["RUL"] / cell_df["RUL"].max()
rul_through_norm = cell_df["RUL_THROUGHPUT"] / cell_df["RUL_THROUGHPUT"].max()

axes[1].plot(
    cell_df["cycle"],
    rul_norm,
    label="RUL normalized",
    linewidth=2,
    c="C0",
)
axes[1].plot(
    cell_df["cycle"],
    rul_through_norm,
    label="RUL_THROUGHPUT normalized",
    linewidth=2,
    c="C1",
)
axes[1].set_title(f"Normalized Targets - Cell {example_cell}")
axes[1].set_xlabel("Cycle")
axes[1].set_ylabel("Normalized remaining life")
axes[1].grid(alpha=0.3)
axes[1].legend()

fig.tight_layout()
plt.show()

# %%
