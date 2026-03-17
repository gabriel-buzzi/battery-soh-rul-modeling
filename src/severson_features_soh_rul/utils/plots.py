"""Util functions for data analysis."""

from typing import Any

from matplotlib import pyplot as plt

# Widths from your LaTeX template
ELSEVIER_LAYOUT = {
    "one_column": 255,
    "one_and_half_column": 397,
    "double_column": 539,
}


def _normalize_layout(layout_type: str) -> str:
    """Normalize layout aliases to canonical layout keys."""
    normalized = str(layout_type).strip().lower().replace("-", "_")
    aliases = {
        "single": "one_column",
        "one_column": "one_column",
        "one_and_a_half": "one_and_half_column",
        "one_and_half": "one_and_half_column",
        "one_and_half_column": "one_and_half_column",
        "double": "double_column",
        "double_column": "double_column",
    }
    return aliases.get(normalized, "one_column")


def create_figure(
    layout_type: str = "one_column",
    aspect_ratio: float = 0.618,
    nrows: int = 1,
    ncols: int = 1,
) -> tuple[Any, Any]:
    """Create a matplotlib figure and axes using the configured layout."""
    # Matplotlib already loaded the matplotlibrc settings!

    width_pt = ELSEVIER_LAYOUT[_normalize_layout(layout_type)]
    width_in = width_pt / 72.27
    height_in = width_in * aspect_ratio * (nrows / ncols)

    # Just create the plot—the RC file handles the fonts/ticks/lines
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width_in, height_in),
        constrained_layout=True,
    )
    axes = ax.flat if hasattr(ax, "flat") else (ax,)
    for axis in axes:
        axis.tick_params(axis="both", pad=4)
    return fig, ax
