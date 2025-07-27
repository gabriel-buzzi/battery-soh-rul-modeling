"""Util functions for data analysis."""

from matplotlib import pyplot as plt


def mm_to_inches(mm):
    """Convert millimeters to inches."""
    return mm / 25.4


def get_elsevier_single_column_fig(height_mm=85):
    """Return a single-column Elsevier matplotlib figure.

    Return a matplotlib figure and axis sized for
    a single-column Elsevier figure.

    Parameters
    ----------
    height_mm : float
        Height of the figure in millimeters (default: 85 mm)

    Returns
    -------
        fig, ax
    """
    width_mm = 85  # Single-column width
    fig, ax = plt.subplots(
        figsize=(mm_to_inches(width_mm), mm_to_inches(height_mm)),
        constrained_layout=True,
    )
    ax.tick_params(axis="both", pad=4)
    return fig, ax


def get_elsevier_double_column_fig(height_mm=85):
    """Return a double-column Elsevier matplotlib figure.

    Return a matplotlib figure and axis sized for
    a double-column Elsevier figure.

    Parameters
    ----------
    height_mm : float
        Height of the figure in millimeters (default: 85 mm)

    Returns
    -------
        fig, ax
    """
    width_mm = 170  # Double-column width
    fig, ax = plt.subplots(
        figsize=(mm_to_inches(width_mm), mm_to_inches(height_mm)),
        constrained_layout=True,
    )
    ax.tick_params(axis="both", pad=4)
    return fig, ax


def get_elsevier_figure_with_subplots(
    n_rows,
    n_cols,
    column="single",
    subplot_aspect_ratio=3 / 4,
    spacing_factor=1.1,
):
    """Return a figure and axes grid for Elsevier-style multi-subplot figures.

    Parameters
    ----------
        n_rows (int): Number of subplot rows
        n_cols (int): Number of subplot columns
        column (str): 'single' (85mm) or 'double' (170mm)
        subplot_aspect_ratio (float): Height/width for each subplot
        spacing_factor (float): Multiplier to add space between subplots

    Returns
    -------
        fig, axes
    """
    if column == "single":
        total_width_mm = 85
    elif column == "double":
        total_width_mm = 170
    else:
        raise ValueError("column must be 'single' or 'double'")

    total_width_in = mm_to_inches(total_width_mm)

    # Compute height based on subplot layout and spacing
    subplot_width = total_width_in / n_cols
    subplot_height = subplot_width * subplot_aspect_ratio
    total_height_in = subplot_height * n_rows * spacing_factor

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(total_width_in, total_height_in),
        constrained_layout=True,
    )

    return fig, axes
