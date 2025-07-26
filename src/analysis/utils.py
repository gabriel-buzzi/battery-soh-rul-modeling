"""Util function for data analysis."""

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
        figsize=(mm_to_inches(width_mm), mm_to_inches(height_mm))
    )
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
        figsize=(mm_to_inches(width_mm), mm_to_inches(height_mm))
    )
    return fig, ax
