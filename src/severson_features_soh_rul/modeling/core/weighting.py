"""Training-weight helpers for weighting-aware stage execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from severson_features_soh_rul.modeling.config.schema import WeightingConfig


def build_sample_weights(
    y_train: pd.Series,
    weighting_cfg: WeightingConfig,
    reference_series: pd.Series | None = None,
) -> np.ndarray | None:
    """Build per-sample weights according to configured strategy.

    Parameters
    ----------
    y_train : pd.Series
        Training targets for the current fold.
    weighting_cfg : WeightingConfig
        Parsed weighting configuration.
    reference_series : pd.Series | None
        Optional alternative series for weighting rules.

    Returns
    -------
    np.ndarray | None
        Sample weights array or ``None`` when weighting is disabled.
    """
    if not weighting_cfg.enabled or weighting_cfg.strategy == "none":
        return None

    values = (
        reference_series.astype(float)
        if reference_series is not None
        else y_train.astype(float)
    )

    if weighting_cfg.strategy == "sample_weight_inverse_life_density":
        return _inverse_density_weights(
            values=values,
            n_bins=weighting_cfg.n_bins,
        )

    if weighting_cfg.strategy == "sample_weight_long_life_boost":
        return _long_life_boost_weights(
            values=values,
            long_life_quantile=weighting_cfg.long_life_quantile,
            boost_factor=weighting_cfg.long_life_boost_factor,
        )

    raise ValueError(
        "Unsupported weighting strategy='{}'".format(weighting_cfg.strategy)
    )


def _inverse_density_weights(values: pd.Series, n_bins: int) -> np.ndarray:
    """Assign inverse-density weights over quantile bins."""
    clean_values = values.astype(float)
    try:
        bins = pd.qcut(
            clean_values,
            q=max(2, int(n_bins)),
            duplicates="drop",
        )
    except ValueError:
        bins = pd.Series(
            ["single_bin"] * clean_values.shape[0], index=values.index
        )

    counts = bins.value_counts(dropna=False)
    weights = bins.map(lambda value: 1.0 / float(counts.loc[value]))
    weights = weights.astype(float).to_numpy()
    mean_weight = float(np.mean(weights))
    if mean_weight <= 0:
        return np.ones_like(weights)
    return weights / mean_weight


def _long_life_boost_weights(
    values: pd.Series,
    long_life_quantile: float,
    boost_factor: float,
) -> np.ndarray:
    """Boost samples above a configured quantile threshold."""
    clean_values = values.astype(float)
    threshold = float(clean_values.quantile(float(long_life_quantile)))
    weights = np.ones(clean_values.shape[0], dtype=float)
    weights[clean_values.to_numpy() >= threshold] = float(boost_factor)
    mean_weight = float(np.mean(weights))
    return weights / mean_weight if mean_weight > 0 else weights
