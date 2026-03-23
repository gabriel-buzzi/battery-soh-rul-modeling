"""Model-builder helpers for the prediction-first pipeline."""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL_NAME = "extratrees_quantile"


def build_model(
    model_params: dict[str, Any] | None,
    random_seed: int,
    n_jobs: int,
) -> Any:
    """Build configured model instance.

    Parameters
    ----------
    model_params : dict[str, Any] | None
        Optional hyperparameters.
    random_seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of worker processes.

    Returns
    -------
    Any
        Instantiated estimator.
    """
    try:
        from quantile_forest import ExtraTreesQuantileRegressor
    except ImportError as exc:
        raise ImportError(
            "quantile-forest is required for ExtraTreesQuantileRegressor. "
            "Install dependency 'quantile-forest'."
        ) from exc

    params: dict[str, Any] = {
        "n_estimators": 300,
        "criterion": "squared_error",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": False,
        "random_state": int(random_seed),
        "n_jobs": int(n_jobs),
        # Keep point-prediction behavior in optimize/objective with median.
        "default_quantiles": 0.5,
    }
    if model_params:
        params.update(model_params)
    params["random_state"] = int(random_seed)
    params["n_jobs"] = int(n_jobs)
    params.setdefault("default_quantiles", 0.5)
    return ExtraTreesQuantileRegressor(**params)
