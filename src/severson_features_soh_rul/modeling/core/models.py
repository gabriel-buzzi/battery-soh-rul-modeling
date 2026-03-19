"""Model-builder helpers for the prediction-first pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesRegressor


def build_model(
    model_name: str,
    model_params: dict[str, Any] | None,
    random_seed: int,
    n_jobs: int,
) -> ExtraTreesRegressor:
    """Build configured model instance.

    Parameters
    ----------
    model_name : str
        Supported model identifier.
    model_params : dict[str, Any] | None
        Optional hyperparameters.
    random_seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of worker processes.

    Returns
    -------
    ExtraTreesRegressor
        Instantiated estimator.
    """
    normalized = str(model_name).strip().lower()
    if normalized != "extratrees":
        raise ValueError(
            "Unsupported model_name='{}'. Supported: ['extratrees']".format(
                model_name
            )
        )

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
    }
    if model_params:
        params.update(model_params)
    params["random_state"] = int(random_seed)
    params["n_jobs"] = int(n_jobs)
    return ExtraTreesRegressor(**params)
