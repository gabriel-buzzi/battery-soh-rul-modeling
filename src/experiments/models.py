"""Model factories used by experiment tracks."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesRegressor


def build_extratrees(
    params: dict[str, Any],
    random_seed: int,
    n_jobs: int,
) -> ExtraTreesRegressor:
    """Instantiate ExtraTrees regressor with deterministic seed."""
    model_params = {
        "n_estimators": 300,
        "criterion": "squared_error",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": False,
        "random_state": random_seed,
        "n_jobs": n_jobs,
    }
    model_params.update(params)
    model_params["random_state"] = random_seed
    model_params["n_jobs"] = n_jobs
    return ExtraTreesRegressor(**model_params)

