"""Feature importances using random forest regressor."""

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
import joblib
from omegaconf import DictConfig
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _get_model_params(
    model_name: str,
    trial: optuna.Trial,
) -> dict[str, Any]:
    if model_name == "ElasticNet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "max_iter": 5000,
        }
    elif model_name == "KNeighborsRegressor":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
            "p": trial.suggest_int("p", 1, 2),  # 1 = Manhattan, 2 = Euclidean
        }
    elif model_name == "ExtraTreesRegressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "criterion": trial.suggest_categorical(
                "criterion",
                ["squared_error", "absolute_error", "friedman_mse"],
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    elif model_name == "XGBRegressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),  # L1 reg
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True
            ),  # L2 reg
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "dart"]
            ),
            "tree_method": trial.suggest_categorical(
                "tree_method", ["auto", "exact", "hist"]
            ),
        }
    else:
        return None


class EarlyStoppingCallback:
    """Early stopping callback for hyperparam optimization."""

    def __init__(self, patience: int):
        self.patience = patience
        self._best_value = float("inf")
        self._no_improvement_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Class caller to evalute improvement."""
        if (
            study.best_value < self._best_value - 1e-8
        ):  # significant improvement
            self._best_value = study.best_value
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            print(
                f"Early stopping triggered after {self.patience} "
                "trials with no improvement."
            )
            study.stop()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def compute_features_importances(cfg: DictConfig) -> None:
    """Compute feature importance for both targets (RUL and SOH)."""
    train_data_path = Path(cfg["data"]["train_data_path"])
    train_df = pd.read_parquet(train_data_path)

    logger.info(
        f"Loaded train data from {train_data_path} with {train_df.shape} shape"
    )

    feature_importances_dir = Path(
        cfg["modeling"]["feature_importances_dir"]
    ).absolute()

    target = cfg["modeling"]["target"]

    feature_importances_path = feature_importances_dir / f"{target}.json"

    if not feature_importances_path.exists():
        raise Exception(
            f"Feature importances for {target} not found at "
            f"{feature_importances_path}. "
            "Please run the src/modeling/feature_importances.py"
        )

    with open(feature_importances_path, "r") as f:
        feature_importances = json.load(f)

    features_sorted_by_importance = sorted(
        feature_importances, key=feature_importances.get, reverse=True
    )

    selected_features = features_sorted_by_importance[
        : cfg["modeling"]["num_features"]
    ]

    X_train = train_df[selected_features]
    y_train = train_df[target]
    cells = train_df["cell"]

    gkf = GroupKFold(n_splits=5)

    model_partial = cfg["model"]
    model_name = model_partial["_target_"].split(".")[-1]

    logger.info(f"Starting optimization of {model_name} for {target}.")

    trial_scores = []

    def objective(trial: optuna.Trial):
        model_params = _get_model_params(model_name, trial)

        # Instantiate the model using Hydra
        model = instantiate(model_partial, **model_params)

        model = make_pipeline(StandardScaler(), model())

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=gkf,
            groups=cells,
            scoring="neg_root_mean_squared_error",
        )
        rmse = -scores.mean()

        # Log trial info
        trial_scores.append(
            {"trial": trial.number, "rmse": rmse, **trial.params}
        )

        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["random_seed"]),
    )
    study.optimize(
        objective, n_trials=cfg["modeling"]["max_trials"], timeout=300
    )

    # Save trial results
    df_scores = pd.DataFrame(trial_scores)
    optimization_results_dir = Path(
        cfg["modeling"]["optimization_results_dir"]
    )
    optimization_results_dir /= f"{model_name}_{target}_optimization"
    optimization_results_dir.mkdir(parents=True, exist_ok=True)

    optimization_history_path = optimization_results_dir / "history.csv"
    df_scores.to_csv(optimization_history_path, index=False)
    logger.info(f"Saved optimization history to {optimization_history_path}")

    # Train final model with best params
    logger.info("Retriaining best model")
    best_params = study.best_trial.params
    best_model = instantiate(model_partial, **best_params)
    final_model = make_pipeline(StandardScaler(), best_model())
    final_model.fit(X_train, y_train)
    optimized_model_path = optimization_results_dir / "best_model.joblib"
    joblib.dump(final_model, optimized_model_path)
    logger.info("Saved best model to ")


if __name__ == "__main__":
    compute_features_importances()
