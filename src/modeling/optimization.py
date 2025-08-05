"""Perform hyperparam optimization and model training on train data."""

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
import joblib
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
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
            "n_neighbors": trial.suggest_int("n_neighbors", 10, 40),
            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
            "p": trial.suggest_int("p", 1, 2),
        }
    elif model_name == "ExtraTreesRegressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "criterion": trial.suggest_categorical(
                "criterion",
                ["squared_error", "absolute_error"],
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", None]
            ),
            "bootstrap": False,
        }

    elif model_name == "XGBRegressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 0, 2),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                1.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 1.0, log=True
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "booster": "gbtree",
            "tree_method": "hist",
        }
    elif model_name == "MLPRegressor":
        # Suggest number of layers: 1 to 3 layers
        n_layers = trial.suggest_int("n_layers", 1, 3)

        # For each layer, suggest number of neurons from a small discrete set
        # Use a tuple as hidden_layer_sizes for MLPRegressor
        hidden_layer_sizes = tuple(
            trial.suggest_categorical(f"n_units_l{i + 1}", [16, 32, 64, 128])
            for i in range(n_layers)
        )

        return {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh"]
            ),
            "solver": trial.suggest_categorical("solver", ["adam", "lbfgs"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            "max_iter": 5000,
            "early_stopping": True,
            "random_state": 42,
        }
    elif model_name == "TweedieRegressor":
        return {
            "power": trial.suggest_float(
                "power",
                1.3,
                1.8,
            ),  # 1=Poisson, 2=Gamma, 1.5~Tweedie
            "alpha": trial.suggest_float(
                "alpha",
                1e-5,
                1.0,
                log=True,
            ),  # L2 regularization
            "link": "identity",  # link function
            "fit_intercept": trial.suggest_categorical(
                "fit_intercept", [True, False]
            ),
            "max_iter": 5000,
            "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
        }
    elif model_name == "LGBMRegressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 1.0, log=True
            ),
            "verbosity": -1,
        }
    else:
        return None


# class EarlyStoppingCallback:
#     """Early stopping callback for hyperparam optimization."""

#     def __init__(self, patience: int):
#         self.patience = patience
#         self._best_value = float("inf")
#         self._no_improvement_count = 0

#     def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
#         """Class caller to evalute improvement."""
#         if (
#             study.best_value < self._best_value - 1e-8
#         ):  # significant improvement
#             self._best_value = study.best_value
#             self._no_improvement_count = 0
#         else:
#             self._no_improvement_count += 1

#         if self._no_improvement_count >= self.patience:
#             print(
#                 f"Early stopping triggered after {self.patience} "
#                 "trials with no improvement."
#             )
#             study.stop()


class MultiObjectiveEarlyStopping:
    """Multi-objective early stopping callback for hyperparam optimization.

    It tracks the Pareto front (list of study.best_trials).
    If the current Pareto front is unchanged for patience trials, it stops.
    """

    def __init__(self, patience: int = 10):
        self.patience = patience
        self._best_trials = []
        self._no_improvement_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Class caller to evalute improvement."""
        # Keep only non-dominated (Pareto optimal) trials
        current_pareto_trials = study.best_trials

        # Check if Pareto front improved
        if not self._is_same_front(current_pareto_trials):
            self._best_trials = current_pareto_trials
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        # Trigger early stopping if no improvement
        if self._no_improvement_count >= self.patience:
            print(
                "Early stopping triggered (no Pareto improvement"
                f"in {self.patience} trials)."
            )
            study.stop()

    def _is_same_front(self, trials: list[optuna.trial.FrozenTrial]) -> bool:
        def extract_values(trial_list):
            return sorted(
                [
                    tuple(trial.values)
                    for trial in trial_list
                    if trial.values is not None
                ]
            )

        old_values = extract_values(self._best_trials)
        new_values = extract_values(trials)
        return old_values == new_values


def _estimate_mlp_params(input_dim, hidden_layers):
    # input_dim: int
    # hidden_layers: tuple of int
    total_params = 0
    prev_layer = input_dim
    for units in hidden_layers:
        total_params += prev_layer * units + units  # weights + bias
        prev_layer = units
    total_params += prev_layer * 1 + 1  # last layer to output + bias
    return total_params


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def optimize_model(cfg: DictConfig) -> None:
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

    num_features = cfg["modeling"]["num_features"]

    selected_features = features_sorted_by_importance[:num_features]

    X_train = train_df[selected_features]
    y_train = train_df[target]
    cells = train_df["cell"]

    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Using features: {selected_features}")

    gkf = GroupKFold(n_splits=5)

    model_partial = cfg["model"]
    model_name = model_partial["_target_"].split(".")[-1]

    logger.info(f"Starting optimization of {model_name} for {target}.")

    trial_scores = []

    n_jobs = cfg["modeling"]["n_jobs"]

    optimization_results_dir = Path(
        cfg["modeling"]["optimization_results_dir"]
    )
    optimization_results_dir /= (
        f"{num_features}_features_{model_name}_{target}_optimization"
    )
    optimization_results_dir.mkdir(parents=True, exist_ok=True)

    trials_train_rmse = []
    trials_val_rmse = []
    minimun_rmse = []
    minimun_trials = []

    def objective(trial: optuna.Trial):
        model_params = _get_model_params(model_name, trial)
        model = instantiate(model_partial, **model_params)
        if hasattr(model, "n_jobs"):
            model_params.update("n_jobs", cfg["modeling"]["n_jobs"])
            model = instantiate(model_partial, **model_params)
        if hasattr(model, "random_state"):
            model_params.update("random_state", cfg["random_seed"])
            model = instantiate(model_partial, **model_params)
        pipeline = make_pipeline(StandardScaler(), model())

        def fit_and_score(train_idx, val_idx):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline_clone = clone(pipeline)
            pipeline_clone.fit(X_tr, y_tr)

            y_val_pred = pipeline_clone.predict(X_val)
            val_rmse = root_mean_squared_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            y_tr_pred = pipeline_clone.predict(X_tr)
            train_rmse = root_mean_squared_error(y_tr, y_tr_pred)
            train_r2 = r2_score(y_tr, y_tr_pred)

            return val_rmse, val_r2, train_rmse, train_r2

        scores = Parallel(n_jobs=n_jobs)(
            delayed(fit_and_score)(train_idx, val_idx)
            for train_idx, val_idx in gkf.split(X_train, y_train, groups=cells)
        )

        val_rmse_list, val_r2_list, train_rmse_list, train_r2_list = zip(
            *scores
        )

        val_rmse = np.mean(val_rmse_list)
        val_r2 = np.mean(val_r2_list)
        train_rmse = np.mean(train_rmse_list)
        train_r2 = np.mean(train_r2_list)

        if model_name == "MLPRegressor":
            n_params = {
                "n_model_params": _estimate_mlp_params(
                    num_features, model_params["hidden_layer_sizes"]
                )
            }
        else:
            n_params = {}

        trial_scores.append(
            {
                "trial": trial.number,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                **trial.params,
                **n_params,
            }
        )

        if trials_val_rmse:
            if val_rmse < min(trials_val_rmse):
                minimun_trials.append(trial.number)
                minimun_rmse.append(val_rmse)

        trials_train_rmse.append(train_rmse)
        trials_val_rmse.append(val_rmse)

        plt.plot(
            range(len(trials_train_rmse)),
            trials_train_rmse,
            label="Train RMSE",
        )
        plt.plot(
            range(len(trials_val_rmse)),
            trials_val_rmse,
            label="Validation RMSE",
        )
        plt.plot(minimun_trials, minimun_rmse, label="Best Trials")
        plt.xlabel("Training Size")
        plt.ylabel("RMSE")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(optimization_results_dir / "optimization_losses.png")
        plt.clf()

        return (
            val_rmse,
            abs(train_rmse - val_rmse) / val_rmse,
        )

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=cfg["random_seed"]),
    )

    early_stopping_cb = MultiObjectiveEarlyStopping(
        patience=cfg["modeling"]["early_stopping_patience"]
    )

    study.optimize(
        objective,
        n_trials=cfg["modeling"]["max_trials"],
        callbacks=[early_stopping_cb],
    )

    # Best trial balancing val_rmse and less overfitting
    best_trial = min(
        study.best_trials, key=lambda t: t.values[0] + t.values[1]
    )

    # Save trial results
    df_scores = pd.DataFrame(trial_scores)
    df_scores["is_best_trial"] = df_scores["trial"] == best_trial.number
    df_scores = df_scores.sort_values(by="is_best_trial", ascending=False)

    optimization_history_path = optimization_results_dir / "history.csv"
    df_scores.to_csv(optimization_history_path, index=False)
    logger.info(f"Saved optimization history to {optimization_history_path}")

    # Train final model with best params
    logger.info("Retriaining best model")
    best_params = best_trial.params
    best_model = instantiate(model_partial, **best_params)
    final_model = make_pipeline(StandardScaler(), best_model())
    final_model.fit(X_train, y_train)
    optimized_model_path = optimization_results_dir / "best_model.joblib"
    joblib.dump(final_model, optimized_model_path)
    logger.info("Saved best model to ")


if __name__ == "__main__":
    optimize_model()
