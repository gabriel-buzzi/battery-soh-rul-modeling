"""Evaluate trained model selected on configuration file."""

import json
import logging
from pathlib import Path

import hydra
import joblib
from omegaconf import DictConfig
import pandas as pd

from src.modeling.utils import regression_metrics

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def evaluate_performance(cfg: DictConfig) -> None:
    """Compute feature importance for both targets (RUL and SOH)."""
    test_data_path = Path(cfg["data"]["test_data_path"])
    test_df = pd.read_parquet(test_data_path)

    logger.info(
        f"Loaded test data from {test_data_path} with {test_df.shape} shape"
    )

    model_partial = cfg["model"]
    model_name = model_partial["_target_"].split(".")[-1]

    target = cfg["modeling"]["target"]

    optimization_results_dir = Path(
        cfg["modeling"]["optimization_results_dir"]
    ).absolute()
    optimization_results_dir /= f"{model_name}_{target}_optimization"
    optimized_model_path = optimization_results_dir / "best_model.joblib"

    if not optimized_model_path.exists():
        raise Exception(
            f"Optimized model not found at {optimized_model_path}. "
            f"Confirm if you have already optimized a {model_name} model "
            f"for {target}."
        )

    loaded_model = joblib.load(optimized_model_path)

    feature_importances_dir = Path(
        cfg["modeling"]["feature_importances_dir"]
    ).absolute()

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

    X_test = test_df[selected_features]
    y_test = test_df[target]
    cells = test_df["cell"]

    logger.info(f"Starting evaluation of {model_name} for {target}.")

    y_pred = loaded_model.predict(X_test)

    evaluation_results_dir = Path(
        cfg["modeling"]["evaluation_results_dir"]
    ).absolute()
    evaluation_results_dir /= f"{model_name}_{target}_evaluation"
    evaluation_results_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        {"cell": cells, "y_true": y_test, "y_pred": y_pred}
    )

    predictions_path = evaluation_results_dir / "predictions.csv"

    predictions.to_csv(predictions_path, index=False)

    logger.info(f"Predictions saved to {predictions_path}.")

    overall_metrics = regression_metrics(y_test, y_pred)

    overall_metrics_path = evaluation_results_dir / "overall_metrics.json"
    with open(overall_metrics_path, "w") as f:
        json.dump(overall_metrics, f)

    logger.info(f"Overall metrics saved at {overall_metrics_path}.")


if __name__ == "__main__":
    evaluate_performance()
