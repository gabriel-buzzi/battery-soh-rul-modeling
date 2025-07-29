"""Feature importances using random forest regressor."""

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def compute_features_importances(cfg: DictConfig) -> None:
    """Compute feature importance for both targets (RUL and SOH)."""
    train_data_path = Path(cfg["data"]["train_data_path"])
    train_df = pd.read_parquet(train_data_path)

    logger.info(
        f"Loaded train data from {train_data_path} with {train_df.shape} shape"
    )

    all_features_cols = [
        "V_mean",
        "V_median",
        "V_std",
        "V_iqr",
        "V_kurtosis",
        "V_entropy",
        "I_mean",
        "I_median",
        "I_std",
        "I_iqr",
        "I_kurtosis",
        "T_mean",
        "T_median",
        "T_std",
        "T_iqr",
        "T_kurtosis",
    ]

    feature_importances_dir = Path(cfg["modeling"]["feature_importances_dir"])
    feature_importances_dir.mkdir(parents=True, exist_ok=True)

    targets = ["SOH", "RUL"]

    for target in targets:
        feature_importances_path = feature_importances_dir / f"{target}.json"

        if feature_importances_path.exists():
            logger.info(
                f"Feature importances for {target} already found at"
                f"{feature_importances_path}. Those will be used for"
                "futher feature selection"
            )
            continue

        X_train = train_df[all_features_cols]
        y_train = train_df[target]

        logger.info(f"Computing feature importances for {target}")
        rf = RandomForestRegressor(
            n_estimators=50,
            random_state=cfg["random_seed"],
            n_jobs=cfg["modeling"]["n_jobs"],
        )
        rf.fit(X_train, y_train)

        feature_importances = dict(
            zip(all_features_cols, rf.feature_importances_)
        )

        logger.info(
            f"Saving feature importances for {target} at"
            f"{feature_importances_path}"
        )
        with open(feature_importances_path, "w") as f:
            json.dump(feature_importances, f, indent=4)


if __name__ == "__main__":
    compute_features_importances()
