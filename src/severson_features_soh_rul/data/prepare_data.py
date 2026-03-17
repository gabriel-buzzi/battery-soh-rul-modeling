"""Split train and test data and process train data."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def prepare_data(cfg: DictConfig) -> None:
    """Prepare train and test data."""
    features_data_path = Path(cfg["data"]["features_data_path"])
    features_df = pd.read_parquet(features_data_path)
    logger.info(
        f"Loaded features data from {features_data_path}"
        f"with {features_df.shape} shape."
    )

    # Drop lines with NaN values
    logger.info("Dropping lines with NaNs")
    features_df.dropna(axis=0, inplace=True)
    logger.info(f"Features data shape after removing NaNs:{features_df.shape}")

    logger.info("Spliting train and test cells")
    cells = list(features_df["cell"].unique())
    train_cells, test_cells, _, _ = train_test_split(
        cells,
        cells,
        train_size=cfg["data"]["train_cells_proportion"],
        random_state=cfg["random_seed"],
    )

    logger.info(f"Train cells: {train_cells}\nTest cells: {test_cells}")

    train_df = features_df[features_df["cell"].isin(train_cells)]
    test_df = features_df[features_df["cell"].isin(test_cells)]

    feature_cols = [
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

    logger.info("Removing spikes from training data.")
    # Remove spikes with moving median filter.
    for cell_id, cell_data in train_df.groupby("cell"):
        train_df.loc[train_df["cell"] == cell_id, feature_cols] = (
            cell_data[feature_cols]
            .rolling(window=cfg["data"]["features_process_window_size"])
            .median()
            .bfill()
        )

    logger.info("Smmothing training data features.")
    # Smooth signal with Sav. Gol. filter.
    for cell_id, cell_data in train_df.groupby("cell"):
        train_df.loc[train_df["cell"] == cell_id, feature_cols] = cell_data[
            feature_cols
        ].apply(
            lambda feature: savgol_filter(
                feature,
                window_length=cfg["data"]["features_process_window_size"],
                polyorder=cfg["data"]["features_process_savgol_order"],
            )
        )

    # Drop eventual NaNs introduced by processing.
    train_df.dropna(axis=0, inplace=True)

    train_data_path = Path(cfg["data"]["train_data_path"])
    train_data_path.parent.mkdir(parents=True, exist_ok=True)
    test_data_path = Path(cfg["data"]["test_data_path"])
    test_data_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving prepared train and test data")
    train_df.to_parquet(
        train_data_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    test_df.to_parquet(
        test_data_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    logger.info(f"Data saved at {train_data_path} and {test_data_path}")


if __name__ == "__main__":
    prepare_data()
