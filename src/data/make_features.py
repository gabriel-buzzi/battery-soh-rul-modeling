"""Extract features from signals of each cycle."""

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from scipy.stats import differential_entropy, iqr, kurtosis
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _extract_cycle_features(
    cell_id: str, cycle_id: str, cycle_data: pd.DataFrame
) -> dict[str, float | int | str]:
    """
    Extract statistical features from a single cycle's data.

    Parameters
    ----------
    cell_id : str
        Unique identifier for the battery cell.
    cycle_id : str
        Unique identifier for the cycle.
    cycle_data : pd.DataFrame
        Processed cycle data.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing extracted features or None if extraction fails.
    """
    voltage = cycle_data["V"].values
    current = cycle_data["I"].values
    temperature = cycle_data["T"].values

    metrics = {
        "V_mean": np.mean(voltage).astype("float"),
        "V_median": np.median(voltage).astype("float"),
        "V_std": np.std(voltage).astype("float"),
        "V_iqr": iqr(voltage).astype("float"),
        "V_kurtosis": kurtosis(voltage).astype("float"),
        "V_entropy": differential_entropy(voltage).astype("float"),
        "I_mean": np.mean(current).astype("float"),
        "I_median": np.median(current).astype("float"),
        "I_std": np.std(current).astype("float"),
        "I_iqr": iqr(current).astype("float"),
        "I_kurtosis": kurtosis(current).astype("float"),
        "T_mean": np.mean(temperature).astype("float"),
        "T_median": np.median(temperature).astype("float"),
        "T_std": np.std(temperature).astype("float"),
        "T_iqr": iqr(temperature).astype("float"),
        "T_kurtosis": kurtosis(temperature).astype("float"),
    }

    # Extract metadata
    soh = cycle_data["SOH"].iloc[0]

    # Compute features for each voltage signal
    features = {"cell": cell_id, "cycle": cycle_id, "SOH": soh, **metrics}

    return features


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def make_features(cfg: DictConfig) -> None:
    """Compute statistical features from voltage, current and temperature."""
    cells_data_folder = Path(cfg["data"]["processed_cells_data_folder"])

    cell_data_files = list(cells_data_folder.glob("*.parquet"))

    features = []

    for cell_file in tqdm(cell_data_files, desc="Extracting features"):
        try:
            cell_data = pd.read_parquet(cell_file)
        except Exception as e:
            raise ValueError(
                f"Error reading cell data from {cell_file}: {str(e)}"
            )

        cell_features = []

        for (cell_id, cycle_id), cycle_data in cell_data.groupby(
            ["cell", "cycle"]
        ):
            try:
                cycle_features = _extract_cycle_features(
                    cell_id, cycle_id, cycle_data
                )
                if cycle_features:
                    cell_features.append(cycle_features)
            except Exception as e:
                logger.warning(
                    f"Error extracting features for cell {cell_id},"
                    f"cycle {cycle_id}: {str(e)}"
                )
                continue

        cell_df = pd.DataFrame(cell_features)

        # Calculate RUL for each cycle of the cell
        eol_soh = cfg["data"]["eol_definition"]
        eol_idx = (cell_df["SOH"] - eol_soh).abs().idxmin()
        eol_cycle = cell_df.loc[eol_idx, "cycle"]
        cell_df["RUL"] = eol_cycle - cell_df["cycle"]

        # Ensure cycle beyond EoL are removed
        # Last cycle should be the EoL it self with zero RUL
        cell_df = cell_df.iloc[: eol_idx + 1]

        features.append(cell_df)

    features_df = pd.concat(features)

    features_data_path = Path(cfg["data"]["features_data_path"])
    features_data_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(
        features_data_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )


if __name__ == "__main__":
    make_features()
