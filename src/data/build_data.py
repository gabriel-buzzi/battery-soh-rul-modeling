"""Module to validate, interpolate, and export battery cell cycle data."""

import logging
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig  # OmegaConf currently unused
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _validate_cycle_data(cfg: DictConfig, cycle_data: h5py.Group) -> bool:
    """Validate the structure and content of a cycle data group.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object with validation thresholds.
    cycle_data : h5py.Group
        HDF5 group containing cycle data.

    Returns
    -------
    bool
        True if data meets all structural and numerical conditions,
        False otherwise.
    """
    required_fields = ["t", "V", "I", "Qc", "Qd"]

    for field in required_fields:
        if field not in cycle_data:
            return False

    data_length = len(cycle_data["t"][:])
    min_len = cfg["data"]["min_cycle_length"]
    max_len = cfg["data"]["max_cycle_length"]

    if data_length < min_len or data_length > max_len:
        return False

    max_capacity = max(
        np.max(cycle_data["Qc"][:]),
        np.max(cycle_data["Qd"][:]),
    )
    rated = cfg["data"]["cells_rated_capacity"]
    margin = cfg["data"]["margin_for_upper_capacity_bound"]

    if max_capacity > rated * margin:
        return False

    return True


def _fix_time_axis(
    time: np.ndarray,
    jump_indices: np.ndarray,
    time_diff: np.ndarray,
) -> np.ndarray:
    """Correct discontinuities in time axis caused by time jumps.

    Parameters
    ----------
    time : np.ndarray
        Original time vector (in seconds).
    jump_indices : np.ndarray
        Indices where time discontinuities occur.
    time_diff : np.ndarray
        First-order difference of the time vector.

    Returns
    -------
    np.ndarray
        Corrected time vector, or None if monotonicity cannot be restored.
    """
    logger.warning("Detected %d time jump(s) in cycle data", len(jump_indices))
    corrected_time = time.copy()
    cumulative_shift = 0.0

    for idx in jump_indices:
        jump_size = time_diff[idx]
        corrected_time[idx + 1 :] -= (
            jump_size - time_diff[idx - 1] if idx > 0 else jump_size
        )
        cumulative_shift += jump_size
        logger.debug(
            "Corrected time jump at index %d: %.2f minutes",
            idx,
            jump_size / 60,
        )

    if not np.all(np.diff(corrected_time) >= 0):
        logger.warning("Corrected time is not monotonically increasing")
        return None

    logger.info(
        "Corrected total time shift: %.2f minutes", cumulative_shift / 60
    )
    return corrected_time


def _process_cells_data(cfg: DictConfig, cells_data: h5py.File) -> None:
    """Process each cell in the HDF5 file.

    Validate, interpolate, and export to Parquet.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    cells_data : h5py.File
        Opened HDF5 file with battery cell data.
    """
    for cell_id in tqdm(cells_data.keys(), desc="Processing cells"):
        cell_group = cells_data[cell_id]

        try:
            charge_policy = cell_group.attrs["charge_policy"]
        except KeyError:
            logger.warning(
                "Missing charge_policy for cell %s. Skipping cell.", cell_id
            )
            continue

        cell_frames = []

        cycles_group = cell_group["cycles"]
        for cycle_id in cycles_group.keys():
            cycle_data = cycles_group[cycle_id]

            if not _validate_cycle_data(cfg, cycle_data):
                logger.warning(
                    "Invalid cycle %s of cell %s! Skipping.", cycle_id, cell_id
                )
                continue

            time = cycle_data["t"][:] * 60  # convert to seconds
            voltage = cycle_data["V"][:]
            current = cycle_data["I"][:]
            temperature = cycle_data["T"][:]
            charge_capacity = cycle_data["Qc"][:]
            discharge_capacity = cycle_data["Qd"][:]

            cells_rated_capacity = cfg["data"]["cells_rated_capacity"]
            soh = (max(discharge_capacity) / cells_rated_capacity) * 100

            time_diff = np.diff(time)
            threshold = cfg["data"]["time_jump_threshold"]
            jump_indices = np.where(time_diff > threshold)[0]

            if len(jump_indices) > 0:
                time = _fix_time_axis(time, jump_indices, time_diff)

            if time is None:
                logger.warning(
                    "Skipping cycle %s from cell %s.", cycle_id, cell_id
                )
                continue

            freq = cfg["data"]["target_frequency"]
            interp_time = np.arange(0, time[-1], 1 / freq)

            interp_voltage = np.interp(interp_time, time, voltage)
            interp_current = np.interp(interp_time, time, current)
            interp_temperature = np.interp(interp_time, time, temperature)
            interp_charge_capacity = np.interp(
                interp_time, time, charge_capacity
            )
            interp_discharge_capacity = np.interp(
                interp_time, time, discharge_capacity
            )

            cycle_df = pd.DataFrame(
                {
                    "t": interp_time,
                    "V": interp_voltage,
                    "I": interp_current,
                    "T": interp_temperature,
                    "Qc": interp_charge_capacity,
                    "Qd": interp_discharge_capacity,
                }
            )

            cycle_df["cell"] = cell_id
            cycle_df["cycle"] = int(cycle_id)
            cycle_df["SOH"] = soh
            cycle_df["charge_policy"] = charge_policy

            cell_frames.append(cycle_df)

        if cell_frames:
            full_df = pd.concat(cell_frames)
            output_dir = Path(cfg["data"]["processed_cells_data_folder"])
            output_dir.mkdir(parents=True, exist_ok=True)
            full_df.to_parquet(
                output_dir / f"{cell_id}.parquet",
                engine="pyarrow",
                compression="snappy",
                index=False,
            )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def build_data(cfg: DictConfig) -> None:
    """Load raw data, filter invalide cycles, interpolate signals and save.

    This function will process each cycle from each cell, removing invalid
    cycles, interpolating battery signals to uniform sample rate and
    saving on .parquet file for each cell.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object with I/O paths and processing options.

    Raises
    ------
    FileNotFoundError
        If the HDF5 file specified in the configuration does not exist.
    ValueError
        If the file cannot be opened or parsed.
    """
    logger.info("Processing external loaded data...")

    loaded_path = Path(cfg["data"]["loaded_data_path"])
    if not loaded_path.exists():
        raise FileNotFoundError(f"Loaded data file not found: {loaded_path}")

    try:
        with h5py.File(loaded_path, "r") as cells_data:
            _process_cells_data(cfg, cells_data)
    except Exception as e:
        raise ValueError(f"Error reading HDF5 file {loaded_path}: {str(e)}")


if __name__ == "__main__":
    build_data()
