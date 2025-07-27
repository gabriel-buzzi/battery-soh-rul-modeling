"""Load and organize data from Severson et. al."""

import logging
from typing import Any

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig  # OmegaConf currently unused

logger = logging.getLogger(__name__)


def load_matlab_batch(filename: str, batch_prefix: str) -> dict[str, Any]:
    """
    Load a single batch from MATLAB file and extract relevant data.

    Parameters
    ----------
    filename : str
        Path to the MATLAB file
    batch_prefix : str
        Prefix for battery keys (e.g., 'b1c', 'b2c', 'b3c')

    Returns
    -------
    dict
        Dictionary with battery data
    """
    logger.info(f"Loading {filename}...")

    with h5py.File(filename, "r") as f:
        batch = f["batch"]
        num_cells = batch["summary"].shape[0]
        bat_dict = {}

        for i in range(num_cells):
            # Extract cycle life and charge policy
            cl = f[batch["cycle_life"][i, 0]][()]
            policy = (
                f[batch["policy_readable"][i, 0]][:].tobytes()[::2].decode()
            )

            # Extract cycles data
            cycles = f[batch["cycles"][i, 0]]
            cycle_dict = {}

            for j in range(cycles["I"].shape[0]):
                # Extract only the required variables
                I = np.hstack(f[cycles["I"][j, 0]][()])  # noqa: E741 # Current
                Qc = np.hstack(f[cycles["Qc"][j, 0]][()])  # Charge capacity
                Qd = np.hstack(f[cycles["Qd"][j, 0]][()])  # Discharge capacity
                T = np.hstack(f[cycles["T"][j, 0]][()])  # Temperature
                V = np.hstack(f[cycles["V"][j, 0]][()])  # Voltage
                t = np.hstack(f[cycles["t"][j, 0]][()])  # Time

                cycle_data = {
                    "time": t,
                    "voltage": V,
                    "current": I,
                    "temperature": T,
                    "charge_capacity": Qc,
                    "discharge_capacity": Qd,
                }
                cycle_dict[j] = (
                    cycle_data  # Use integer keys for proper ordering
                )

            cell_dict = {
                "cycle_life": cl,
                "charge_policy": policy,
                "cycles": cycle_dict,
            }

            key = batch_prefix + str(i)
            bat_dict[key] = cell_dict

    return bat_dict


def apply_batch_filters(batch1: dict, batch2: dict, batch3: dict) -> tuple:
    """
    Apply the same filtering logic as in the original code.

    Returns
    -------
    tuple
        Filtered batch dictionaries
    """
    logger.info("Applying filters...")

    # Remove batteries from batch1 that do not reach 80% capacity
    batch1_remove = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
    for key in batch1_remove:
        if key in batch1:
            del batch1[key]

    # Remove noisy channels from batch3
    batch3_remove = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]
    for key in batch3_remove:
        if key in batch3:
            del batch3[key]

    return batch1, batch2, batch3


def merge_batch_continuation_data(batch1: dict, batch2: dict) -> dict:
    """Merge continuation data from batch2 into batch1 cells.

    This handles the special case where some batch1 cells continued in batch2.
    """
    logger.info("Merging batch continuation data...")

    # Mapping of batch2 keys to batch1 keys and additional cycle lengths
    batch2_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
    batch1_keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
    add_len = [662, 981, 1060, 208, 482]

    for i, (b1_key, b2_key) in enumerate(zip(batch1_keys, batch2_keys)):
        if b1_key in batch1 and b2_key in batch2:
            # Update cycle life
            batch1[b1_key]["cycle_life"] = (
                batch1[b1_key]["cycle_life"] + add_len[i]
            )

            # Get the last cycle number from batch1
            last_cycle = max(batch1[b1_key]["cycles"].keys()) + 1

            # Merge cycles from batch2, maintaining proper order
            for j, cycle_data in batch2[b2_key]["cycles"].items():
                batch1[b1_key]["cycles"][last_cycle + j] = cycle_data

    # Remove the merged cells from batch2
    for key in batch2_keys:
        if key in batch2:
            del batch2[key]

    return batch1, batch2


def save_to_hdf5(all_batteries: dict, output_filename: str):
    """Save all battery data to a single HDF5 file.

    Parameters
    ----------
    all_batteries : dict
        Combined dictionary of all battery data
    output_filename : str
        Output HDF5 filename
    """
    logger.info(f"Saving data to {output_filename}...")

    with h5py.File(output_filename, "w") as f:
        # Create main batteries group
        batteries_group = f.create_group("batteries")

        for battery_id, battery_data in all_batteries.items():
            # Create group for each battery
            battery_group = batteries_group.create_group(battery_id)

            # Save battery metadata
            battery_group.attrs["cycle_life"] = battery_data["cycle_life"]
            battery_group.attrs["charge_policy"] = battery_data[
                "charge_policy"
            ]

            # Create cycles group
            cycles_group = battery_group.create_group("cycles")

            # Save each cycle
            for cycle_num, cycle_data in battery_data["cycles"].items():
                cycle_group = cycles_group.create_group(
                    f"cycle_{cycle_num:04d}"
                )

                # Save cycle data
                for var_name, var_data in cycle_data.items():
                    cycle_group.create_dataset(
                        var_name, data=var_data, compression="gzip"
                    )

        # Save dataset metadata
        f.attrs["description"] = "Battery dataset with cycles data"
        f.attrs["variables"] = [
            "time",
            "voltage",
            "current",
            "temperature",
            "charge_capacity",
            "discharge_capacity",
        ]
        f.attrs["total_batteries"] = len(all_batteries)


def load_from_hdf5(filename: str) -> dict:
    """
    Load battery data from HDF5 file (utility function for verification).

    Parameters
    ----------
    filename : str
        HDF5 filename to load

    Returns
    -------
    dict
        Dictionary with battery data
    """
    batteries = {}

    with h5py.File(filename, "r") as f:
        batteries_group = f["batteries"]

        for battery_id in batteries_group.keys():
            battery_group = batteries_group[battery_id]

            # Load metadata
            cycle_life = battery_group.attrs["cycle_life"]
            charge_policy = battery_group.attrs["charge_policy"]

            # Load cycles
            cycles = {}
            cycles_group = battery_group["cycles"]

            for cycle_name in cycles_group.keys():
                cycle_num = int(cycle_name.split("_")[1])
                cycle_group = cycles_group[cycle_name]

                cycle_data = {}
                for var_name in cycle_group.keys():
                    cycle_data[var_name] = cycle_group[var_name][:]

                cycles[cycle_num] = cycle_data

            batteries[battery_id] = {
                "cycle_life": cycle_life,
                "charge_policy": charge_policy,
                "cycles": cycles,
            }

    return batteries


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def load_data(cfg: DictConfig) -> None:
    """Load batches MATLAB files provided and build a HDF5 databased.

    This function should load the MATLAB of three batches and reproduce the
    reorganization from Severson et. al.. The organized data is saved as a
    HDF5 dataset.
    """
    # File paths (update these to match your file locations)
    batch1_file = cfg["data"]["external_data_paths"][0]
    batch2_file = cfg["data"]["external_data_paths"][1]
    batch3_file = cfg["data"]["external_data_paths"][2]
    output_file = cfg["data"]["loaded_data_path"]

    # Load all batches
    batch1 = load_matlab_batch(batch1_file, "b1c")
    batch2 = load_matlab_batch(batch2_file, "b2c")
    batch3 = load_matlab_batch(batch3_file, "b3c")

    # Remove faulty cells
    batch1, batch2, batch3 = apply_batch_filters(batch1, batch2, batch3)

    # Merge continuation data
    batch1, batch2 = merge_batch_continuation_data(batch1, batch2)

    # Combine all batches
    all_batteries = {**batch1, **batch2, **batch3}

    # Save to HDF5
    save_to_hdf5(all_batteries, output_file)

    # Print summary statistics
    logger.info("\nProcessing complete!")
    logger.info(f"Total batteries: {len(all_batteries)}")
    logger.info(f"Batch 1: {len(batch1)} batteries")
    logger.info(f"Batch 2: {len(batch2)} batteries")
    logger.info(f"Batch 3: {len(batch3)} batteries")
    logger.info(f"Data saved to: {output_file}")


if __name__ == "__main__":
    load_data()
