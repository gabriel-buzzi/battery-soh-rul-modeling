"""Module to load and store battery dataset from .mat files into .h5."""

import gc
import logging
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def load_data(cfg: DictConfig) -> None:
    """Load data from .mat files and resave them into a unique .h5.

    Load .mat files containing battery test data, process it,
    and save as a structured HDF5 file.

    This function:
    - Reads paths from the Hydra configuration.
    - Extracts scalar and timeseries data per cell from the .mat files.
    - Writes this data into an HDF5 file.
    - Applies filtering (e.g., removing noisy or incomplete cells).
    - Merges data from different batches.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing:
            - data.external_data_paths : list of str
                Paths to the raw .mat data files.
            - data.loaded_data_path : str
                Destination HDF5 file path for the processed data.
    """
    mat_files = cfg["data"]["external_data_paths"]
    output_file = Path(cfg["data"]["loaded_data_path"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize empty HDF5 file
    with h5py.File(output_file, "w"):
        pass

    # Process each batch
    for batch_idx, mat_filename in enumerate(mat_files, 1):
        logger.info(f"Loading {batch_idx} of {len(mat_files)}")

        with h5py.File(mat_filename, "r") as file_handle:
            batch = file_handle["batch"]
            num_cells = batch["summary"].shape[0]

            for i in tqdm(range(num_cells), desc="Cell"):
                try:
                    cycle_life = file_handle[batch["cycle_life"][i, 0]][()]
                    policy = (
                        file_handle[batch["policy_readable"][i, 0]][:]
                        .tobytes()[::2]
                        .decode()
                    )
                except KeyError as e:
                    print(
                        f"KeyError: {e} in file {mat_filename}, cell {i}."
                        "Skipping."
                    )
                    continue

                with h5py.File(output_file, "a") as out_f:
                    cell_key = f"b{batch_idx}c{i}"
                    cell_group = out_f.create_group(cell_key)
                    cell_group.attrs["cycle_life"] = cycle_life
                    cell_group.attrs["charge_policy"] = policy

                    cycles = file_handle[batch["cycles"][i, 0]]
                    cycles_group = cell_group.create_group("cycles")

                    for j in range(cycles["I"].shape[0]):
                        try:
                            time = np.hstack(
                                file_handle[cycles["t"][j, 0]][()],
                            ).astype(np.float32)
                            voltage = np.hstack(
                                file_handle[cycles["V"][j, 0]][()]
                            ).astype(np.float32)
                            current = np.hstack(
                                file_handle[cycles["I"][j, 0]][()]
                            ).astype(np.float32)
                            temperature = np.hstack(
                                file_handle[cycles["T"][j, 0]][()]
                            ).astype(np.float32)
                            charge_capacity = np.hstack(
                                file_handle[cycles["Qc"][j, 0]][()]
                            ).astype(np.float32)
                            discharge_capacity = np.hstack(
                                file_handle[cycles["Qd"][j, 0]][()]
                            ).astype(np.float32)

                            cycle_group = cycles_group.create_group(str(j))
                            cycle_group.create_dataset(
                                "t",
                                data=time,
                                compression="gzip",
                                compression_opts=4,
                            )
                            cycle_group.create_dataset(
                                "V",
                                data=voltage,
                                compression="gzip",
                                compression_opts=4,
                            )
                            cycle_group.create_dataset(
                                "I",
                                data=current,
                                compression="gzip",
                                compression_opts=4,
                            )
                            cycle_group.create_dataset(
                                "T",
                                data=temperature,
                                compression="gzip",
                                compression_opts=4,
                            )
                            cycle_group.create_dataset(
                                "Qc",
                                data=charge_capacity,
                                compression="gzip",
                                compression_opts=4,
                            )
                            cycle_group.create_dataset(
                                "Qd",
                                data=discharge_capacity,
                                compression="gzip",
                                compression_opts=4,
                            )
                        except KeyError as e:
                            print(
                                f"KeyError: {e} in file {mat_filename},"
                                "cell {i}, cycle {j}. Skipping cycle."
                            )
                            continue

                gc.collect()

    # Post-process the HDF5 file
    with h5py.File(output_file, "a") as out_file_handle:
        # Remove underperforming cells in batch 1
        for key in ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]:
            if key in out_file_handle:
                del out_file_handle[key]

        # Merge batch 2 into batch 1
        batch2_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
        batch1_keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
        additional_cycles = [662, 981, 1060, 208, 482]

        for i, (b1k, b2k) in enumerate(zip(batch1_keys, batch2_keys)):
            if b1k in out_file_handle and b2k in out_file_handle:
                out_file_handle[b1k].attrs["cycle_life"] += additional_cycles[
                    i
                ]
                last_cycle = len(out_file_handle[b1k]["cycles"])

                for j, cycle_key in enumerate(
                    out_file_handle[b2k]["cycles"].keys()
                ):
                    out_file_handle.move(
                        f"{b2k}/cycles/{cycle_key}",
                        f"{b1k}/cycles/{last_cycle + j}",
                    )

                del out_file_handle[b2k]

        # Remove noisy batch 3 cells
        for key in ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]:
            if key in out_file_handle:
                del out_file_handle[key]

    print(f"Successfully saved processed data to {output_file}")
    gc.collect()


if __name__ == "__main__":
    load_data()
