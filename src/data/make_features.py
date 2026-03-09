"""Extract cycle-level features and targets from processed signals."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from scipy.stats import differential_entropy, iqr, kurtosis
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _signal_metrics(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute statistical metrics for voltage/current/temperature arrays."""
    return {
        f"{prefix}V_mean": float(np.mean(voltage)),
        f"{prefix}V_median": float(np.median(voltage)),
        f"{prefix}V_std": float(np.std(voltage)),
        f"{prefix}V_iqr": float(iqr(voltage)),
        f"{prefix}V_kurtosis": float(kurtosis(voltage)),
        f"{prefix}V_entropy": float(differential_entropy(voltage)),
        f"{prefix}I_mean": float(np.mean(current)),
        f"{prefix}I_median": float(np.median(current)),
        f"{prefix}I_std": float(np.std(current)),
        f"{prefix}I_iqr": float(iqr(current)),
        f"{prefix}I_kurtosis": float(kurtosis(current)),
        f"{prefix}T_mean": float(np.mean(temperature)),
        f"{prefix}T_median": float(np.median(temperature)),
        f"{prefix}T_std": float(np.std(temperature)),
        f"{prefix}T_iqr": float(iqr(temperature)),
        f"{prefix}T_kurtosis": float(kurtosis(temperature)),
    }


def _cycle_throughput_ah(cycle_data: pd.DataFrame) -> float:
    """Integrate |I|*dt over a cycle and return throughput in Ah.

    This follows the intended cumsum(abs(dt * current)) strategy.
    """
    cycle_sorted = cycle_data.sort_values("t")
    time_s = cycle_sorted["t"].to_numpy(dtype=float)
    current_a = cycle_sorted["I"].to_numpy(dtype=float)

    if time_s.size <= 1:
        return 0.0

    dt_s = np.diff(time_s, prepend=time_s[0])
    throughput_as = np.sum(np.abs(dt_s * current_a))
    return float(throughput_as / 3600.0)


def _extract_cycle_features(
    cell_id: str,
    cycle_id: int,
    cycle_data: pd.DataFrame,
    charge_current_threshold: float,
) -> dict[str, float | int | str]:
    """Extract full-cycle and charge-only features from one cycle."""
    cycle_sorted = cycle_data.sort_values("t")
    voltage = cycle_sorted["V"].to_numpy(dtype=float)
    current = cycle_sorted["I"].to_numpy(dtype=float)
    temperature = cycle_sorted["T"].to_numpy(dtype=float)

    full_metrics = _signal_metrics(voltage, current, temperature, prefix="")

    charge_mask = current > charge_current_threshold
    if int(np.sum(charge_mask)) < 2:
        raise ValueError(
            "Charge-step samples are insufficient for feature extraction "
            f"(cell={cell_id}, cycle={cycle_id})."
        )
    charge_metrics = _signal_metrics(
        voltage[charge_mask],
        current[charge_mask],
        temperature[charge_mask],
        prefix="charge_",
    )

    soh = float(cycle_sorted["SOH"].iloc[0])
    throughput_ah_cycle = _cycle_throughput_ah(cycle_sorted)

    return {
        "cell": cell_id,
        "cycle": int(cycle_id),
        "SOH": soh,
        "throughput_ah_cycle": throughput_ah_cycle,
        **full_metrics,
        **charge_metrics,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def make_features(cfg: DictConfig) -> None:
    """Compute full-cycle/charge-step features and RUL targets."""
    cells_data_folder = Path(cfg["data"]["processed_cells_data_folder"])
    cell_data_files = list(cells_data_folder.glob("*.parquet"))

    features: list[pd.DataFrame] = []
    charge_current_threshold = float(
        cfg["data"].get("charge_current_threshold", 0.0)
    )

    for cell_file in tqdm(cell_data_files, desc="Extracting features"):
        try:
            cell_data = pd.read_parquet(cell_file)
        except Exception as exc:
            raise ValueError(
                f"Error reading cell data from {cell_file}: {str(exc)}"
            ) from exc

        cell_features: list[dict[str, float | int | str]] = []
        grouped = cell_data.groupby(["cell", "cycle"], sort=True)
        for (cell_id, cycle_id), cycle_data in grouped:
            try:
                cycle_features = _extract_cycle_features(
                    cell_id=cell_id,
                    cycle_id=int(cycle_id),
                    cycle_data=cycle_data,
                    charge_current_threshold=charge_current_threshold,
                )
                cell_features.append(cycle_features)
            except Exception as exc:
                logger.warning(
                    "Error extracting features for cell %s cycle %s: %s",
                    cell_id,
                    cycle_id,
                    str(exc),
                )

        if not cell_features:
            logger.warning("No valid cycles for cell file %s", cell_file.name)
            continue

        cell_df = (
            pd.DataFrame(cell_features)
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        # EoL defined by SOH threshold; keep cycles up to and including EoL.
        eol_soh = float(cfg["data"]["eol_definition"])
        eol_idx = int((cell_df["SOH"] - eol_soh).abs().idxmin())
        eol_cycle = int(cell_df.loc[eol_idx, "cycle"])
        cell_df = cell_df.loc[:eol_idx].copy().reset_index(drop=True)

        # Cycle-based RUL target.
        cell_df["RUL"] = eol_cycle - cell_df["cycle"]

        # Throughput-based target (remaining Ah throughput until EoL).
        cell_df["throughput_ah_cumulative"] = cell_df[
            "throughput_ah_cycle"
        ].cumsum()
        throughput_at_eol = float(cell_df["throughput_ah_cumulative"].iloc[-1])
        cell_df["RUL_THROUGHPUT"] = (
            throughput_at_eol - cell_df["throughput_ah_cumulative"]
        )

        features.append(cell_df)

    if not features:
        raise ValueError(
            "No features were extracted from processed cell files."
        )

    features_df = pd.concat(features, ignore_index=True)

    features_data_path = Path(cfg["data"]["features_data_path"])
    features_data_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(
        features_data_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    logger.info(
        "Saved features to %s with shape %s",
        features_data_path,
        features_df.shape,
    )


if __name__ == "__main__":
    make_features()
