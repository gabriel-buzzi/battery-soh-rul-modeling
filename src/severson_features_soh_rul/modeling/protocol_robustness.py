"""Protocol-family robustness utilities."""

from __future__ import annotations

import logging
import re

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from severson_features_soh_rul.modeling.models import build_extratrees

logger = logging.getLogger(__name__)


def _infer_rest_presence(charge_policy: str) -> str:
    """Infer whether protocol includes an explicit rest/zero-current step."""
    policy_text = str(charge_policy).lower().strip()
    if not policy_text:
        return "unknown"
    if any(token in policy_text for token in ("rest", "pause", "wait")):
        return "rest"
    if re.search(r"(^|[^a-z0-9])0(\.0+)?c([^a-z0-9]|$)", policy_text):
        return "rest"
    return "no_rest"


def _merge_sparse_families(
    cell_stats_df: pd.DataFrame,
    min_cells_per_family: int,
) -> pd.DataFrame:
    """Merge sparse protocol families into nearest non-sparse families."""
    if min_cells_per_family <= 1:
        return cell_stats_df

    work_df = cell_stats_df.copy()
    counts_df = (
        work_df.groupby("protocol_family", as_index=False)
        .agg(n_cells=("cell", "nunique"))
        .copy()
    )
    sparse_families = set(
        counts_df.loc[
            counts_df["n_cells"] < int(min_cells_per_family), "protocol_family"
        ].tolist()
    )
    if not sparse_families:
        return work_df

    non_sparse_families = set(
        counts_df.loc[
            counts_df["n_cells"] >= int(min_cells_per_family), "protocol_family"
        ].tolist()
    )
    if not non_sparse_families:
        logger.warning(
            "All protocol families are sparse (min_cells_per_family=%d); keeping raw families.",
            int(min_cells_per_family),
        )
        return work_df

    profile_df = (
        work_df.groupby("protocol_family", as_index=False)
        .agg(
            family_rate_center=("max_charge_c_rate", "median"),
            family_rest_presence=("rest_presence", "first"),
        )
        .copy()
    )
    mapping: dict[str, str] = {}
    for sparse_family in sorted(list(sparse_families)):
        sparse_row = profile_df[
            profile_df["protocol_family"] == sparse_family
        ].iloc[0]
        candidate_df = profile_df[
            profile_df["protocol_family"].isin(non_sparse_families)
        ]
        same_rest_df = candidate_df[
            candidate_df["family_rest_presence"]
            == sparse_row["family_rest_presence"]
        ]
        if not same_rest_df.empty:
            candidate_df = same_rest_df
        nearest_row = candidate_df.iloc[
            (
                candidate_df["family_rate_center"]
                - float(sparse_row["family_rate_center"])
            )
            .abs()
            .argmin()
        ]
        mapping[sparse_family] = str(nearest_row["protocol_family"])

    work_df["protocol_family"] = work_df["protocol_family"].replace(mapping)
    return work_df


def build_protocol_families(
    features_df: pd.DataFrame,
    cells_rated_capacity: float,
    max_c_rate_bins: int,
    min_cells_per_family: int,
) -> pd.DataFrame:
    """Assign protocol families by max C-rate bin and protocol rest presence."""
    required_cols = {"cell", "charge_I_mean", "cycle"}
    if not required_cols.issubset(features_df.columns):
        missing = sorted(list(required_cols - set(features_df.columns)))
        raise ValueError(
            "Missing columns for protocol robustness family assignment: "
            f"{missing}. Ensure charge-only features were extracted."
        )

    agg_cols: dict[str, tuple[str, str]] = {
        "avg_charge_current": ("charge_I_mean", "mean"),
        "max_charge_current": ("charge_I_mean", "max"),
    }
    if "charge_policy" in features_df.columns:
        agg_cols["charge_policy"] = ("charge_policy", "first")
    cell_stats_df = (
        features_df.groupby("cell", as_index=False, sort=False)
        .agg(**agg_cols)
        .copy()
    )
    if "charge_policy" not in cell_stats_df.columns:
        logger.warning(
            "Column 'charge_policy' is missing from features_df. Assigning rest_presence='unknown' for all cells."
        )
        cell_stats_df["charge_policy"] = "unknown"

    cell_stats_df["avg_charge_current"] = cell_stats_df[
        "avg_charge_current"
    ].abs()
    cell_stats_df["max_charge_current"] = cell_stats_df[
        "max_charge_current"
    ].abs()
    cell_stats_df["avg_charge_c_rate"] = cell_stats_df[
        "avg_charge_current"
    ] / float(cells_rated_capacity)
    cell_stats_df["max_charge_c_rate"] = cell_stats_df[
        "max_charge_current"
    ] / float(cells_rated_capacity)
    cell_stats_df["rest_presence"] = cell_stats_df["charge_policy"].map(
        _infer_rest_presence
    )
    try:
        max_c_rate_bin = pd.qcut(
            cell_stats_df["max_charge_c_rate"],
            q=int(max_c_rate_bins),
            labels=False,
            duplicates="drop",
        )
        cell_stats_df["max_c_rate_bin"] = (
            max_c_rate_bin.fillna(0).astype(int)
        )
    except ValueError:
        cell_stats_df["max_c_rate_bin"] = 0

    cell_stats_df["protocol_family"] = cell_stats_df.apply(
        lambda row: (
            f"bin_{int(row['max_c_rate_bin'])}"
            f"__{str(row['rest_presence'])}"
        ),
        axis=1,
    )
    cell_stats_df = _merge_sparse_families(
        cell_stats_df=cell_stats_df,
        min_cells_per_family=int(min_cells_per_family),
    )
    return cell_stats_df[
        [
            "cell",
            "charge_policy",
            "rest_presence",
            "avg_charge_c_rate",
            "max_charge_c_rate",
            "max_c_rate_bin",
            "protocol_family",
        ]
    ].copy()


def run_protocol_family_holdout(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    best_params: dict,
    n_jobs: int,
    family_df: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Hold out one protocol family at a time and evaluate performance."""
    merged_df = features_df.merge(
        family_df[
            [
                "cell",
                "protocol_family",
                "rest_presence",
                "avg_charge_c_rate",
                "max_charge_c_rate",
                "max_c_rate_bin",
            ]
        ],
        on="cell",
        how="inner",
    )
    cell_life_df = (
        merged_df.groupby(
            [
                "cell",
                "protocol_family",
                "rest_presence",
                "avg_charge_c_rate",
                "max_charge_c_rate",
                "max_c_rate_bin",
            ],
            as_index=False,
        )
        .agg(cycle_life=("cycle", "max"))
        .copy()
    )

    rows: list[dict] = []
    for family in sorted(merged_df["protocol_family"].unique().tolist()):
        train_df = merged_df[merged_df["protocol_family"] != family]
        test_df = merged_df[merged_df["protocol_family"] == family]
        held_out_cell_life = cell_life_df[
            cell_life_df["protocol_family"] == family
        ]["cycle_life"]
        if train_df.empty or test_df.empty:
            continue

        model = build_extratrees(
            params=best_params,
            random_seed=random_seed,
            n_jobs=n_jobs,
        )
        model.fit(train_df[feature_columns], train_df[target])
        y_pred = model.predict(test_df[feature_columns])
        y_true = test_df[target]

        rows.append(
            {
                "held_out_family": family,
                "n_train_cells": int(train_df["cell"].nunique()),
                "n_test_cells": int(test_df["cell"].nunique()),
                "n_test_samples": int(test_df.shape[0]),
                "rmse": float(root_mean_squared_error(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
                "family_rest_presence_held_out": str(
                    test_df["rest_presence"].iloc[0]
                ),
                "max_c_rate_bin_held_out": int(
                    test_df["max_c_rate_bin"].iloc[0]
                ),
                "avg_charge_c_rate_held_out": float(
                    test_df["avg_charge_c_rate"].mean()
                ),
                "max_charge_c_rate_held_out": float(
                    test_df["max_charge_c_rate"].mean()
                ),
                "cycle_life_mean_held_out": float(held_out_cell_life.mean()),
                "cycle_life_std_held_out": float(
                    held_out_cell_life.std(ddof=0)
                ),
                "cycle_life_min_held_out": int(held_out_cell_life.min()),
                "cycle_life_median_held_out": float(
                    held_out_cell_life.median()
                ),
                "cycle_life_max_held_out": int(held_out_cell_life.max()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("held_out_family")
        .reset_index(drop=True)
    )


def summarize_protocol_robustness(results_df: pd.DataFrame) -> dict:
    """Create compact summary payload for protocol-family robustness."""
    if results_df.empty:
        return {
            "n_families_evaluated": 0,
            "rmse_mean": 0.0,
            "rmse_std": 0.0,
            "mae_mean": 0.0,
            "mae_std": 0.0,
            "r2_mean": 0.0,
            "r2_std": 0.0,
        }
    return {
        "n_families_evaluated": int(results_df.shape[0]),
        "rmse_mean": float(results_df["rmse"].mean()),
        "rmse_std": float(results_df["rmse"].std(ddof=0)),
        "mae_mean": float(results_df["mae"].mean()),
        "mae_std": float(results_df["mae"].std(ddof=0)),
        "r2_mean": float(results_df["r2"].mean()),
        "r2_std": float(results_df["r2"].std(ddof=0)),
    }
