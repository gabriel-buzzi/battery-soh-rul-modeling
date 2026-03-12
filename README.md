# 🔋 Li-ion Battery Prognostics: A Reproducible Research Framework

## 📌 Project Overview

This repository hosts a comprehensive research framework for estimating State-of-Health (SoH) and Remaining Useful Life (RUL) of Lithium-Ion batteries. The primary focus of this work is not just predictive performance, but the development of a **robust, modular, and reproducible experimentation pipeline.**

The project investigates whether **single-cycle diagnostic features** can replace full-history data for prognostics. To answer this, I implemented a rigorous evaluation protocol to ensure zero data leakage and validated the models on unseen battery cells with distinct cycling patterns.

### 🔬 Key Research & Engineering Highlights

* **Rigorous Evaluation Strategy:** Implemented **Cell-Wise Cross-Validation** to strictly prevent data leakage. Time-series data from the same battery never bleeds between train and validation sets.
* **Bayesian Hyperparameter Optimization:** Utilized **Optuna** with Tree-structured Parzen Estimator (TPE) samplers to efficiently search high-dimensional parameter spaces, outperforming standard grid searches.
* **Representative Subsampling:** Optimized training efficiency by utilizing statistically representative subsets of the data. Distribution matching was performed to ensure the subset preserved the statistical properties of the full dataset.
* **Scalable Architecture:** Leveraged **Joblib** for multiprocessing and **Hydra** for configuration management, allowing for parallel execution of multiple experiments and easy reproduction of results.

## 🛠️ Methodology & Experimental Design

### 1. Data Processing & Feature Engineering

* **Source:** Severson et al. dataset (124 LFP/Graphite cells).
* **Signal Processing:** Raw signals (Voltage, Current, Temperature) are processed via spike removal (5th-95th percentile filtering) and Savitzky-Golay smoothing.
* **Feature Extraction:** 16 statistical features (e.g., differential entropy, kurtosis, skewness) are extracted per cycle.
* **Note:** All preprocessing that requires features from more than one cycle were not applied on the test set.

### 2. The Experimentation Pipeline

The codebase is structured to facilitate rapid iteration and hypothesis testing:

* **Config-Driven:** All experimental parameters (model types, feature sets, hyperparameters) are controlled via `src/conf/config.yaml`.
* **Multirun Support:** The pipeline supports dispatching multiple experiments simultaneously to compare model architectures (e.g., LightGBM vs. Extra Trees) and feature subsets (16 vs. 4 features) in a single run.

### 3. Model Optimization

Instead of manual tuning, the project employs an automated optimization stage:

* **Sampler:** Bayesian Optimization (TPE) via Optuna.
* **Objective:** Minimizing RMSE on the validation fold.
* **Pruning:** Early stopping of unpromising trials to save compute resources.

## 📊 Results & Validation

Models were optimized on a training set of 99 cells and validated on a completely **unseen test set of 25 cells**.

| Target | Model Architecture | Metric (Test Set) |
| --- | --- | --- |
| **State of Health** | LightGBM Regressor | **RMSE < 0.80%** |
| **RUL** | Extra Trees Regressor | **RMSE < 60 cycles** |

> **Research Insight:** Error analysis showed that tree-based ensembles generalized significantly better across different fast-charging policies compared to linear baselines. The subsampling strategy reduced training time by ~60% with negligible impact on final test accuracy.

## 📂 Project Structure

The repository follows a strict separation of concerns, ensuring that data processing, modeling, and configuration are decoupled.

```text
├── data/               # Data versioning (Raw -> Processed)
├── notebooks/          # Research notebooks (EDA, Distribution Checks, Result Analysis)
├── paper/              # LaTeX source for the associated scientific paper
├── src/
│   ├── conf/           # Hydra configs (Defining search spaces & pipeline args)
│   ├── data/           # ETL pipelines & Statistical Feature Extraction
│   ├── modeling/       # Logic for Training, Inference, and Evaluation
│   │   ├── feature_importance.py
│   │   ├── optimization.py  # Optuna + Joblib implementation
│   │   └── evaluation.py    # Unseen test set validation
│   └── analysis/       # Scripts for distribution comparison
├── pixi.lock           # Reproducible environment lockfile
└── README.md

```

## 🚀 Reproduction Instructions

This project uses [Pixi](https://pixi.sh/latest/) to guarantee a reproducible scientific environment.

### 1. Environment Setup

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi install
pixi shell
```

### 2. Data Pipeline Execution

Download the [Severson et al. dataset](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) (three batches `.mat` files) to `data/raw/`.

```bash
# 1) Ingest .mat files into a unified HDF5
python -m src.data.load_data

# 2) Build cycle-level processed parquet per cell
python -m src.data.build_data

# 3) Extract full-cycle + charge-only features and targets
#    (SOH, RUL)
python -m src.data.make_features
```

### 3. Manual Track Execution (New Experiments Pipeline)

All revision experiments are run manually track-by-track via `src.experiments.runner`.
This is the recommended flow for paper writing because each step generates explicit artifacts you can inspect before moving to the next one.

#### 3.1 Baseline Full-Cycle Training/Evaluation

What it does:
- creates/reuses deterministic train/test cell split;
- runs (or reuses cached) optimization;
- trains on train split and evaluates on held-out test split;
- exports paper-ready summaries and diagnostics artifacts.

```bash
python -m src.experiments.runner \
  tracks=final_eval \
  target=SOH
```

Main outputs:
- `results/experiments/final_eval/<run_id>/run_summary.json`
- `table_main_metrics.csv/json`
- `predictions_test.csv/json`

#### 3.2 Full-Cycle Feature Analysis

What it does:
- permutation and intrinsic ranking;
- top-k sweep with cached full-feature baseline row;
- heuristic/manual `selected_k`;
- leave-one-out (LOO) inside selected subset;
- no-temperature comparison;
- top-k plots for RMSE and relative gap.

```bash
python -m src.experiments.runner \
  tracks=full_cycle_feature_analysis \
  target=SOH
```

Main outputs:
- `feature_ranking_permutation.csv/json`
- `topk_sweep_metrics.csv/json`
- `loo_metrics.csv/json`
- `no_temp_metrics.json`
- `topk_vs_val_rmse.png`
- `topk_vs_relative_gap.png`

#### 3.3 Charge-Only Feature Analysis

What it does:
- same analysis flow as full-cycle, but using `charge_*` features only.

```bash
python -m src.experiments.runner \
  tracks=charge_only_feature_analysis \
  target=SOH
```

Main outputs:
- same artifact interface as full-cycle feature analysis.

#### 3.4 Uncertainty Analysis

What it does:
- repeated-seed retraining with fixed split and fixed optimized hyperparameters;
- aggregates sample-level uncertainty and region summaries.

```bash
python -m src.experiments.runner \
  tracks=uncertainty \
  target=SOH
```

Main outputs:
- `predictions_repeated.csv/json`
- `uncertainty_summary.json`
- `uncertainty_by_region.csv/json`

#### 3.5 Difficult-Cell Diagnostics

What it does:
- computes per-cell error diagnostics;
- identifies difficult cells and where error concentrates along life.

```bash
python -m src.experiments.runner \
  tracks=diagnostics \
  target=SOH
```

Main outputs:
- `error_cells_summary.csv/json`
- `diagnostics_summary.json`

#### 3.6 Protocol-Family Robustness

What it does:
- builds protocol families from max charging C-rate bins + rest presence inferred from protocol labels;
- performs leave-one-family-out robustness evaluation.

```bash
python -m src.experiments.runner \
  tracks=protocol_robustness \
  target=SOH \
  features.set_id=charge_all
```

Main outputs:
- `protocol_family_results.csv/json`
- `protocol_robustness_summary.json`

#### 3.7 Export Manuscript-Ready Tables

What it does:
- collects latest outputs from relevant tracks;
- writes merged tables under `results/paper_tables/<export_id>/`.

```bash
python -m src.experiments.export_paper_tables
```

Main outputs:
- `table_main_comparison.csv/json`
- `table_feature_analysis.csv/json`
- `table_uncertainty.csv/json`
- `table_robustness.csv/json`

### 4. Notes on Reproducibility and Caching

- Optimization cache is enabled by default. If target, feature space, split, or optimization settings do not change, optimization results are reused automatically.
- Split files are persisted under `results/experiments/splits/`; keep them stable to compare runs fairly.
- For full reproducibility in papers, cite `run_summary.json` and `artifacts_index.json` from each run directory.

## 📄 Scientific Context

This code supports a research paper. The directory `paper/` contains the LaTeX source, which details the physical interpretation of the features (e.g., correlation between voltage curve variance and capacity fade) and the degradation patterns observed.
