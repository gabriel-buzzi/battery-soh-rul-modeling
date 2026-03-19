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

### 3. Modeling Pipeline Execution

The modeling flow now uses a single entrypoint:

```bash
python -m severson_features_soh_rul.modeling.pipeline stage=<stage>
```

Configuration is defined in `config/modeling.yaml`, with `features.columns` as the only feature source of truth.

#### 3.1 Baseline Flow (One Command)

Runs: `optimize -> fit_final_model -> predict`

```bash
python -m severson_features_soh_rul.modeling.pipeline \
  stage=baseline_flow \
  target=SOH
```

#### 3.2 Individual Stages

```bash
python -m severson_features_soh_rul.modeling.pipeline stage=optimize target=SOH
python -m severson_features_soh_rul.modeling.pipeline stage=rank target=SOH
python -m severson_features_soh_rul.modeling.pipeline stage=topk_sweep target=SOH
python -m severson_features_soh_rul.modeling.pipeline stage=fit_final_model target=SOH
python -m severson_features_soh_rul.modeling.pipeline stage=predict target=SOH
python -m severson_features_soh_rul.modeling.pipeline stage=robustness_protocol_lopo target=SOH
```

#### 3.3 Artifact Layout and Guarantees

Artifacts are stored under:

```text
results/modeling/<run_key>/<stage>/
```

Each stage writes:
- `config.resolved.yaml`
- `run_info.json`
- stage-specific artifacts (for example `predictions_test.csv`, `best_params.json`, ranking/top-k outputs)

### 4. Notes on Reproducibility and Caching

- Run identity is deterministic from metadata (`target`, `feature_hash`, `split_seed`, `model_name`, `weighting_strategy`, optional `k_selected`).
- Split files are persisted under `results/modeling/splits/`; keep them stable to compare runs fairly.
- Re-running with the same config reuses stage artifacts by default (`artifacts.overwrite=false`).
- Upstream dependencies are resolved by `run_info.json` metadata matching, with deterministic failure on missing or ambiguous matches when `artifacts.require_exact_match=true`.

## 📄 Scientific Context

This code supports a research paper. The directory `paper/` contains the LaTeX source, which details the physical interpretation of the features (e.g., correlation between voltage curve variance and capacity fade) and the degradation patterns observed.

In order to push the paper content from the Overleaf project please run the following at the root dir of this project
```bash
git submodule update --init --recursive
```
