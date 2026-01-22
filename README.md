This is a smart pivot. For **AI Research** and **Machine Learning Engineer** roles, the hiring manager is less interested in "Can you put this in a Docker container?" and more interested in "Did you evaluate this correctly?" and "Can you design a rigorous experiment?"

They want to see that you understand **data leakage, statistical significance, and experimental reproducibility.**

Here is the revised README. It highlights the rigorous evaluation protocols, the sophisticated optimization pipeline (Bayesian search), and the engineering that went into making the experimentation scalable.

---

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

| Target | Model Architecture | Metric (Test Set) | Generalization Capability |
| --- | --- | --- | --- |
| **State of Health** | LightGBM Regressor | **RMSE < 0.80%** | High () |
| **RUL** | Extra Trees Regressor | **RMSE < 60 cycles** | High () |

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

Download the [Severson et al. dataset](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) (three batches `.mat` files) to `data/external/`.

```bash
# 1. Ingest and convert to HDF5
python src/data/load_data.py

# 2. Preprocess signals (Resampling & Smoothing)
python src/data/build_data.py

# 3. Extract Statistical Features
python src/data/make_features.py

```

### 3. Running Experiments

You can reproduce the specific optimization runs used in the paper or launch new explorations.

```bash
# A. Feature Selection (Recursive Feature Elimination)
python src/modeling/feature_importance.py

# B. Run Bayesian Optimization (Single Model)
python src/modeling/optimization.py

# C. Run Full Experimental Grid (Multirun)
# This utilizes Joblib to parallelize training across available cores
python src/modeling/optimization.py -m

# D. Final Evaluation on Test Set
python src/modeling/evaluation.py

```

## 📄 Scientific Context

This code supports a research paper. The directory `paper/` contains the LaTeX source, which details the physical interpretation of the features (e.g., correlation between voltage curve variance and capacity fade) and the degradation patterns observed.
