# Lightweight Single-Cycle Prognostics for Li-ion Batteries: Feature Extraction and Cross-Cell Generalization

## Project Overview

This repository contains a comprehensive research project on machine learning techniques for estimating the State-of-Health (SoH) and Remaining Useful Life (RUL) of lithium-ion batteries. The project encompasses data processing, feature engineering, model training and optimization, and is documented in a scientific paper. The primary objective is to develop lightweight, feature-based prognostic models that can operate on single charge-discharge cycles, making them practical for deployment in real-world battery management systems.

### Key Research Goals

- **Single-cycle diagnostics**: Demonstrate that accurate SoH and RUL estimation is possible from a single diagnostic cycle, eliminating the need for continuous monitoring of long degradation histories
- **Feature analysis**: Identify and analyze the most informative statistical features from voltage, current, and temperature signals for battery prognostics
- **Model comparison**: Systematically evaluate multiple machine learning algorithms to understand the trade-offs between interpretability, efficiency, and predictive performance
- **Lightweight deployment**: Develop models with minimal feature counts suitable for embedded battery management systems
- **Cross-cell generalization**: Validate models on unseen cells to ensure robust generalization beyond training data

## Research Findings Summary

This work proposes a single-cycle, feature-based approach for estimating the State-of-Health (SoH) and Remaining Useful Life (RUL) of lithium-ion batteries using the public Severson LFP dataset (124 cells). We extract 16 statistical features from voltage, current, and temperature signals of a diagnostic full cycle. Gradient boosting and random forest regressors are trained with cell-wise cross-validation and tested on 25 unseen cells. Results show median RMSE of 0.7–0.8% for SoH and 42–60 cycles for RUL, with consistent generalization across batches. Our analysis highlights the most informative features, the trade-off between feature count and accuracy, and the deployment potential of lightweight models.

## Research Details

### Introduction

The transition to renewable energy and the rapid adoption of electric mobility have placed lithium-ion batteries at the center of modern energy storage technologies. This work addresses key challenges in battery prognostics: most ML approaches rely on handcrafted features from full charge-discharge cycles, which limits applicability to controlled datasets. Additionally, many studies require multiple consecutive cycles for reliable features and were validated only on laboratory data with aggressive fast-charging protocols. We develop ML models trained on 124 commercial LiFePO₄/graphite cells cycled under fast-charging conditions, demonstrating that accurate SoH and RUL estimates can be achieved from a single diagnostic cycle—reflecting realistic operational scenarios without requiring continuous monitoring of full degradation histories.

### Dataset

The dataset comprises cycling data from 124 commercial lithium iron phosphate (LFP)/graphite batteries manufactured by A123 Systems (model APR18650M1A, nominal capacity 1.1 Ah, nominal voltage 3.3 V). All batteries were cycled to failure under various fast-charging conditions in a controlled environment at 30°C on a 48-channel Arbin potentiostat. Charging employed one-step or two-step fast-charging policies, followed by constant current-constant voltage (CC-CV) charging. The dataset includes measurements of voltage, current, temperature, charge/discharge capacity, and internal resistance for each cycle. Temperature data was collected using thermocouples, though reliability may vary. Voltage and current cutoffs were 3.6 V and 2.0 V respectively, adhering to manufacturer specifications.

### Methodology

Our methodology extracts 16 statistical features per cycle from voltage, current, and temperature signals: mean, median, standard deviation, interquartile range, kurtosis, and differential entropy (voltage only). Preprocessing includes time-gap removal, invalid cycle filtering, and sampling rate standardization to 1 Hz. Feature processing applies spike removal (5th–95th percentile windowing) and Savitzky-Golay smoothing to reduce noise while preserving degradation trends. Feature selection uses Random Forest importance rankings with 10-fold cross-validation grouped by cell to identify the most informative features while avoiding data leakage. Four ML models are evaluated: Tweedie Regressor, K-Nearest Neighbors, Extra Trees Regressor, and LightGBM Regressor, using cell-wise cross-validation on 99 training cells and testing on 25 unseen cells.

### Results

Tree-based ensemble methods significantly outperformed linear approaches. The LGBMRegressor achieved RMSE below 0.80% for SoH on 75% of test cells with R² > 0.97, while ExtraTreesRegressor achieved RMSE below 59.91 cycles for RUL on 75% of test cells with R² > 0.96. Performance dropped only slightly when reducing from 16 to 4 features, indicating lightweight models are feasible. However, RUL models show a tendency toward under-estimation in extreme scenarios (high cycle count cells), highlighting a limitation in generalization to exceptional battery lifespans. SoH predictions demonstrate balanced under/over-estimation across the full range of conditions.

### Discussion and Conclusions

The results demonstrate that statistical feature representations of single cycles effectively capture essential information for battery prognosis. This work confirms well-known degradation phenomena (knee point in SoH evolution, capacity fade under high charging rates, internal resistance growth) and establishes a practical diagnostic framework where SoH and RUL can be estimated from periodic diagnostic cycles without requiring continuous monitoring of long cycle histories. However, limitations include the exclusive focus on one chemistry (LFP/graphite), highly controlled laboratory conditions, and the requirement for complete charge-discharge cycles. Real-world scenarios (electric vehicles, grid storage) involve partial, irregular cycling under variable environmental conditions. Future work will extend this approach to heterogeneous datasets, incorporate physics-informed features for better generalization, and evaluate deployment in realistic battery management systems.

## Reproduce results
First you should install [Pixi](https://pixi.sh/latest/) on your machine.

For Linux & macOS run the following:
```
curl -fsSL https://pixi.sh/install.sh | sh
```

Once you have Pixi installed navigate to the project root folder and run `pixi install` to setup the environment.

Set the .mat files paths at `src/conf/config.yaml`.

On your terminal, at the root project path run the following:
```shell
pixi shell
```
make sure all the commands are ran within this shell.

To load the downloaded .mat files from Severson et. al. run the following
```shell
python src/data/load_data.py
```
this will convert data from the `.mat` files into a `.h5` dataset inside `data/external` folder.

Once the loading is done, run the following to process the raw batteries' signals:
```shell
python src/data/build_data.py
```
this should save a `.csv` file for each cell at `data/processed/cells` folder.

After building the data from all cells you might want to execute the features extraction process by running the following:
```shell
python src/data/make_features.py
```
This should save at `data/interim` a file named `features.parquet` that contains the statistical features from voltage, current and temperature from each valid cycle of each cell, as well as cycle and cell ids and SOH and RUL values for each cycle.

The next step of the pipeline is preparing data for training and testing the models, for this run the following:
```shell
python src/data/prepare_data.py
```

## Research Outcomes

### Scientific Paper

The research is documented in a comprehensive peer-reviewed scientific paper located in the `paper/latex/` directory. The paper provides:

- **Detailed methodology**: Complete description of feature extraction, preprocessing, and model selection procedures
- **Comprehensive results**: Performance metrics for all evaluated models across 25 unseen test cells
- **In-depth analysis**: Feature importance analysis, degradation pattern investigation, and model-specific performance characteristics
- **Critical discussion**: Limitations of the current approach, practical deployment considerations, and directions for future work

The paper covers the full research lifecycle including exploratory data analysis, statistical features derivation, machine learning model evaluation with hyperparameter optimization, and cross-cell generalization validation. 

#### Robust Python Framework

This research project yields a robust python framework for machine learning experimentation leveraging Hydra combined with joblib for pipeline configuration and multiprocessing, Optuna for hyperparameter optimization and scikit-learn for modeling.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
│
├── data               <- All project data
│   ├── external       <- Data from third party sources (Severson et al. dataset)
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks for analysis and visualization
│                         Naming convention: number_initials-description (e.g. 1.0-jqp-eda)
│
├── paper              <- Research paper and related materials (key output of research)
│   ├── latex/         <- LaTeX source for the scientific paper
│   │   ├── main.tex   <- Main paper document
│   │   ├── content/   <- Paper sections (introduction, methodology, results, etc.)
│   │   ├── figures/   <- Generated graphics and figures for the paper
│   │   ├── tables/    <- Data tables for the paper
│   │   ├── references.bib <- Bibliography
│   │   └── preamble.tex   <- LaTeX configuration
│   ├── markdown/      <- Markdown version of paper for documentation
│   └── figures/       <- Generated graphics and figures to be used in reporting
│
├── pyproject.toml     <- Project configuration and dependencies (Pixi/Poetry)
│
├── references/        <- Data dictionaries, manuals, and explanatory materials
│
└── src/               <- Source code for data processing and modeling
    ├── __init__.py
    ├── data/          <- Data loading and preprocessing scripts
    ├── conf/          <- Configuration files (paths, parameters)
    ├── analysis/      <- Exploratory data analysis scripts
    ├── modeling/      <- Model training, evaluation, and prediction
    └── functions.py   <- Utility functions
```