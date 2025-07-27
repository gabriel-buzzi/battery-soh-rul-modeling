# Machine Learning Techniques for Li-ion Battery State-of-Health and Remaining Useful Life Estimation

Utilizing features of cycling data from Severson et. al. fast-charging battery dataset.

## Run the code
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

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── paper          <- Scientific paper associated.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

