Google-maps-Elevation-Predictor
==============================

A Bayesian Optimization project that uses Gaussian Process models to find the location with the highest temperature on a world map.

## Overview

This project implements Bayesian Optimization with a Gaussian Process (GP) using a Squared Exponential kernel to efficiently search for the maximum temperature location on Earth. The model makes a limited number of guesses and uses the Upper Confidence Bound (UCB) acquisition function to balance exploration and exploitation.

## Features

- **Gaussian Process Model**: Uses Squared Exponential kernel for smooth spatial predictions
- **Bayesian Optimization**: Efficiently searches for the maximum temperature with limited samples
- **Temperature API Integration**: Fetches real-time temperature data from Open-Meteo API
- **Caching System**: Stores previous queries to avoid redundant API calls
- **Configurable Parameters**: All hyperparameters are easily adjustable via `config.yaml`

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the model** (optional):
   Edit `config.yaml` to adjust:
   - Kernel variance and lengthscale
   - Noise parameter
   - Number of allowed guesses
   - Exploration parameter

3. **Run the optimization**:
   ```bash
   python main.py
   ```

## Configuration

The `config.yaml` file contains all model hyperparameters:

```yaml
kernel:
  variance: 1.0           # Signal variance
  lengthscale: 10.0       # Smoothness parameter
  noise: 0.01             # Observation noise

optimization:
  n_guesses: 20           # Number of allowed guesses
  exploration: 2.0        # Exploration parameter (UCB)

bounds:
  lat_min: -90
  lat_max: 90
  lng_min: -180
  lng_max: 180
```

## Model Architecture

The model consists of three main components:

1. **SquaredExponentialKernel**: Implements the RBF kernel for the GP
2. **GaussianProcessModel**: Handles GP regression with mean and variance predictions
3. **BayesianOptimizer**: Orchestrates the optimization using UCB acquisition function

### Input
- Temperature values (standardized to [0, 1])
- Map bounds (latitude: -90 to 90, longitude: -180 to 180)

### Output
- All guesses and their order
- Best temperature found
- Best location coordinates

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
