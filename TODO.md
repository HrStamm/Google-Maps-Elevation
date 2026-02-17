# 📋 Project To-Do List: Global Temperature Optimization

## Project Overview
The goal is to use **Bayesian Optimization (BO)** with a **Gaussian Process (GP)** surrogate model to find the point on Earth with the highest current temperature. 

---

## 🚨 Phase 0: Critical Foundation (Do these first!)
*These tasks must be completed so the rest of the team has a stable environment and data to work with.*

- [x] **Standardized Data Format** (Data/Scientist): **Confirmed!** Using `src/data/results.csv` with headers: `timestamp, lat, lng, temp, search_method`.
- [x] **Coordinate Strategy** (All): **Confirmed!** We use decimal (latitude, longitude). Example: Copenhagen is `(55.67, 12.56)`.
- [x] **Data Caching System** (Data Engineer): **Implemented!** `fetch_temperature` now automatically checks `results.csv` before calling the API.
- [x] **Environment Sync** (All): **Ready!** Everyone should run `git pull` and `uv sync`.

---

## 🏗️ Workstreams & Roles

### 1. Data & Infrastructure (The Data Engineer)
*Focus: API reliability, caching, and data collection.*
- [x] **API Hardening**: Enhanced `weather_api.py` with retries, exponential backoff, and timeouts for better reliability.
- [x] **Data Caching**: Automated system implemented in `src/data/data_manager.py` and `src/data/weather_api.py`.
- [x] **Coordinate Validation**: Checks implemented to ensure lat/lng are within valid global bounds.

### 2. Bayesian Optimization Logic (The ML Engineer)
*Focus: GP modeling and acquisition strategy.*
- [ ] **GP Model Implementation**: Set up the GP using `scikit-learn` or `GPy`.
- [ ] **Kernel Selection**: Design and test kernels (e.g., RBF, Matern, or Periodic for longitude to handle the wrap-around).
- [ ] **Acquisition Functions**: Implement **Expected Improvement (EI)** and **Upper Confidence Bound (UCB)** to balance exploration and exploitation.
- [ ] **Optimization Loop**: Build the main loop that iterates between fitting the GP and suggesting new points.

### 3. Baselines & Benchmarking (The Scientist)
*Focus: Measuring performance against simpler methods.*
- [ ] **Random Search**: Implement a random sampling baseline script to establish a baseline performance.
- [ ] **Manual Search Utility**: Create a tool for human teammates to input their "best guesses" to compare human vs. AI.
- [ ] **Hyperparameter Tuning**: Perform a grid or manual search for the best GP length-scales (e.g., how far does a temperature measurement "carry" over the earth?).
- [ ] **Standardized Evaluation**: Create metrics for "Max Temp vs. Iteration" and "Sample Efficiency".

### 4. Visualization & Reporting (The Lead)
*Focus: Communicating the model's "beliefs" and results.*
- [ ] **Global Temperature Heatmaps**: Visualize the GP's predicted temperature distribution across the globe.
- [ ] **Uncertainty Maps**: Plot the model's variance (where it is least confident and needs more data).
- [ ] **Acquisition Surface**: Visualize which areas the model wants to explore next based on the acquisition function.
- [ ] **Final Comparisons**: Create the final plots comparing BO, Random Search, and Manual Search.

---

## 📈 Evaluation Metrics
- **Objective**: Find the absolute highest temperature.
- **Efficiency**: Minimize the number of API calls to find said maximum.
- **Robustness**: How much do the results change with different GP kernels?

---

## 🛠️ Tech Stack
- **Environment**: Managed by `uv`
- **APIs**: Open-Meteo (Weather) - *Elevation is no longer required.*
- **Modeling**: `scikit-learn` or `GPy`
- **Visualization**: `matplotlib`, `plotly`, or `folium`
