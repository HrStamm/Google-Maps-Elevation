# Bayesian Optimization Model - Structure Documentation

## Overview
This document describes the clean, simple structure of the Bayesian Optimization model for finding maximum temperature locations.

## File Structure

### Core Files

1. **`config.yaml`** (Root directory)
   - Central configuration file for all model hyperparameters
   - Contains:
     - Kernel parameters (variance, lengthscale, noise)
     - Optimization parameters (n_guesses, exploration)
     - Map bounds (latitude and longitude limits)
   - Easy to modify without touching code

2. **`src/models/bayesian_optimizer.py`**
   - Main model implementation
   - Contains three classes:
     - `SquaredExponentialKernel`: Implements the RBF kernel
     - `GaussianProcessModel`: Handles GP regression
     - `BayesianOptimizer`: Orchestrates the optimization
   - Well-commented and modular

3. **`src/data/weather_api.py`**
   - Fetches temperature data from Open-Meteo API
   - Includes caching mechanism to avoid redundant API calls
   - Retry logic for reliability

4. **`src/data/data_manager.py`**
   - Manages data storage and retrieval
   - Validates coordinates
   - Handles CSV caching of results

5. **`src/visualization/visualize.py`**
   - Provides visualization utilities
   - Functions:
     - `print_optimization_summary()`: Shows statistics
     - `print_guess_table()`: Displays all guesses
     - `create_ascii_map()`: Simple map visualization
     - `visualize_results()`: Comprehensive visualization

6. **`main.py`**
   - Entry point for running the optimization
   - Simple, clean interface
   - Calls optimizer and visualization

## Model Flow

```
1. main.py
   ↓
2. BayesianOptimizer.__init__()
   - Loads config.yaml
   - Initializes GP model with SE kernel
   ↓
3. BayesianOptimizer.optimize()
   - Evaluates initial guesses
   - For each iteration:
     a. Update GP with current data
     b. Find next point using UCB acquisition
     c. Fetch temperature (weather_api.py)
     d. Standardize temperature to [0, 1]
   ↓
4. visualize_results()
   - Display summary, table, and map
```

## Key Design Decisions

### Simplicity
- Three focused classes, each with a single responsibility
- Clear separation between model logic and data fetching
- Configuration externalized to YAML file

### Clarity
- Extensive comments throughout the code
- Docstrings for all functions and classes
- Descriptive variable names

### Modularity
- Each component can be tested independently
- Easy to swap kernel or acquisition function
- Visualization is separate from optimization

## Model Components

### 1. Squared Exponential Kernel
```python
k(x, x') = variance * exp(-||x - x'||^2 / (2 * lengthscale^2))
```
- Controls smoothness of the GP
- `variance`: Signal strength
- `lengthscale`: Correlation distance

### 2. Gaussian Process
- Predicts mean and variance at any point
- Uses kernel to model spatial correlations
- Provides uncertainty estimates

### 3. Upper Confidence Bound (UCB)
```python
UCB = mean + exploration * sqrt(variance)
```
- Balances exploration vs exploitation
- Higher `exploration` → more exploration
- Guides where to sample next

## Configuration Parameters

### Kernel Parameters
- **variance** (default: 1.0): Signal variance of the kernel
- **lengthscale** (default: 10.0): Smoothness parameter
  - Lower = more wiggly
  - Higher = smoother predictions
- **noise** (default: 0.01): Observation noise

### Optimization Parameters
- **n_guesses** (default: 20): Total number of samples
- **exploration** (default: 2.0): Exploration-exploitation tradeoff
  - Lower = more exploitation (greedy)
  - Higher = more exploration

### Bounds
- **lat_min/max**: -90 to 90 (South to North)
- **lng_min/max**: -180 to 180 (West to East)

## Usage Example

```python
from src.models.bayesian_optimizer import BayesianOptimizer

# Initialize with config
optimizer = BayesianOptimizer(config_path="config.yaml")

# Run optimization
results = optimizer.optimize(
    initial_guesses=[(25.0, 15.0)]  # Optional
)

# Results dictionary contains:
# - guesses: List of (lat, lng) tuples
# - temperatures: List of actual temperatures
# - standardized_temps: List of normalized values [0, 1]
# - best_guess: (lat, lng) of maximum temperature
# - best_temperature: Maximum temperature found
```

## Extending the Model

### Change the Kernel
Create a new kernel class with the same interface:
```python
class NewKernel:
    def __init__(self, param1, param2):
        ...
    
    def __call__(self, X1, X2):
        # Return kernel matrix
        ...
```

### Change the Acquisition Function
Add a new method to `BayesianOptimizer`:
```python
def _acquisition_ei(self, X):
    # Expected Improvement
    ...
```

### Add New Bounds or Constraints
Modify `config.yaml` and update `_optimize_acquisition()` method.

## Dependencies

Minimal dependencies for simplicity:
- `numpy`: Array operations and linear algebra
- `pyyaml`: Configuration file parsing
- `requests`: API calls for temperature data

## Testing

Run the model:
```bash
python main.py
```

Test individual components:
```bash
# Test weather API
python src/data/weather_api.py

# Test visualization
python src/visualization/visualize.py
```

## Future Improvements

Potential enhancements while maintaining simplicity:
1. Add more acquisition functions (EI, PI)
2. Support for different kernels (Matern, etc.)
3. Parallel sampling (batch Bayesian Optimization)
4. Plotting with matplotlib (optional)
5. Save/load optimization state
