import pandas as pd
import numpy as np
import os
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from scipy.stats import norm

# Add project root to sys.path so we can import src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.data.weather_api import fetch_temperature
from src.data.data_manager import add_result

DATA_FILE = os.path.join(PROJECT_ROOT, "src", "data", "results.csv")

# ==============================================================================
# Bayesian Optimization Logic
# ==============================================================================

def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["timestamp", "lat", "lng", "temp", "search_method"])
    return pd.read_csv(DATA_FILE)

def expected_improvement(X_grid, y_sample, gp, xi=0.01):
    """
    Computes the Expected Improvement (EI) at points X_grid.
    EI balances exploration (high sigma) and exploitation (high mu).
    """
    mu, sigma = gp.predict(X_grid, return_std=True)
    mu_sample_opt = np.max(y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        
    return ei

def upper_confidence_bound(X_grid, gp, kappa=1.96):
    """
    Computes the Upper Confidence Bound (UCB) at points X_grid.
    kappa controls the exploration-exploitation trade-off.
    Higher kappa = more exploration.
    """
    mu, sigma = gp.predict(X_grid, return_std=True)
    return mu + kappa * sigma

def suggest_next_point(df, method="UCB", grid_step=2):
    X = df[["lat", "lng"]].values
    y = df["temp"].values

    # Kernel: Matern is often better than RBF for spatial data because it's less smooth.
    # We use a ConstantKernel to scale the variance, and Matern for the length scale.
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[10.0, 10.0], length_scale_bounds=(1e-1, 1e3), nu=2.5)
    
    # alpha adds noise to the diagonal to prevent numerical instability and account for noisy measurements
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1.0, normalize_y=True)
    
    print("Fitting Gaussian Process...")
    gp.fit(X, y)
    print(f"Learned kernel: {gp.kernel_}")

    # Create grid to evaluate acquisition function
    lat_grid = np.arange(-90, 90 + grid_step, grid_step)
    lng_grid = np.arange(-180, 180 + grid_step, grid_step)
    lng_mesh, lat_mesh = np.meshgrid(lng_grid, lat_grid)
    grid_points = np.column_stack([lat_mesh.ravel(), lng_mesh.ravel()])

    print(f"Evaluating {method} acquisition function on {len(grid_points)} grid points...")
    if method == "EI":
        acq_values = expected_improvement(grid_points, y, gp)
    elif method == "UCB":
        acq_values = upper_confidence_bound(grid_points, gp, kappa=2.5)
    else:
        raise ValueError(f"Unknown acquisition method: {method}")
        
    best_idx = np.argmax(acq_values)
    best_lat, best_lng = grid_points[best_idx]
    
    return best_lat, best_lng, acq_values[best_idx]

def run_optimization_step(method="UCB"):
    print(f"--- Running BO Step ({method}) ---")
    df = load_data()
    if len(df) < 2:
        print("Error: Need at least 2 points to start BO.")
        return
        
    print(f"Current data points: {len(df)}")
    best_current = df["temp"].max()
    print(f"Current best temperature: {best_current}°C")
    
    lat, lng, acq_val = suggest_next_point(df, method=method)
    
    print(f"Suggested next point: Lat {lat:.4f}, Lng {lng:.4f} (Acq Score: {acq_val:.4f})")
    
    print("Fetching real temperature from API...")
    temp = fetch_temperature(lat, lng, search_method=f"BO_{method}")
    
    if temp is not None:
        print(f"Result: {temp}°C at ({lat:.4f}, {lng:.4f})")
    else:
        print("Failed to fetch temperature.")

if __name__ == "__main__":
    # You can change to 'EI' if you want to try Expected Improvement
    run_optimization_step(method="UCB")
