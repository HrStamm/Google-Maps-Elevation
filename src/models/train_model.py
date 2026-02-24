"""
Bayesian Optimization for Temperature Search
============================================
Uses Gaussian Process with UCB acquisition function to find the highest
temperature on a 2D world map. The model balances exploration and
exploitation using grid search to optimize the acquisition function.
"""

import numpy as np
import yaml
import os
from src.data.weather_api import fetch_temperature


class BayesianOptimizationSearch:
    """
    Bayesian Optimization using Gaussian Process with RBF kernel and UCB acquisition.
    
    The model maintains a GP over observed temperatures and uses UCB to select
    the next sampling location via grid search.
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the Bayesian Optimization model.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration YAML file.
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # GP hyperparameters
        self.kernel_variance = config['kernel_variance']
        self.lengthscale = config['lengthscale']
        self.noise = config['noise']
        
        # UCB parameter
        self.kappa = config['kappa']
        
        # Search parameters
        self.n_iterations = config['n_iterations']
        self.grid_resolution = config['grid_resolution']
        
        # Bounds
        self.lat_min = config['lat_min']
        self.lat_max = config['lat_max']
        self.lng_min = config['lng_min']
        self.lng_max = config['lng_max']
        
        # Storage for observations
        self.X_observed = []  # List of (lat, lng) tuples
        self.y_observed = []  # List of temperatures
        
        # Temperature normalization (will be set after first observation)
        self.temp_min = None
        self.temp_max = None
        
    def _rbf_kernel(self, X1, X2):
        """
        Compute the RBF (Squared Exponential) kernel matrix.
        
        K(x, x') = variance * exp(-||x - x'||^2 / (2 * lengthscale^2))
        
        Parameters
        ----------
        X1 : np.ndarray, shape (n1, 2)
            First set of points.
        X2 : np.ndarray, shape (n2, 2)
            Second set of points.
            
        Returns
        -------
        K : np.ndarray, shape (n1, n2)
            Kernel matrix.
        """
        # Compute squared distances with longitude wraparound
        X1_expanded = X1[:, np.newaxis, :]  # (n1, 1, 2)
        X2_expanded = X2[np.newaxis, :, :]  # (1, n2, 2)
        
        # Latitude difference (no wraparound)
        dlat = X1_expanded[:, :, 0] - X2_expanded[:, :, 0]
        
        # Longitude difference with wraparound: min(|Δlng|, 360 - |Δlng|)
        dlng = np.abs(X1_expanded[:, :, 1] - X2_expanded[:, :, 1])
        dlng = np.minimum(dlng, 360.0 - dlng)
        
        sq_dist = dlat ** 2 + dlng ** 2
        
        # RBF kernel
        K = self.kernel_variance * np.exp(-sq_dist / (2 * self.lengthscale ** 2))
        return K
    
    def _normalize_temperature(self, temp):
        """Normalize temperature to [0, 1] range."""
        if self.temp_min is None or self.temp_max is None:
            return temp
        if self.temp_max == self.temp_min:
            return 0.5
        return (temp - self.temp_min) / (self.temp_max - self.temp_min)
    
    def _update_normalization(self):
        """Update temperature normalization bounds based on observed data."""
        if len(self.y_observed) > 0:
            self.temp_min = min(self.y_observed)
            self.temp_max = max(self.y_observed)
    
    def _predict(self, X_new):
        """
        Predict mean and standard deviation at new points using GP.
        
        Parameters
        ----------
        X_new : np.ndarray, shape (n, 2)
            Points to predict at.
            
        Returns
        -------
        mean : np.ndarray, shape (n,)
            Predicted mean values (normalized).
        std : np.ndarray, shape (n,)
            Predicted standard deviations.
        """
        if len(self.X_observed) == 0:
            # No observations yet - return prior
            mean = np.zeros(len(X_new))
            std = np.sqrt(self.kernel_variance) * np.ones(len(X_new))
            return mean, std
        
        # Convert observations to arrays
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        
        # Normalize observations
        y_normalized = np.array([self._normalize_temperature(y) for y in y_obs])
        
        # Compute kernel matrices
        K = self._rbf_kernel(X_obs, X_obs)
        K_star = self._rbf_kernel(X_obs, X_new)
        K_star_star = self._rbf_kernel(X_new, X_new)
        
        # Add noise to diagonal of K
        K_noisy = K + self.noise * np.eye(len(X_obs))
        
        # Compute mean: k_star^T * K^-1 * y
        try:
            K_inv_y = np.linalg.solve(K_noisy, y_normalized)
            mean = K_star.T @ K_inv_y
            
            # Compute variance: k_star_star - k_star^T * K^-1 * k_star
            K_inv_k_star = np.linalg.solve(K_noisy, K_star)
            variance = np.diag(K_star_star) - np.sum(K_star * K_inv_k_star, axis=0)
            variance = np.maximum(variance, 1e-10)  # Ensure non-negative
            std = np.sqrt(variance)
            
        except np.linalg.LinAlgError:
            # Fallback to prior if numerical issues
            mean = np.zeros(len(X_new))
            std = np.sqrt(self.kernel_variance) * np.ones(len(X_new))
        
        return mean, std
    
    def _ucb_acquisition(self, X_grid):
        """
        Compute UCB acquisition function values on a grid.
        
        UCB(x) = μ(x) + κ * σ(x)
        
        Parameters
        ----------
        X_grid : np.ndarray, shape (n, 2)
            Grid points to evaluate.
            
        Returns
        -------
        ucb_values : np.ndarray, shape (n,)
            UCB values at each grid point.
        """
        mean, std = self._predict(X_grid)
        ucb_values = mean + self.kappa * std
        return ucb_values
    
    def _create_grid(self):
        """
        Create a uniform grid over the search space.
        
        Returns
        -------
        grid : np.ndarray, shape (grid_resolution^2, 2)
            Grid points as (lat, lng) pairs.
        """
        lat_vals = np.linspace(self.lat_min, self.lat_max, self.grid_resolution)
        lng_vals = np.linspace(self.lng_min, self.lng_max, self.grid_resolution)
        lat_grid, lng_grid = np.meshgrid(lat_vals, lng_vals)
        grid = np.column_stack([lat_grid.ravel(), lng_grid.ravel()])
        return grid
    
    def _select_next_point(self):
        """
        Select the next sampling point by maximizing UCB over a grid.
        
        Returns
        -------
        next_point : tuple
            (lat, lng) of the next point to sample.
        """
        # Create grid
        grid = self._create_grid()
        
        # Evaluate UCB on grid
        ucb_values = self._ucb_acquisition(grid)
        
        # Mask out grid points near already-observed locations
        if len(self.X_observed) > 0:
            X_obs = np.array(self.X_observed)
            min_dist_threshold = 2.0  # degrees
            for obs in X_obs:
                dlat = grid[:, 0] - obs[0]
                dlng = np.abs(grid[:, 1] - obs[1])
                dlng = np.minimum(dlng, 360.0 - dlng)
                dist = np.sqrt(dlat ** 2 + dlng ** 2)
                ucb_values[dist < min_dist_threshold] = -np.inf
        
        # Select point with highest UCB
        best_idx = np.argmax(ucb_values)
        next_point = tuple(grid[best_idx])
        
        return next_point
    
    def run_search(self, seed=None):
        """
        Run the Bayesian Optimization search.
        
        Parameters
        ----------
        seed : int or None
            Random seed for reproducibility (used for initial random sample).
            
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'guesses': list of (lat, lng) tuples in order
            - 'temperatures': list of observed temperatures
            - 'best_location': (lat, lng) of highest temperature
            - 'best_temperature': highest temperature found
        """
        if seed is not None:
            np.random.seed(seed)
        
        results = {
            'guesses': [],
            'temperatures': [],
            'iterations': []
        }
        
        print("=" * 60)
        print("🌍  Bayesian Optimization Temperature Search")
        print("=" * 60)
        print(f"Parameters: κ={self.kappa}, lengthscale={self.lengthscale}")
        print(f"Grid resolution: {self.grid_resolution}x{self.grid_resolution}")
        print(f"Max iterations: {self.n_iterations}\n")
        
        for i in range(self.n_iterations):
            # Select next point
            if i == 0:
                # Start with a random point for initial exploration
                lat = np.random.uniform(self.lat_min, self.lat_max)
                lng = np.random.uniform(self.lng_min, self.lng_max)
                next_point = (lat, lng)
            else:
                next_point = self._select_next_point()
            
            lat, lng = next_point
            
            # Query temperature
            temp = fetch_temperature(lat, lng, search_method="bayesian_optimization", use_cache=False)
            
            if temp is not None:
                # Store observation
                self.X_observed.append((lat, lng))
                self.y_observed.append(temp)
                
                # Update normalization
                self._update_normalization()
                
                # Store in results
                results['guesses'].append((round(lat, 4), round(lng, 4)))
                results['temperatures'].append(temp)
                
                # Find current best
                best_temp = max(self.y_observed)
                best_idx = self.y_observed.index(best_temp)
                best_loc = self.X_observed[best_idx]
                
                results['iterations'].append({
                    'iteration': i + 1,
                    'lat': round(lat, 4),
                    'lng': round(lng, 4),
                    'temp': temp,
                    'best_temp': best_temp
                })
                
                print(f"[BO] Iter {i+1}/{self.n_iterations}  ({lat:.2f}, {lng:.2f})  "
                      f"temp={temp:.1f}°C  best={best_temp:.1f}°C")
            else:
                print(f"[BO] Iter {i+1}/{self.n_iterations}  Failed to fetch temperature")
        
        # Set best results
        if len(self.y_observed) > 0:
            best_temp = max(self.y_observed)
            best_idx = self.y_observed.index(best_temp)
            results['best_location'] = self.X_observed[best_idx]
            results['best_temperature'] = best_temp
        else:
            results['best_location'] = None
            results['best_temperature'] = None
        
        # Print summary
        print("\n" + "=" * 60)
        print("Bayesian Optimization Summary")
        print("=" * 60)
        if results['best_temperature'] is not None:
            lat, lng = results['best_location']
            print(f"Best temperature found: {results['best_temperature']:.1f}°C")
            print(f"Location: ({lat:.4f}, {lng:.4f})")
        print(f"Total iterations: {len(results['guesses'])}")
        
        return results


def bayesian_optimization_search(n_iterations=50, config_path="config.yaml", seed=None):
    """
    Convenience function to run Bayesian Optimization search.
    
    Parameters
    ----------
    n_iterations : int
        Number of iterations (overrides config if provided).
    config_path : str
        Path to configuration file.
    seed : int or None
        Random seed for reproducibility.
        
    Returns
    -------
    results : dict
        Search results including guesses, temperatures, and best location.
    """
    model = BayesianOptimizationSearch(config_path=config_path)
    
    # Override n_iterations if provided
    if n_iterations is not None:
        model.n_iterations = n_iterations
    
    results = model.run_search(seed=seed)
    return results


if __name__ == "__main__":
    # Run Bayesian Optimization with default config
    results = bayesian_optimization_search(n_iterations=20, seed=42)
