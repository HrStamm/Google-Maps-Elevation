"""
Gaussian Process Model for Bayesian Optimization
================================================
This module implements a Gaussian Process (GP) model using a Squared Exponential kernel
for finding the maximum temperature on a world map through Bayesian Optimization.

The model uses:
- Input: Standardized temperature values [0, 1] and axis bounds
- Kernel: Squared Exponential (RBF) kernel
- Acquisition: Upper Confidence Bound (UCB) for exploration-exploitation tradeoff
- Output: Guesses with their order and the best guess found
"""

import numpy as np
import yaml
import os
from typing import Tuple, List, Dict
from src.data.weather_api import fetch_temperature


class SquaredExponentialKernel:
    """
    Squared Exponential (SE) kernel, also known as Radial Basis Function (RBF) kernel.
    
    k(x, x') = variance * exp(-||x - x'||^2 / (2 * lengthscale^2))
    """
    
    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0):
        """
        Initialize the SE kernel.
        
        Args:
            variance: Signal variance (amplitude of the kernel)
            lengthscale: Lengthscale parameter (controls smoothness)
        """
        self.variance = variance
        self.lengthscale = lengthscale
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix between two sets of points.
        
        Args:
            X1: Array of shape (n1, d) - first set of points
            X2: Array of shape (n2, d) - second set of points
            
        Returns:
            Kernel matrix of shape (n1, n2)
        """
        # Compute squared Euclidean distances
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                 np.sum(X2**2, axis=1) - \
                 2 * np.dot(X1, X2.T)
        
        # Apply SE kernel formula
        return self.variance * np.exp(-0.5 * sqdist / self.lengthscale**2)


class GaussianProcessModel:
    """
    Gaussian Process model for Bayesian Optimization.
    
    This class handles the GP regression and provides predictions with uncertainty
    for guiding the optimization process.
    """
    
    def __init__(self, kernel: SquaredExponentialKernel, noise: float = 0.01):
        """
        Initialize the Gaussian Process model.
        
        Args:
            kernel: Kernel function to use
            noise: Observation noise variance
        """
        self.kernel = kernel
        self.noise = noise
        
        # Data storage
        self.X_train = None  # Training inputs (coordinates)
        self.y_train = None  # Training outputs (temperatures)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the GP model to observed data.
        
        Args:
            X: Training inputs of shape (n, 2) - coordinates (lat, lng)
            y: Training outputs of shape (n,) - temperatures
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at test points.
        
        Args:
            X_test: Test inputs of shape (m, 2)
            
        Returns:
            mean: Predicted mean of shape (m,)
            variance: Predicted variance of shape (m,)
        """
        # If no training data, return prior
        if self.X_train is None or len(self.X_train) == 0:
            return np.zeros(len(X_test)), np.ones(len(X_test)) * self.kernel.variance
        
        # Compute kernel matrices
        K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        K_star = self.kernel(self.X_train, X_test)
        K_star_star = self.kernel(X_test, X_test)
        
        # Solve for the mean
        K_inv_y = np.linalg.solve(K, self.y_train)
        mean = K_star.T @ K_inv_y
        
        # Solve for the variance
        K_inv_K_star = np.linalg.solve(K, K_star)
        variance = np.diag(K_star_star - K_star.T @ K_inv_K_star)
        
        return mean, variance


class BayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Process for finding maximum temperature.
    
    Uses Upper Confidence Bound (UCB) acquisition function for balancing
    exploration and exploitation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Bayesian Optimizer.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Extract parameters from config
        kernel_config = self.config['kernel']
        opt_config = self.config['optimization']
        bounds_config = self.config['bounds']
        
        # Initialize kernel and GP model
        kernel = SquaredExponentialKernel(
            variance=kernel_config['variance'],
            lengthscale=kernel_config['lengthscale']
        )
        self.gp_model = GaussianProcessModel(kernel, noise=kernel_config['noise'])
        
        # Optimization parameters
        self.n_guesses = opt_config['n_guesses']
        self.exploration = opt_config['exploration']
        
        # Map bounds
        self.bounds = np.array([
            [bounds_config['lat_min'], bounds_config['lat_max']],
            [bounds_config['lng_min'], bounds_config['lng_max']]
        ])
        
        # Storage for guesses
        self.guesses: List[Tuple[float, float]] = []
        self.temperatures: List[float] = []
        self.standardized_temps: List[float] = []
        
        # Temperature normalization bounds (will be updated as we collect data)
        self.temp_min = None
        self.temp_max = None
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _standardize_temperature(self, temp: float) -> float:
        """
        Standardize temperature to [0, 1] range.
        
        Args:
            temp: Raw temperature value
            
        Returns:
            Standardized temperature in [0, 1]
        """
        # Update min/max
        if self.temp_min is None or temp < self.temp_min:
            self.temp_min = temp
        if self.temp_max is None or temp > self.temp_max:
            self.temp_max = temp
        
        # Avoid division by zero
        if self.temp_max == self.temp_min:
            return 0.5
        
        return (temp - self.temp_min) / (self.temp_max - self.temp_min)
    
    def _acquisition_ucb(self, X: np.ndarray) -> np.ndarray:
        """
        Upper Confidence Bound (UCB) acquisition function.
        
        UCB = mean + exploration * sqrt(variance)
        
        Args:
            X: Candidate points of shape (n, 2)
            
        Returns:
            UCB values of shape (n,)
        """
        mean, variance = self.gp_model.predict(X)
        std = np.sqrt(np.maximum(variance, 1e-8))  # Ensure numerical stability
        return mean + self.exploration * std
    
    def _optimize_acquisition(self) -> Tuple[float, float]:
        """
        Find the point that maximizes the acquisition function.
        
        Uses random search over the map bounds.
        
        Returns:
            Best coordinates (lat, lng)
        """
        # Generate random candidate points
        n_candidates = 10000
        candidates = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(n_candidates, 2)
        )
        
        # Evaluate acquisition function
        ucb_values = self._acquisition_ucb(candidates)
        
        # Return best candidate
        best_idx = np.argmax(ucb_values)
        return tuple(candidates[best_idx])
    
    def optimize(self, initial_guesses: List[Tuple[float, float]] = None) -> Dict:
        """
        Run Bayesian Optimization to find the maximum temperature.
        
        Args:
            initial_guesses: Optional list of initial (lat, lng) coordinates to evaluate
            
        Returns:
            Dictionary with:
                - 'guesses': List of all guessed coordinates
                - 'temperatures': List of actual temperatures
                - 'standardized_temps': List of standardized temperatures
                - 'best_guess': Coordinates of the best guess
                - 'best_temperature': Best temperature found
        """
        # Handle initial guesses
        if initial_guesses is None:
            # Start with a random guess
            initial_guesses = [
                tuple(np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]))
            ]
        
        # Evaluate initial guesses
        for guess in initial_guesses[:self.n_guesses]:
            lat, lng = guess
            temp = fetch_temperature(lat, lng, search_method="bayesian_optimization")
            
            if temp is not None:
                self.guesses.append(guess)
                self.temperatures.append(temp)
                standardized = self._standardize_temperature(temp)
                self.standardized_temps.append(standardized)
                
                print(f"Guess {len(self.guesses)}/{self.n_guesses}: "
                      f"({lat:.2f}, {lng:.2f}) -> {temp:.2f}°C (standardized: {standardized:.3f})")
        
        # Main optimization loop
        n_remaining = self.n_guesses - len(self.guesses)
        for i in range(n_remaining):
            # Update GP model with current data
            X_train = np.array(self.guesses)
            y_train = np.array(self.standardized_temps)
            self.gp_model.fit(X_train, y_train)
            
            # Find next point to evaluate
            next_guess = self._optimize_acquisition()
            lat, lng = next_guess
            
            # Evaluate the temperature
            temp = fetch_temperature(lat, lng, search_method="bayesian_optimization")
            
            if temp is not None:
                self.guesses.append(next_guess)
                self.temperatures.append(temp)
                standardized = self._standardize_temperature(temp)
                self.standardized_temps.append(standardized)
                
                print(f"Guess {len(self.guesses)}/{self.n_guesses}: "
                      f"({lat:.2f}, {lng:.2f}) -> {temp:.2f}°C (standardized: {standardized:.3f})")
        
        # Find best result
        best_idx = np.argmax(self.temperatures)
        best_guess = self.guesses[best_idx]
        best_temp = self.temperatures[best_idx]
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best temperature found: {best_temp:.2f}°C")
        print(f"Best location: ({best_guess[0]:.2f}, {best_guess[1]:.2f})")
        print(f"Total guesses: {len(self.guesses)}")
        print(f"{'='*60}")
        
        return {
            'guesses': self.guesses,
            'temperatures': self.temperatures,
            'standardized_temps': self.standardized_temps,
            'best_guess': best_guess,
            'best_temperature': best_temp
        }


if __name__ == "__main__":
    # Example usage
    print("Starting Bayesian Optimization for Maximum Temperature Search")
    print("="*60)
    
    # Initialize optimizer
    optimizer = BayesianOptimizer(config_path="config.yaml")
    
    # Run optimization
    results = optimizer.optimize()
    
    # Print summary
    print("\nAll guesses:")
    for i, (guess, temp) in enumerate(zip(results['guesses'], results['temperatures']), 1):
        print(f"{i}. ({guess[0]:.2f}, {guess[1]:.2f}) -> {temp:.2f}°C")
