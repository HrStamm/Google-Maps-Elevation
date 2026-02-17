"""
Main script for running Bayesian Optimization to find maximum temperature.

This script demonstrates the use of the BayesianOptimizer class to find
the location with the highest temperature on a world map.
"""

from src.models.bayesian_optimizer import BayesianOptimizer
from src.visualization.visualize import visualize_results


def main():
    """
    Run Bayesian Optimization to find the location with maximum temperature.
    """
    print("="*70)
    print("Bayesian Optimization for Maximum Temperature Search")
    print("="*70)
    print()
    
    # Initialize the optimizer (loads config from config.yaml)
    optimizer = BayesianOptimizer(config_path="config.yaml")
    
    # Optional: Provide initial guesses (if you have domain knowledge)
    # For example, we know deserts tend to be hot:
    # - Sahara Desert: (25.0, 15.0)
    # - Arabian Desert: (24.0, 45.0)
    # - Death Valley: (36.5, -117.0)
    initial_guesses = [
        (25.0, 15.0),  # Sahara Desert region
    ]
    
    # Run the optimization
    results = optimizer.optimize(initial_guesses=initial_guesses)
    
    # Visualize results
    visualize_results(results, show_map=True)


if __name__ == "__main__":
    main()

