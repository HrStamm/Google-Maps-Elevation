"""
Example script demonstrating different usage patterns of the Bayesian Optimizer.

This script shows:
1. Basic usage with default configuration
2. Custom configuration
3. Different initial guess strategies
"""

from src.models.bayesian_optimizer import BayesianOptimizer
from src.visualization.visualize import visualize_results
import yaml


def example_basic():
    """
    Example 1: Basic usage with default configuration.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    optimizer = BayesianOptimizer(config_path="config.yaml")
    results = optimizer.optimize()
    
    return results


def example_custom_config():
    """
    Example 2: Usage with custom configuration.
    
    This example creates a temporary config with more exploration.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Configuration")
    print("="*70)
    
    # Create custom config
    custom_config = {
        'kernel': {
            'variance': 1.5,
            'lengthscale': 15.0,
            'noise': 0.005
        },
        'optimization': {
            'n_guesses': 10,  # Fewer guesses for quick demo
            'exploration': 3.0  # More exploration
        },
        'bounds': {
            'lat_min': -90,
            'lat_max': 90,
            'lng_min': -180,
            'lng_max': 180
        }
    }
    
    # Save temporary config
    config_path = "config_custom.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(custom_config, f)
    
    print("Using custom configuration:")
    print(f"  - Lengthscale: {custom_config['kernel']['lengthscale']}")
    print(f"  - Exploration: {custom_config['optimization']['exploration']}")
    print(f"  - Number of guesses: {custom_config['optimization']['n_guesses']}")
    print()
    
    optimizer = BayesianOptimizer(config_path=config_path)
    results = optimizer.optimize()
    
    return results


def example_strategic_initial_guesses():
    """
    Example 3: Using strategic initial guesses.
    
    This example demonstrates using domain knowledge to start
    the search in likely hot locations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Strategic Initial Guesses")
    print("="*70)
    
    # Known hot locations
    initial_guesses = [
        (25.0, 15.0),    # Sahara Desert (Libya/Chad border)
        (24.0, 45.0),    # Arabian Desert (Saudi Arabia)
        (36.5, -117.0),  # Death Valley, USA
        (30.0, 40.0),    # Iraq/Saudi Arabia
        (-24.0, 133.0),  # Australian Outback
    ]
    
    print("Starting with strategic guesses in known hot regions:")
    for i, (lat, lng) in enumerate(initial_guesses, 1):
        print(f"  {i}. ({lat:.1f}, {lng:.1f})")
    print()
    
    optimizer = BayesianOptimizer(config_path="config.yaml")
    results = optimizer.optimize(initial_guesses=initial_guesses)
    
    return results


def example_focused_search():
    """
    Example 4: Focused search in a specific region.
    
    This example restricts the search to a specific geographic area.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Focused Search (Africa only)")
    print("="*70)
    
    # Create config for Africa only
    africa_config = {
        'kernel': {
            'variance': 1.0,
            'lengthscale': 5.0,  # Smaller lengthscale for regional search
            'noise': 0.01
        },
        'optimization': {
            'n_guesses': 15,
            'exploration': 2.0
        },
        'bounds': {
            'lat_min': -35,   # South Africa
            'lat_max': 37,    # Mediterranean coast
            'lng_min': -18,   # West coast
            'lng_max': 52     # East coast (Somalia)
        }
    }
    
    config_path = "config_africa.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(africa_config, f)
    
    print("Searching only within Africa:")
    print(f"  Lat: {africa_config['bounds']['lat_min']} to {africa_config['bounds']['lat_max']}")
    print(f"  Lng: {africa_config['bounds']['lng_min']} to {africa_config['bounds']['lng_max']}")
    print()
    
    optimizer = BayesianOptimizer(config_path=config_path)
    results = optimizer.optimize()
    
    return results


def main():
    """
    Run all examples.
    
    Uncomment the example you want to run.
    """
    # Choose which example to run:
    
    # results = example_basic()
    # results = example_custom_config()
    results = example_strategic_initial_guesses()
    # results = example_focused_search()
    
    # Visualize final results
    print("\n" + "="*70)
    print("FINAL VISUALIZATION")
    print("="*70)
    visualize_results(results, show_map=True)


if __name__ == "__main__":
    main()
