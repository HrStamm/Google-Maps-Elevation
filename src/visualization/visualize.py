"""
Visualization utilities for Bayesian Optimization results.

This module provides functions to visualize the optimization process,
including the guesses made and the temperature distribution.
"""

import numpy as np
from typing import Dict, List, Tuple


def print_optimization_summary(results: Dict) -> None:
    """
    Print a detailed summary of the optimization results.
    
    Args:
        results: Dictionary containing optimization results with keys:
                 'guesses', 'temperatures', 'standardized_temps', 
                 'best_guess', 'best_temperature'
    """
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    
    # Best result
    print(f"\nBest Temperature: {results['best_temperature']:.2f}°C")
    print(f"Best Location: Lat={results['best_guess'][0]:.2f}, Lng={results['best_guess'][1]:.2f}")
    
    # Statistics
    temps = results['temperatures']
    print(f"\nTemperature Statistics:")
    print(f"  Mean: {np.mean(temps):.2f}°C")
    print(f"  Std:  {np.std(temps):.2f}°C")
    print(f"  Min:  {np.min(temps):.2f}°C")
    print(f"  Max:  {np.max(temps):.2f}°C")
    
    # Improvement over iterations
    print(f"\nImprovement Over Time:")
    cummax = np.maximum.accumulate(temps)
    for i in [0, len(temps)//4, len(temps)//2, 3*len(temps)//4, len(temps)-1]:
        if i < len(temps):
            print(f"  After guess {i+1}: {cummax[i]:.2f}°C")
    
    print("="*70)


def print_guess_table(results: Dict) -> None:
    """
    Print a formatted table of all guesses.
    
    Args:
        results: Dictionary containing optimization results
    """
    print("\nDetailed Guess Table:")
    print("-" * 80)
    print(f"{'#':<5} {'Latitude':<12} {'Longitude':<12} {'Temp (°C)':<12} {'Standardized':<12} {'Best So Far':<12}")
    print("-" * 80)
    
    cummax = np.maximum.accumulate(results['temperatures'])
    
    for i, (guess, temp, std_temp, best_so_far) in enumerate(
        zip(results['guesses'], results['temperatures'], 
            results['standardized_temps'], cummax),
        start=1
    ):
        lat, lng = guess
        marker = " *" if temp == results['best_temperature'] else ""
        print(f"{i:<5} {lat:<12.2f} {lng:<12.2f} {temp:<12.2f} {std_temp:<12.3f} {best_so_far:<12.2f}{marker}")
    
    print("-" * 80)
    print("* = Best guess found")


def create_ascii_map(results: Dict, width: int = 60, height: int = 20) -> None:
    """
    Create a simple ASCII map showing the guessed locations.
    
    Args:
        results: Dictionary containing optimization results
        width: Width of the ASCII map
        height: Height of the ASCII map
    """
    print("\nASCII Map of Guesses:")
    print("(* = best guess, numbers = order of guesses)")
    print()
    
    # Initialize map
    map_grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Map coordinates to grid positions
    guesses = results['guesses']
    best_guess = results['best_guess']
    
    for i, (lat, lng) in enumerate(guesses, start=1):
        # Convert lat/lng to grid position
        # lat: -90 to 90 -> height-1 to 0 (inverted)
        # lng: -180 to 180 -> 0 to width-1
        x = int((lng + 180) / 360 * (width - 1))
        y = int((90 - lat) / 180 * (height - 1))
        
        # Ensure within bounds
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        
        # Mark on map
        if (lat, lng) == best_guess:
            map_grid[y][x] = '*'
        else:
            # Use numbers for first 9, then letters
            if i <= 9:
                map_grid[y][x] = str(i)
            else:
                map_grid[y][x] = chr(ord('A') + i - 10)
    
    # Print map with borders
    print("  +" + "-" * width + "+")
    for row in map_grid:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * width + "+")
    print(f"  {'-180°':<{width//2}}{'180°':>{width//2}}")
    print(f"\n  90°N (top) to 90°S (bottom)")


def visualize_results(results: Dict, show_map: bool = True) -> None:
    """
    Comprehensive visualization of optimization results.
    
    Args:
        results: Dictionary containing optimization results
        show_map: Whether to show the ASCII map
    """
    print_optimization_summary(results)
    print_guess_table(results)
    
    if show_map:
        create_ascii_map(results)


if __name__ == "__main__":
    # Example usage with dummy data
    dummy_results = {
        'guesses': [(25.0, 15.0), (36.5, -117.0), (24.0, 45.0)],
        'temperatures': [35.5, 42.3, 38.7],
        'standardized_temps': [0.0, 1.0, 0.47],
        'best_guess': (36.5, -117.0),
        'best_temperature': 42.3
    }
    
    visualize_results(dummy_results)
