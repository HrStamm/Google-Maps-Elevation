"""
Standardized Evaluation
=======================
Loads results from data/results.csv, groups by search_method, and
produces standard performance metrics:
  - Max Temperature vs. Iteration (convergence curve)
  - Sample Efficiency (iterations to reach X% of best known temp)
  - Summary statistics per method
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "results.csv")


def load_results(filepath=None):
    """Load results.csv into a list of dicts, sorted by timestamp."""
    filepath = filepath or RESULTS_FILE
    if not os.path.isfile(filepath):
        print(f"Results file not found: {filepath}")
        return []

    results = []
    with open(filepath, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["lat"] = float(row["lat"])
                row["lng"] = float(row["lng"])
                row["temp"] = float(row["temp"])
            except (ValueError, KeyError):
                continue
            results.append(row)
    
    # Sort by timestamp to ensure chronological order
    results.sort(key=lambda r: r.get("timestamp", ""))
    return results


def group_by_method(results):
    """Group results by search_method, preserving insertion order."""
    groups = defaultdict(list)
    for row in results:
        method = row.get("search_method", "unknown")
        groups[method].append(row)
    return dict(groups)


def compute_convergence(group):
    """
    Compute the running maximum temperature for a list of results.

    Returns
    -------
    iterations : list[int]
        1-based iteration numbers.
    best_temps : list[float]
        Running max temperature at each iteration.
    """
    iterations = []
    best_temps = []
    best = -np.inf
    for i, row in enumerate(group, start=1):
        if row["temp"] is not None:
            best = max(best, row["temp"])
        iterations.append(i)
        best_temps.append(best if best > -np.inf else np.nan)
    return iterations, best_temps


def sample_efficiency(group, threshold_pct=0.95, global_best=None):
    """
    Number of iterations needed to reach `threshold_pct` of the best
    known temperature.

    Parameters
    ----------
    group : list[dict]
        Results for one method.
    threshold_pct : float
        Fraction of the target temperature to reach (e.g. 0.95 = 95%).
    global_best : float or None
        If provided, use this as the target. Otherwise use the best
        within the group.

    Returns
    -------
    int or None
        Iteration number where threshold was first reached, or None.
    """
    target = global_best if global_best is not None else max(
        (r["temp"] for r in group if r["temp"] is not None), default=None
    )
    if target is None or target <= 0:
        return None

    threshold = threshold_pct * target
    best = -np.inf
    for i, row in enumerate(group, start=1):
        if row["temp"] is not None:
            best = max(best, row["temp"])
        if best >= threshold:
            return i
    return None


def summary_table(groups):
    """Print a summary table comparing all methods."""
    print(f"\n{'Method':<20} {'N Samples':>10} {'Max Temp (°C)':>14} {'Mean Temp (°C)':>15}")
    print("-" * 62)
    for method, rows in groups.items():
        temps = [r["temp"] for r in rows if r["temp"] is not None]
        if temps:
            print(f"{method:<20} {len(temps):>10} {max(temps):>14.1f} {np.mean(temps):>15.1f}")
        else:
            print(f"{method:<20} {len(rows):>10} {'N/A':>14} {'N/A':>15}")


def plot_convergence(groups, save_path=None):
    """
    Plot actual temperature and running max for each search method.

    Parameters
    ----------
    groups : dict
        Output of group_by_method().
    save_path : str or None
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, rows in groups.items():
        # Plot actual temperatures
        iters = list(range(1, len(rows) + 1))
        temps = [r["temp"] if r["temp"] is not None else np.nan for r in rows]
        ax.plot(iters, temps, marker="o", markersize=3, alpha=0.6, label=f"{method} (samples)")
        
        # Plot running maximum
        _, bests = compute_convergence(rows)
        ax.plot(iters, bests, marker="", linewidth=2, linestyle="--", label=f"{method} (max)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Samples and Running Maximum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved convergence plot to {save_path}")
    else:
        plt.show()


def plot_sample_locations(groups, save_path=None):
    """
    Scatter plot of all sampled locations, coloured by method.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for method, rows in groups.items():
        lats = [r["lat"] for r in rows]
        lngs = [r["lng"] for r in rows]
        ax.scatter(lngs, lats, s=15, alpha=0.7, label=method)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sampled Locations by Method")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved location plot to {save_path}")
    else:
        plt.show()


def evaluate(filepath=None, save_dir=None):
    """
    Run the full evaluation pipeline:
      1. Load results
      2. Print summary table
      3. Print sample efficiency
      4. Plot convergence curves
      5. Plot sample locations
    """
    results = load_results(filepath)
    if not results:
        print("No results to evaluate.")
        return

    groups = group_by_method(results)
    summary_table(groups)

    # Global best across all methods
    all_temps = [r["temp"] for r in results if r["temp"] is not None]
    global_best = max(all_temps) if all_temps else None

    print(f"\nGlobal best temperature: {global_best}°C")
    print(f"\nSample efficiency (iterations to reach 95% of global best):")
    for method, rows in groups.items():
        eff = sample_efficiency(rows, threshold_pct=0.95, global_best=global_best)
        print(f"  {method}: {eff if eff else 'Not reached'}")

    # Plots
    conv_path = os.path.join(save_dir, "convergence.png") if save_dir else None
    loc_path = os.path.join(save_dir, "sample_locations.png") if save_dir else None

    plot_convergence(groups, save_path=conv_path)
    plot_sample_locations(groups, save_path=loc_path)


if __name__ == "__main__":
    # Save plots in reports/figures/ if run directly
    report_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")
    os.makedirs(report_dir, exist_ok=True)
    evaluate(save_dir=report_dir)
