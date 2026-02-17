"""
Random Search Baseline
======================
Samples random (lat, lng) coordinates uniformly across the globe and
fetches the current temperature. This establishes a baseline for
comparison with Bayesian Optimization and manual search.
"""

import numpy as np
from src.data.weather_api import fetch_temperature


def random_search(n_iterations=50, seed=None):
    """
    Runs a random search over the globe to find the highest temperature.

    Parameters
    ----------
    n_iterations : int
        Number of random points to sample.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    results : list[dict]
        List of dicts with keys: iteration, lat, lng, temp, best_temp.
    """
    rng = np.random.default_rng(seed)
    results = []
    best_temp = -np.inf

    for i in range(n_iterations):
        lat = rng.uniform(-90, 90)
        lng = rng.uniform(-180, 180)

        temp = fetch_temperature(lat, lng, search_method="random_search", use_cache=False)

        if temp is not None and temp > best_temp:
            best_temp = temp

        results.append({
            "iteration": i + 1,
            "lat": round(lat, 4),
            "lng": round(lng, 4),
            "temp": temp,
            "best_temp": best_temp if best_temp > -np.inf else None,
        })

        print(f"[Random] Iter {i+1}/{n_iterations}  ({lat:.2f}, {lng:.2f})  "
              f"temp={temp}°C  best={best_temp:.1f}°C")

    return results


if __name__ == "__main__":
    results = random_search(n_iterations=20, seed=42)

    print("\n" + "=" * 60)
    print("Random Search Summary")
    print("=" * 60)
    best = max((r for r in results if r["temp"] is not None), key=lambda r: r["temp"])
    print(f"Best temperature found: {best['temp']}°C at ({best['lat']}, {best['lng']})")
    print(f"Total API calls: {len(results)}")
