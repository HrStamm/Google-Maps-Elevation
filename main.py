from src.data.google_maps_api import fetch_elevation
from src.data.weather_api import fetch_temperature
from src.models.train_model import bayesian_optimization_search

def main():
    # Example: Run Bayesian Optimization to find the hottest location
    print("Running Bayesian Optimization to find the highest temperature...\n")
    
    # Run the search with 20 iterations for demonstration
    results = bayesian_optimization_search(n_iterations=20, seed=42)
    
    # The results contain all the information about the search
    print("\n" + "=" * 60)
    print("Search complete!")
    print("=" * 60)
    
    if results['best_temperature'] is not None:
        lat, lng = results['best_location']
        print(f"\n🔥 Best location found: ({lat:.4f}, {lng:.4f})")
        print(f"🌡️  Temperature: {results['best_temperature']:.1f}°C")
        print(f"\n📊 Total guesses: {len(results['guesses'])}")

if __name__ == "__main__":
    main()
