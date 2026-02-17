from src.data.google_maps_api import fetch_elevation
from src.data.weather_api import fetch_temperature

def main():
    # Example coordinates (somewhere in the Sahara)
    coords = (25.0, 15.0)
    
    print("--- First evaluation (API call) ---")
    temp1 = fetch_temperature(*coords, search_method="initial_test")
    
    print("\n--- Second evaluation (Should be a CACHE HIT) ---")
    temp2 = fetch_temperature(*coords, search_method="initial_test")
    
    if temp1 is not None:
        print(f"\nFinal result: {temp1}°C")
    else:
        print("\nCould not fetch weather data.")

if __name__ == "__main__":
    main()
