from src.data.google_maps_api import fetch_elevation
from src.data.weather_api import fetch_temperature

def main():
    # Example coordinates
    coords = (55.106423, 13.489815)
    
    try:
        # Fetch Elevation
        elevation = fetch_elevation(*coords)
        
        # Fetch Temperature
        temperature = fetch_temperature(*coords)
        
        print(f"Results for coordinates {coords}:")
        
        if elevation is not None:
            print(f"- Elevation: {elevation:.2f} meters")
        else:
            print("- Elevation: Could not fetch.")
            
        if temperature is not None:
            print(f"- Temperature: {temperature:.1f}°C")
        else:
            print("- Temperature: Could not fetch.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure you have set GOOGLE_MAPS_API_KEY in your .env file.")

if __name__ == "__main__":
    main()
