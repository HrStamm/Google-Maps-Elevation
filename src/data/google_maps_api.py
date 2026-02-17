import os
import googlemaps
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_elevation_client():
    """
    Initializes and returns a Google Maps client.
    Requires GOOGLE_MAPS_API_KEY to be set in the environment.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables. "
                         "Please check your .env file.")
    
    return googlemaps.Client(key=api_key)

def fetch_elevation(lat, lng):
    """
    Fetches elevation for a single coordinate.
    """
    gmaps = get_elevation_client()
    # elevation function returns a list of results
    result = gmaps.elevation((lat, lng))
    
    if result:
        return result[0]['elevation']
    return None

def fetch_elevations(locations):
    """
    Fetches elevations for multiple coordinates.
    locations: List of (lat, lng) tuples
    """
    gmaps = get_elevation_client()
    results = gmaps.elevation(locations)
    return results

if __name__ == "__main__":
    # Test with a known location (e.g., Mount Everest)
    try:
        mt_everest = (27.9881, 86.9250)
        elevation = fetch_elevation(*mt_everest)
        print(f"Elevation of Mt. Everest: {elevation} meters")
    except Exception as e:
        print(f"Error: {e}")
