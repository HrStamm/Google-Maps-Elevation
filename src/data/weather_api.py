import requests
from src.data.data_manager import is_valid_coordinate, get_cached_result, save_result

import time

def fetch_temperature(lat, lng, search_method="unknown", use_cache=True, retries=3, backoff_factor=1):
    """
    Fetches the current temperature for a given latitude and longitude.
    Checks the local cache first unless use_cache is False.
    Includes a retry mechanism for API reliability.
    """
    if not is_valid_coordinate(lat, lng):
        print(f"Error: Invalid coordinates ({lat}, {lng}). Must be -90 <= lat <= 90 and -180 <= lng <= 180.")
        return None

    # 1. Check Cache
    if use_cache:
        cached_temp = get_cached_result(lat, lng)
        if cached_temp is not None:
            print(f"Cache hit for ({lat}, {lng}): {cached_temp}°C")
            return cached_temp
        
    # 2. If not in cache, fetch from API
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": "temperature_2m",
    }
    
    for attempt in range(retries):
        try:
            # Added a 10s timeout
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "current" in data:
                temp = data["current"]["temperature_2m"]
                
                # 3. Save to Cache
                save_result(lat, lng, temp, search_method)
                print(f"Fetched and cached ({lat}, {lng}): {temp}°C")
                
                return temp
            return None
            
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"Error fetching weather data after {retries} attempts: {e}")
                return None
    return None

if __name__ == "__main__":
    # Test with Copenhagen
    copenhagen = (55.6761, 12.5683)
    temp = fetch_temperature(*copenhagen)
    if temp is not None:
        print(f"Current temperature in Copenhagen: {temp}°C")
    else:
        print("Could not fetch temperature.")
