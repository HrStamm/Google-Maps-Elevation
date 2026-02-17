import requests
from src.data.data_manager import is_valid_coordinate, get_cached_result, save_result

def fetch_temperature(lat, lng, search_method="unknown", use_cache=True):
    """
    Fetches the current temperature for a given latitude and longitude.
    Checks the local cache first unless use_cache is False.
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
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "current" in data:
            temp = data["current"]["temperature_2m"]
            
            # 3. Save to Cache
            save_result(lat, lng, temp, search_method)
            print(f"Fetched and cached ({lat}, {lng}): {temp}°C")
            
            return temp
        return None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

if __name__ == "__main__":
    # Test with Copenhagen
    copenhagen = (55.6761, 12.5683)
    temp = fetch_temperature(*copenhagen)
    if temp is not None:
        print(f"Current temperature in Copenhagen: {temp}°C")
    else:
        print("Could not fetch temperature.")
