import requests

def fetch_temperature(lat, lng):
    """
    Fetches the current temperature for a given latitude and longitude.
    Uses the Open-Meteo API (free, no API key required).
    """
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
            return data["current"]["temperature_2m"]
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
