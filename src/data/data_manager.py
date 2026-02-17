import csv
import os
from datetime import datetime

# ==============================================================================
# COORDINATE STRATEGY AGREEMENT:
# We use decimal (latitude, longitude).
# Lat: -90 to 90 (South to North)
# Lng: -180 to 180 (West to East)
# Example: Copenhagen is (55.6761, 12.5683)
# ==============================================================================

def is_valid_coordinate(lat, lng):
    """
    Checks if the given latitude and longitude are within global bounds.
    """
    return -90 <= lat <= 90 and -180 <= lng <= 180

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")

def save_result(lat, lng, temp, search_method):
    """
    Saves a single measurement to the shared results file.
    """
    timestamp = datetime.now().isoformat()
    
    file_exists = os.path.isfile(RESULTS_FILE)
    
    with open(RESULTS_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(["timestamp", "lat", "lng", "temp", "search_method"])
        
        writer.writerow([timestamp, lat, lng, temp, search_method])


def load_results():
    """
    Loads all shared results into a list of dictionaries.
    """
    if not os.path.isfile(RESULTS_FILE):
        return []
        
    results = []
    with open(RESULTS_FILE, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

def get_cached_result(lat, lng, tolerance=0.0001):
    """
    Checks if a result for the given coordinates already exists in the cache.
    tolerance: how close the coordinates must be to be considered a match.
    """
    results = load_results()
    for row in results:
        try:
            r_lat = float(row['lat'])
            r_lng = float(row['lng'])
            if abs(r_lat - lat) < tolerance and abs(r_lng - lng) < tolerance:
                return float(row['temp'])
        except (ValueError, KeyError):
            continue
    return None
