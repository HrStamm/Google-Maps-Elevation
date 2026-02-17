from google_maps_api import fetch_elevation

def main():
    # Example: Fetch elevation for Copenhagen (lat: 55.6761, lng: 12.5683)
    copenhagen = (55.106423, 12.489815)
    
    try:
        elevation = fetch_elevation(*copenhagen)
        if elevation is not None:
            print(f"Elevation of Copenhagen: {elevation:.2f} meters")
        else:
            print("Could not fetch elevation.")
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure you have set GOOGLE_MAPS_API_KEY in your .env file.")

if __name__ == "__main__":
    main()
