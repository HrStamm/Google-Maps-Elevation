"""
Manual Search Utility
=====================
Interactive tool that lets a human teammate input their best guesses
for the hottest point on Earth. Results are logged with
search_method='manual_search' so they can be compared with BO and
random search.
"""

from src.data.weather_api import fetch_temperature


def manual_search(max_guesses=None):
    """
    Interactive loop that prompts the user for (lat, lng) guesses.

    Parameters
    ----------
    max_guesses : int or None
        Maximum number of guesses allowed. None = unlimited.

    Returns
    -------
    results : list[dict]
        List of dicts with keys: iteration, lat, lng, temp, best_temp.
    """
    results = []
    best_temp = None
    iteration = 0

    print("=" * 60)
    print("🌍  Manual Temperature Search")
    print("=" * 60)
    print("Enter latitude and longitude to query the temperature.")
    print("Type 'q' to quit.\n")

    while True:
        if max_guesses is not None and iteration >= max_guesses:
            print(f"\nReached maximum of {max_guesses} guesses.")
            break

        try:
            user_input = input(f"Guess #{iteration + 1} – Enter lat, lng (e.g. 25.0, 15.0): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() == "q":
            break

        # Parse input
        try:
            parts = user_input.replace(",", " ").split()
            lat, lng = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            print("  ⚠️  Invalid input. Please enter two numbers separated by a comma or space.")
            continue

        # Validate
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            print("  ⚠️  Coordinates out of bounds! lat: [-90, 90], lng: [-180, 180]")
            continue

        iteration += 1
        temp = fetch_temperature(lat, lng, search_method="manual_search", use_cache=False)

        if temp is not None:
            if best_temp is None or temp > best_temp:
                best_temp = temp
            print(f"  🌡️  Temperature: {temp}°C   (Best so far: {best_temp}°C)")
        else:
            print("  ❌  Could not fetch temperature for those coordinates.")

        results.append({
            "iteration": iteration,
            "lat": round(lat, 4),
            "lng": round(lng, 4),
            "temp": temp,
            "best_temp": best_temp,
        })

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("Manual Search Summary")
        print("=" * 60)
        valid = [r for r in results if r["temp"] is not None]
        if valid:
            best = max(valid, key=lambda r: r["temp"])
            print(f"Best temperature found: {best['temp']}°C at ({best['lat']}, {best['lng']})")
        print(f"Total guesses: {len(results)}")

    return results


if __name__ == "__main__":
    manual_search()
