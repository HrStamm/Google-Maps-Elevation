import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import webbrowser

# ==============================================================================
# Simple World Map Visualization
# Generates a static PNG of the world with temperature sample points.
# Dark theme, clean coastlines, colored markers.
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(PROJECT_ROOT, "src", "data", "results.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "reports", "temperature_map.png")


def create_map():
    print(f"Loading data from {DATA_FILE}...")

    if not os.path.exists(DATA_FILE):
        print("Error: results.csv not found.")
        return None

    df = pd.read_csv(DATA_FILE)

    if len(df) == 0:
        print("Error: No data.")
        return None

    # ---- Style ----
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Dark ocean and land
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="#0a0a0a")
    ax.add_feature(cfeature.LAND, facecolor="#1a1a1a", edgecolor="#333333", linewidth=0.4)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#444444", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor="#2a2a2a", linewidth=0.3)

    # Remove the white outline around the map
    ax.spines['geo'].set_edgecolor("#222222")

    # ---- Plot Points ----
    lats = df["lat"].values
    lngs = df["lng"].values
    temps = df["temp"].values

    # Color scale: blue (cold) -> green -> red (hot)
    norm = mcolors.Normalize(vmin=temps.min() - 5, vmax=temps.max() + 5)
    cmap = plt.cm.RdYlBu_r  # Red = hot, Blue = cold

    sc = ax.scatter(
        lngs, lats,
        c=temps,
        cmap=cmap,
        norm=norm,
        s=80,
        edgecolors="#ffffff",
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # Highlight the best point
    best_idx = temps.argmax()
    ax.scatter(
        lngs[best_idx], lats[best_idx],
        s=200,
        facecolors="none",
        edgecolors="#ffcc00",
        linewidths=2,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    # ---- Colorbar ----
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.02, aspect=20)
    cbar.set_label("Temperature (°C)", fontsize=11, color="#aaaaaa")
    cbar.ax.tick_params(colors="#888888", labelsize=9)

    # ---- Title ----
    ax.set_title(
        f"Global Temperature Samples  •  {len(df)} points  •  Best: {temps.max():.1f}°C",
        fontsize=14,
        color="#cccccc",
        pad=15,
    )

    # ---- Save ----
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"Map saved to: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    output = create_map()
    if output:
        webbrowser.open("file://" + os.path.realpath(output))
