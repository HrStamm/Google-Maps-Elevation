import pandas as pd
import os
import webbrowser
import json

# ==============================================================================
# BO Playback Generator
# Creates an animated HTML page that replays the Bayesian Optimization process
# point by point on a dark world map.
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(PROJECT_ROOT, "src", "data", "results.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "reports", "playback.html")


def generate_playback():
    print(f"Loading data from {DATA_FILE}...")

    if not os.path.exists(DATA_FILE):
        print("Error: results.csv not found.")
        return None

    df = pd.read_csv(DATA_FILE)
    if len(df) == 0:
        print("Error: No data.")
        return None

    # Build the points list in order
    points = []
    for _, row in df.iterrows():
        points.append({
            "lat": round(row["lat"], 4),
            "lng": round(row["lng"], 4),
            "temp": row["temp"],
            "method": row["search_method"],
            "ts": row["timestamp"],
        })

    points_json = json.dumps(points)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BO Playback – Global Temperature Search</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: #0d0d0d;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  /* ---- Header ---- */
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 28px;
    background: #111;
    border-bottom: 1px solid #222;
    z-index: 1000;
  }}
  .header h1 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    letter-spacing: 2px;
    text-transform: uppercase;
  }}
  .header h1 span {{ color: #00ff66; }}

  .stats {{
    display: flex;
    gap: 30px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
  }}
  .stat-label {{ color: #666; margin-right: 6px; }}
  .stat-value {{ color: #fff; }}
  .stat-value.best {{ color: #ffcc00; }}
  .stat-value.green {{ color: #00ff66; }}

  /* ---- Map ---- */
  #map {{
    flex: 1;
    z-index: 1;
  }}

  /* ---- Controls Bar ---- */
  .controls {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 28px;
    background: #111;
    border-top: 1px solid #222;
    z-index: 1000;
  }}

  .btn {{
    background: #222;
    border: 1px solid #444;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    padding: 6px 16px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
    min-width: 70px;
    text-align: center;
  }}
  .btn:hover {{ background: #333; }}
  .btn.active {{ background: #1a3a1a; border-color: #00ff66; color: #00ff66; }}

  .timeline {{
    flex: 1;
    -webkit-appearance: none;
    height: 4px;
    background: #333;
    border-radius: 2px;
    outline: none;
  }}
  .timeline::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #00ff66;
    cursor: pointer;
    box-shadow: 0 0 8px #00ff6688;
  }}

  .speed-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #666;
  }}
  .speed-select {{
    background: #222;
    color: #fff;
    border: 1px solid #444;
    padding: 4px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }}

  .iter-display {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #888;
    min-width: 80px;
    text-align: right;
  }}

  /* ---- Leaflet overrides ---- */
  .leaflet-control-attribution {{ display: none !important; }}
  .leaflet-control-zoom {{ display: none !important; }}

  /* ---- Pulse animation for new points ---- */
  @keyframes pulse {{
    0% {{ transform: scale(1); opacity: 1; }}
    50% {{ transform: scale(2.2); opacity: 0.4; }}
    100% {{ transform: scale(1); opacity: 1; }}
  }}
  .pulse {{
    animation: pulse 0.6s ease-out;
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>BO Playback: <span>Global Temperature Search</span></h1>
  <div class="stats">
    <div><span class="stat-label">Iteration:</span><span class="stat-value green" id="s-iter">0</span></div>
    <div><span class="stat-label">Best Temp:</span><span class="stat-value best" id="s-best">—</span></div>
    <div><span class="stat-label">Current:</span><span class="stat-value" id="s-cur">—</span></div>
    <div><span class="stat-label">Method:</span><span class="stat-value" id="s-method">—</span></div>
  </div>
</div>

<!-- Map -->
<div id="map"></div>

<!-- Controls -->
<div class="controls">
  <button class="btn" id="btn-play" onclick="togglePlay()">▶ Play</button>
  <button class="btn" onclick="resetPlayback()">⟲ Reset</button>
  <input type="range" class="timeline" id="timeline" min="0" max="{len(points) - 1}" value="0" oninput="seekTo(this.value)">
  <div class="iter-display" id="iter-label">0 / {len(points)}</div>
  <span class="speed-label">Speed:</span>
  <select class="speed-select" id="speed" onchange="updateSpeed()">
    <option value="2000">0.5x</option>
    <option value="1000" selected>1x</option>
    <option value="500">2x</option>
    <option value="200">5x</option>
  </select>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  // ---- Data ----
  const points = {points_json};
  const totalPoints = points.length;

  // ---- Map Setup ----
  const map = L.map('map', {{
    center: [20, 0],
    zoom: 2,
    zoomControl: false,
    attributionControl: false,
  }});

  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    subdomains: 'abcd',
    maxZoom: 19,
  }}).addTo(map);

  // ---- State ----
  let currentIdx = -1;
  let playing = false;
  let interval = null;
  let speed = 1000;
  let markers = [];
  let bestTemp = -Infinity;
  let bestMarker = null;

  function getColor(temp) {{
    if (temp >= 30) return '#ff3333';
    if (temp >= 20) return '#ff8800';
    if (temp >= 10) return '#ffcc00';
    if (temp >= 0)  return '#44aaff';
    return '#6666ff';
  }}

  function showPoint(idx) {{
    if (idx < 0 || idx >= totalPoints) return;

    const p = points[idx];
    currentIdx = idx;

    // Create marker
    const color = getColor(p.temp);
    const marker = L.circleMarker([p.lat, p.lng], {{
      radius: 7,
      color: color,
      fillColor: color,
      fillOpacity: 0.8,
      weight: 1.5,
    }}).addTo(map);

    marker.bindPopup(
      '<div style="font-family:JetBrains Mono,monospace;font-size:12px;">' +
      '<b>' + p.temp + '°C</b><br>' +
      p.lat + ', ' + p.lng + '<br>' +
      '<span style="color:#888">' + p.method + '</span></div>'
    );

    markers.push(marker);

    // Check if this is the new best
    if (p.temp > bestTemp) {{
      bestTemp = p.temp;
      // Remove old best ring
      if (bestMarker) map.removeLayer(bestMarker);
      // Add gold ring around best
      bestMarker = L.circleMarker([p.lat, p.lng], {{
        radius: 14,
        color: '#ffcc00',
        fillColor: 'transparent',
        fillOpacity: 0,
        weight: 2.5,
        dashArray: '4 4',
      }}).addTo(map);
    }}

    // Update stats
    document.getElementById('s-iter').textContent = idx + 1;
    document.getElementById('s-best').textContent = bestTemp.toFixed(1) + '°C';
    document.getElementById('s-cur').textContent = p.temp + '°C';
    document.getElementById('s-method').textContent = p.method;
    document.getElementById('timeline').value = idx;
    document.getElementById('iter-label').textContent = (idx + 1) + ' / ' + totalPoints;
  }}

  function togglePlay() {{
    const btn = document.getElementById('btn-play');
    if (playing) {{
      clearInterval(interval);
      playing = false;
      btn.textContent = '▶ Play';
      btn.classList.remove('active');
    }} else {{
      playing = true;
      btn.textContent = '⏸ Pause';
      btn.classList.add('active');
      interval = setInterval(() => {{
        if (currentIdx >= totalPoints - 1) {{
          clearInterval(interval);
          playing = false;
          btn.textContent = '▶ Play';
          btn.classList.remove('active');
          return;
        }}
        showPoint(currentIdx + 1);
      }}, speed);
    }}
  }}

  function resetPlayback() {{
    clearInterval(interval);
    playing = false;
    document.getElementById('btn-play').textContent = '▶ Play';
    document.getElementById('btn-play').classList.remove('active');

    // Remove all markers
    markers.forEach(m => map.removeLayer(m));
    markers = [];
    if (bestMarker) {{ map.removeLayer(bestMarker); bestMarker = null; }}

    currentIdx = -1;
    bestTemp = -Infinity;

    document.getElementById('s-iter').textContent = '0';
    document.getElementById('s-best').textContent = '—';
    document.getElementById('s-cur').textContent = '—';
    document.getElementById('s-method').textContent = '—';
    document.getElementById('timeline').value = 0;
    document.getElementById('iter-label').textContent = '0 / ' + totalPoints;
  }}

  function seekTo(idx) {{
    idx = parseInt(idx);
    // Reset and replay up to idx
    markers.forEach(m => map.removeLayer(m));
    markers = [];
    if (bestMarker) {{ map.removeLayer(bestMarker); bestMarker = null; }}
    currentIdx = -1;
    bestTemp = -Infinity;

    for (let i = 0; i <= idx; i++) {{
      showPoint(i);
    }}
  }}

  function updateSpeed() {{
    speed = parseInt(document.getElementById('speed').value);
    if (playing) {{
      clearInterval(interval);
      interval = setInterval(() => {{
        if (currentIdx >= totalPoints - 1) {{
          clearInterval(interval);
          playing = false;
          document.getElementById('btn-play').textContent = '▶ Play';
          document.getElementById('btn-play').classList.remove('active');
          return;
        }}
        showPoint(currentIdx + 1);
      }}, speed);
    }}
  }}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(html)

    print(f"Playback saved to: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    output = generate_playback()
    if output:
        webbrowser.open("file://" + os.path.realpath(output))
