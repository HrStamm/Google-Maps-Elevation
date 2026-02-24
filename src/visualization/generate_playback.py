import pandas as pd
import numpy as np
import os
import webbrowser
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ==============================================================================
# BO Playback with Dynamic Overlays
# Replays the optimization AND allows toggling between 
# Temperature (μ), Uncertainty (σ), and Acquisition overlays.
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(PROJECT_ROOT, "src", "data", "results.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "reports", "playback.html")

TOTAL_BUDGET = 20
GRID_STEP = 2  # Finer grid for smoother heatmap

# ==============================================================================
# GP Model Parameters (Tweak these here)
# ==============================================================================
GP_LENGTH_SCALE = 80.0
GP_LENGTH_SCALE_BOUNDS = (30.0, 500.0)
GP_ALPHA = 0.5
GP_KAPPA = 2.5  # Controls exploration in Acquisition (UCB)


def compute_gp_frames(df):
    """Pre-compute μ, σ, and Acquisition grids for every iteration."""
    lat_grid = np.arange(-82, 82 + GRID_STEP, GRID_STEP)
    lng_grid = np.arange(-180, 180 + GRID_STEP, GRID_STEP)
    lng_mesh, lat_mesh = np.meshgrid(lng_grid, lat_grid)
    grid_points = np.column_stack([lat_mesh.ravel(), lng_mesh.ravel()])

    cells = []
    for i in range(len(lat_grid)):
        for j in range(len(lng_grid)):
            cells.append({"lat": float(lat_grid[i]), "lng": float(lng_grid[j])})

    raw_mu = []
    raw_sigma = []
    raw_acq = []

    for step in range(1, len(df) + 1):
        X = df[["lat", "lng"]].values[:step]
        y = df["temp"].values[:step]

        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=GP_LENGTH_SCALE, length_scale_bounds=GP_LENGTH_SCALE_BOUNDS)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=GP_ALPHA, normalize_y=True)
        gp.fit(X, y)

        mu, sigma = gp.predict(grid_points, return_std=True)
        acq = mu + GP_KAPPA * sigma  # UCB

        raw_mu.append(mu)
        raw_sigma.append(sigma)
        raw_acq.append(acq)
        print(f"  Frame {step}/{len(df)} computed")

    # Global normalization
    def normalize_frames(raw_list):
        all_vals = np.concatenate(raw_list)
        v_min, v_max = all_vals.min(), all_vals.max()
        frames = []
        for v in raw_list:
            if v_max > v_min:
                normed = ((v - v_min) / (v_max - v_min)).tolist()
            else:
                normed = [0.0] * len(v)
            frames.append([round(val, 3) for val in normed])
        return frames, float(v_min), float(v_max)

    frames_mu, mu_min, mu_max = normalize_frames(raw_mu)
    frames_sigma, s_min, s_max = normalize_frames(raw_sigma)
    frames_acq, a_min, a_max = normalize_frames(raw_acq)

    return cells, frames_mu, mu_min, mu_max, frames_sigma, s_min, s_max, frames_acq, a_min, a_max


def generate_playback():
    print(f"Loading data from {DATA_FILE}...")

    if not os.path.exists(DATA_FILE):
        print("Error: results.csv not found.")
        return None

    df = pd.read_csv(DATA_FILE)
    if len(df) < 2:
        print("Error: Need at least 2 data points.")
        return None

    points = []
    for _, row in df.iterrows():
        points.append({
            "lat": round(row["lat"], 4),
            "lng": round(row["lng"], 4),
            "temp": row["temp"],
            "method": row["search_method"],
            "ts": row["timestamp"],
        })

    print("Computing GP frames...")
    (cells, frames_mu, mu_min, mu_max, 
     frames_sigma, s_min, s_max, 
     frames_acq, a_min, a_max) = compute_gp_frames(df)

    # Inject into HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BO Playback – Global Temperature</title>
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

  #map {{ flex: 1; z-index: 1; }}

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

  .styled-select {{
    background: #222;
    color: #fff;
    border: 1px solid #444;
    padding: 6px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    outline: none;
  }}

  .iter-display {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #888;
    min-width: 80px;
    text-align: right;
  }}

  .leaflet-control-attribution {{ display: none !important; }}
  .leaflet-control-zoom {{ display: none !important; }}

  .legend {{
    position: fixed;
    top: 50%;
    right: 28px;
    transform: translateY(-50%);
    background: #111e;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 14px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    z-index: 1001;
    display: none;
    flex-direction: column;
    align-items: center;
  }}
  .legend.visible {{ display: flex; }}
  .legend-title {{
    color: #888;
    margin-bottom: 12px;
    letter-spacing: 1px;
    text-align: center;
    font-weight: bold;
  }}
  .legend-wrapper {{ display: flex; align-items: stretch; height: 180px; justify-content: center; width: 100%; }}
  .legend-bar {{
    width: 16px;
    border-radius: 3px;
  }}
  .legend-labels {{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    color: #ccc;
    font-size: 11px;
    margin-left: 12px;
    font-weight: bold;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>BO Playback: <span>Global Temperature Search</span></h1>
  <div class="stats">
    <div><span class="stat-label">Iteration:</span><span class="stat-value green" id="s-iter">0</span></div>
    <div><span class="stat-label">Best Temp:</span><span class="stat-value best" id="s-best">—</span></div>
    <div><span class="stat-label">Current:</span><span class="stat-value" id="s-cur">—</span></div>
    <div><span class="stat-label">Method:</span><span class="stat-value" id="s-method">—</span></div>
  </div>
</div>

<div id="map"></div>

<div class="legend" id="legend">
  <div class="legend-title" id="legend-title">Legend</div>
  <div class="legend-wrapper">
    <div class="legend-bar" id="legend-bar"></div>
    <div class="legend-labels" id="legend-labels">
      <!-- Injected by JS -->
    </div>
  </div>
</div>

<div class="controls">
  <button class="btn" onclick="stepBack()">◀ Prev</button>
  <button class="btn" id="btn-play" onclick="togglePlay()">▶ Play</button>
  <button class="btn" onclick="stepForward()">Next ▶</button>
  <button class="btn" onclick="resetPlayback()">⟲ Reset</button>
  
  <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#666;margin-left:15px;">Layer:</span>
  <select class="styled-select" id="layer-select" onchange="changeLayer()">
    <option value="none" selected>None</option>
    <option value="mu">Predicted Temp (μ)</option>
    <option value="sigma">Uncertainty (σ)</option>
    <option value="acq">Acquisition (UCB)</option>
  </select>

  <input type="range" class="timeline" id="timeline" min="0" max="{len(df) - 1}" value="0" oninput="seekTo(this.value)">
  <div class="iter-display" id="iter-label">0 / {len(df)}</div>
  
  <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#666;">Speed:</span>
  <select class="styled-select" id="speed" onchange="updateSpeed()">
    <option value="2000">0.5x</option>
    <option value="1000" selected>1x</option>
    <option value="500">2x</option>
    <option value="200">5x</option>
  </select>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const points = {json.dumps(points)};
  const cells = {json.dumps(cells)};
  
  const framesMu = {json.dumps(frames_mu)};
  const muMin = {mu_min}; const muMax = {mu_max};

  const framesSigma = {json.dumps(frames_sigma)};
  const sMin = {s_min}; const sMax = {s_max};
  
  const framesAcq = {json.dumps(frames_acq)};
  const aMin = {a_min}; const aMax = {a_max};

  const totalPoints = points.length;
  const gridStep = {GRID_STEP};

  const bounds = [[-90, -180], [90, 180]];
  const map = L.map('map', {{
    center: [20, 0], zoom: 2, zoomControl: false, attributionControl: false,
    maxBounds: bounds, maxBoundsViscosity: 1.0, minZoom: 2, maxZoom: 19
  }});

  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    subdomains: 'abcd', maxZoom: 19, noWrap: true
  }}).addTo(map);

  let currentIdx = -1;
  let playing = false;
  let interval = null;
  let speed = 1000;
  let markers = [];
  let bestTemp = -Infinity;
  let bestMarker = null;
  let currentLayer = 'none';
  let rects = [];

  // Inferno colormap (for Sigma and Acq)
  function infernoColor(t) {{
    const colors = [[13, 0, 68], [74, 0, 120], [181, 48, 90], [247, 160, 60], [252, 255, 164]];
    return interpolateColor(t, colors);
  }}

  // CoolWarm colormap (for Mu and point colors)
  function coolWarmColor(t) {{
    const colors = [[68, 170, 255], [0, 255, 102], [255, 204, 0], [255, 136, 0], [255, 51, 51]];
    return interpolateColor(t, colors);
  }}

  function interpolateColor(t, colors) {{
    t = Math.max(0, Math.min(1, t));
    const idx = t * (colors.length - 1);
    const i = Math.floor(idx);
    const f = idx - i;
    const c0 = colors[Math.min(i, colors.length - 1)];
    const c1 = colors[Math.min(i + 1, colors.length - 1)];
    const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
    const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
    const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }}

  function getPointColor(temp) {{
    if (temp >= 30) return '#ff3333';
    if (temp >= 20) return '#ff8800';
    if (temp >= 10) return '#ffcc00';
    if (temp >= 0)  return '#00ff66';
    return '#44aaff';
  }}

  function clearLayer() {{
    rects.forEach(r => map.removeLayer(r));
    rects = [];
  }}

  function drawLayer(frameIdx) {{
    clearLayer();
    if (currentLayer === 'none' || frameIdx < 0 || frameIdx >= totalPoints) return;

    let frame, colorFn;
    if (currentLayer === 'mu') {{ frame = framesMu[frameIdx]; colorFn = coolWarmColor; }}
    else if (currentLayer === 'sigma') {{ frame = framesSigma[frameIdx]; colorFn = infernoColor; }}
    else if (currentLayer === 'acq') {{ frame = framesAcq[frameIdx]; colorFn = infernoColor; }}

    const halfLat = gridStep / 2;
    const halfLng = gridStep / 2;

    for (let i = 0; i < cells.length; i++) {{
      const c = cells[i];
      const v = frame[i];
      const bounds = [[c.lat - halfLat, c.lng - halfLng], [c.lat + halfLat, c.lng + halfLng]];
      const rect = L.rectangle(bounds, {{
        color: 'transparent', fillColor: colorFn(v), fillOpacity: 0.45, weight: 0
      }}).addTo(map);
      rects.push(rect);
    }}
  }}

  function updateLegend() {{
    const legend = document.getElementById('legend');
    if (currentLayer === 'none') {{
      legend.classList.remove('visible');
      return;
    }}
    legend.classList.add('visible');
    
    let title, cGrad, lMin, lMax;
    if (currentLayer === 'mu') {{
      title = 'Predicted Temp (μ)';
      cGrad = 'linear-gradient(to bottom, #ff3333, #ff8800, #ffcc00, #00ff66, #44aaff)';
      lMin = muMin; lMax = muMax;
    }} else if (currentLayer === 'sigma') {{
      title = 'GP Uncertainty (σ)';
      cGrad = 'linear-gradient(to bottom, #fcffa4aa, #f7a03caa, #b5305aaa, #4a0078aa, #0d004488)';
      lMin = sMin; lMax = sMax;
    }} else {{
      title = 'Acquisition (UCB)';
      cGrad = 'linear-gradient(to bottom, #fcffa4aa, #f7a03caa, #b5305aaa, #4a0078aa, #0d004488)';
      lMin = aMin; lMax = aMax;
    }}

    document.getElementById('legend-title').textContent = title;
    document.getElementById('legend-bar').style.background = cGrad;
    
    let labelsHtml = '';
    for(let i=4; i>=0; i--) {{
      const val = lMin + (i/4) * (lMax - lMin);
      labelsHtml += '<span>' + val.toFixed(2) + '</span>';
    }}
    document.getElementById('legend-labels').innerHTML = labelsHtml;
  }}

  function changeLayer() {{
    currentLayer = document.getElementById('layer-select').value;
    updateLegend();
    if (currentIdx >= 0) drawLayer(currentIdx);
  }}

  function showPoint(idx) {{
    if (idx < 0 || idx >= totalPoints) return;
    const p = points[idx];
    currentIdx = idx;

    const color = getPointColor(p.temp);
    const marker = L.circleMarker([p.lat, p.lng], {{
      radius: 7, color: color, fillColor: color, fillOpacity: 0.85, weight: 1.5,
    }}).addTo(map);

    marker.bindPopup(
      '<div style="font-family:JetBrains Mono,monospace;font-size:12px;">' +
      '<b>' + p.temp + '°C</b><br>' + p.lat + ', ' + p.lng +
      '<br><span style="color:#888">' + p.method + '</span></div>'
    );
    markers.push(marker);

    if (p.temp > bestTemp) {{
      bestTemp = p.temp;
      if (bestMarker) map.removeLayer(bestMarker);
      bestMarker = L.circleMarker([p.lat, p.lng], {{
        radius: 14, color: '#ffcc00', fillColor: 'transparent',
        fillOpacity: 0, weight: 2.5, dashArray: '4 4',
      }}).addTo(map);
    }}

    drawLayer(idx);

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
    markers.forEach(m => map.removeLayer(m));
    markers = [];
    if (bestMarker) {{ map.removeLayer(bestMarker); bestMarker = null; }}
    clearLayer();
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
    markers.forEach(m => map.removeLayer(m));
    markers = [];
    if (bestMarker) {{ map.removeLayer(bestMarker); bestMarker = null; }}
    clearLayer();
    currentIdx = -1;
    bestTemp = -Infinity;
    for (let i = 0; i <= idx; i++) showPoint(i);
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

  function stepForward() {{ if (currentIdx < totalPoints - 1) seekTo(currentIdx + 1); }}
  function stepBack() {{ if (currentIdx > 0) seekTo(currentIdx - 1); else if (currentIdx <= 0) resetPlayback(); }}
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
