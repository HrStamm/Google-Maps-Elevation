import pandas as pd
import os
import webbrowser
import json

# ==============================================================================
# Peak Hunter Dashboard Generator
# Generates a static HTML dashboard matching the "Peak Hunter: Optimized View"
# aesthetic with a dark theme and normalized coordinate space.
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(PROJECT_ROOT, "src", "data", "results.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "reports", "peak_hunter.html")

TOTAL_BUDGET = 20  # Total number of allowed API calls


def normalize_coords(lat, lng):
    """Normalize GPS coordinates to [0, 1] range."""
    x = (lng + 180) / 360
    y = (lat + 90) / 180
    return round(x, 3), round(y, 3)


def generate_dashboard():
    print(f"Loading data from {DATA_FILE}...")

    if not os.path.exists(DATA_FILE):
        print("Error: results.csv not found.")
        return None

    df = pd.read_csv(DATA_FILE)

    if len(df) == 0:
        print("Error: No data in results.csv.")
        return None

    # Normalize coordinates and find the best result
    points = []
    best_temp = -999
    best_idx = 0

    for i, row in df.iterrows():
        x, y = normalize_coords(row["lat"], row["lng"])
        temp = row["temp"]
        points.append({
            "x": x,
            "y": y,
            "lat": round(row["lat"], 4),
            "lng": round(row["lng"], 4),
            "temp": temp,
            "method": row["search_method"],
        })
        if temp > best_temp:
            best_temp = temp
            best_idx = i

    clicks_remaining = max(0, TOTAL_BUDGET - len(df))
    points_json = json.dumps(points)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Peak Hunter: Optimized View</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: #0d0d0d;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 30px 40px;
  }}

  h1 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    letter-spacing: 3px;
    margin-bottom: 30px;
    text-transform: uppercase;
  }}
  h1 span {{
    color: #00ff66;
  }}

  .dashboard {{
    display: flex;
    gap: 30px;
    width: 100%;
    max-width: 1100px;
  }}

  /* ---- Plot Area ---- */
  .plot-area {{
    flex: 1;
    aspect-ratio: 1;
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    position: relative;
    overflow: hidden;
    min-height: 450px;
  }}

  .plot-area .grid-line-h,
  .plot-area .grid-line-v {{
    position: absolute;
    background: #1e1e1e;
  }}
  .plot-area .grid-line-h {{
    width: 100%;
    height: 1px;
  }}
  .plot-area .grid-line-v {{
    height: 100%;
    width: 1px;
  }}

  .point {{
    position: absolute;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: transform 0.2s;
    cursor: pointer;
  }}
  .point:hover {{
    transform: translate(-50%, -50%) scale(1.6);
    z-index: 10;
  }}
  .point.normal {{
    background: #4488ff;
    box-shadow: 0 0 6px #4488ff88;
  }}
  .point.best {{
    background: #ffcc00;
    border: 2px solid #ffcc00;
    box-shadow: 0 0 12px #ffcc0088, 0 0 24px #ffcc0044;
    width: 16px;
    height: 16px;
  }}

  .point .tooltip {{
    display: none;
    position: absolute;
    bottom: 18px;
    left: 50%;
    transform: translateX(-50%);
    background: #222;
    border: 1px solid #444;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 11px;
    white-space: nowrap;
    color: #ccc;
    font-family: 'JetBrains Mono', monospace;
    z-index: 20;
  }}
  .point:hover .tooltip {{
    display: block;
  }}

  /* ---- Right Panel ---- */
  .panel {{
    width: 340px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }}

  .card {{
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 20px;
  }}

  .card h3 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #888;
    margin-bottom: 12px;
  }}

  /* Sliders */
  .slider-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }}
  .slider-row label {{
    font-size: 13px;
    color: #aaa;
  }}
  .slider-row .val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #fff;
    min-width: 45px;
    text-align: right;
  }}
  input[type="range"] {{
    -webkit-appearance: none;
    width: 100%;
    height: 4px;
    background: #333;
    border-radius: 2px;
    outline: none;
    margin: 4px 0 12px 0;
  }}
  input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #4488ff;
    cursor: pointer;
    box-shadow: 0 0 6px #4488ff88;
  }}

  .badge {{
    display: inline-block;
    background: #1a3a1a;
    color: #00ff66;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 4px;
    margin-top: 6px;
  }}

  /* Clicks Remaining */
  .clicks-card {{
    border-left: 3px solid #ffcc00;
  }}
  .clicks-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 42px;
    font-weight: 700;
    color: #00ff66;
    line-height: 1;
  }}
  .clicks-sub {{
    font-size: 13px;
    color: #666;
    margin-top: 4px;
  }}

  /* Table */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }}
  th {{
    text-align: left;
    padding: 6px 8px;
    border-bottom: 1px solid #333;
    color: #888;
    font-weight: 600;
  }}
  td {{
    padding: 6px 8px;
    border-bottom: 1px solid #1e1e1e;
  }}
  tr.best-row {{
    background: #2a2a00;
    color: #ffcc00;
    font-weight: 700;
  }}
  tr.best-row td {{
    color: #ffcc00;
  }}
</style>
</head>
<body>

<h1>Peak Hunter: <span>Optimized View</span></h1>

<div class="dashboard">
  <!-- Left: Plot -->
  <div class="plot-area" id="plot">
    <!-- Grid lines injected by JS -->
  </div>

  <!-- Right: Controls -->
  <div class="panel">

    <div class="card">
      <h3>Visualization Layer</h3>
      <select style="width:100%;padding:8px;background:#222;color:#fff;border:1px solid #444;border-radius:4px;font-family:'JetBrains Mono',monospace;">
        <option>Nothing</option>
        <option>GP Mean</option>
        <option>GP Uncertainty</option>
        <option>Acquisition Function</option>
      </select>

      <div style="margin-top: 18px;">
        <div class="slider-row">
          <label>Kernel Lengthscale:</label>
          <span class="val" id="ls-val">0.12</span>
        </div>
        <input type="range" min="0.01" max="1" step="0.01" value="0.12"
               oninput="document.getElementById('ls-val').textContent=this.value">

        <div class="slider-row">
          <label>Epsilon (ε):</label>
          <span class="val" id="eps-val">0.010</span>
        </div>
        <input type="range" min="0.001" max="0.5" step="0.001" value="0.01"
               oninput="document.getElementById('eps-val').textContent=parseFloat(this.value).toFixed(3)">

        <div class="slider-row">
          <label>Delta (δ):</label>
          <span class="val" id="delta-val">0.010</span>
        </div>
        <input type="range" min="0.001" max="0.5" step="0.001" value="0.01"
               oninput="document.getElementById('delta-val').textContent=parseFloat(this.value).toFixed(3)">

        <div class="slider-row">
          <label>Calculated Beta (√β_t):</label>
          <span class="val" style="color:#00ff66;">28.63</span>
        </div>

        <div class="badge">Normalized Acq Heatmap Enabled</div>
      </div>
    </div>

    <div class="card clicks-card">
      <h3>Clicks Remaining</h3>
      <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <div class="clicks-value">{best_temp}</div>
        <span class="clicks-sub">{len(df)} / {TOTAL_BUDGET}</span>
      </div>
      <div class="clicks-sub" style="margin-top:2px;">Best temperature found</div>
    </div>

    <div class="card">
      <h3>Sample Log</h3>
      <table>
        <thead>
          <tr><th>#</th><th>X</th><th>Y</th><th>Value</th></tr>
        </thead>
        <tbody id="table-body">
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
  const points = {points_json};
  const bestIdx = {best_idx};
  const plot = document.getElementById('plot');
  const tbody = document.getElementById('table-body');

  // Draw grid lines
  for (let i = 1; i < 5; i++) {{
    const h = document.createElement('div');
    h.className = 'grid-line-h';
    h.style.top = (i * 20) + '%';
    plot.appendChild(h);

    const v = document.createElement('div');
    v.className = 'grid-line-v';
    v.style.left = (i * 20) + '%';
    plot.appendChild(v);
  }}

  // Draw points and populate table
  points.forEach((p, i) => {{
    // Plot point
    const dot = document.createElement('div');
    dot.className = 'point ' + (i === bestIdx ? 'best' : 'normal');
    dot.style.left = (p.x * 100) + '%';
    dot.style.bottom = (p.y * 100) + '%';

    const tip = document.createElement('div');
    tip.className = 'tooltip';
    tip.textContent = p.temp + '°C  (' + p.lat + ', ' + p.lng + ')';
    dot.appendChild(tip);

    plot.appendChild(dot);

    // Table row
    const tr = document.createElement('tr');
    if (i === bestIdx) tr.className = 'best-row';
    const method = p.method.charAt(0).toUpperCase();
    tr.innerHTML = '<td>' + method + '</td><td>' + p.x + '</td><td>' + p.y + '</td><td>' + p.temp + '</td>';
    tbody.appendChild(tr);
  }});
</script>

</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write(html)

    print(f"Dashboard saved to: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    output = generate_dashboard()
    if output:
        webbrowser.open("file://" + os.path.realpath(output))
