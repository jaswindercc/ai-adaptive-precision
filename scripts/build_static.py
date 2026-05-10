"""
Build static site for GitHub Pages deployment.

Generates static JSON data files and a self-contained dashboard
that works without the Python server.

Usage:
    python scripts/build_static.py
    # Output goes to _site/
"""

import json
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_precision.controller import PrecisionController, Strategy
from adaptive_precision.cost_estimator import CostEstimator
from adaptive_precision.monitor import MonitoringAgent
from adaptive_precision.pipeline import AutoScaler
from adaptive_precision.precision import PrecisionLevel

OUT = Path(__file__).parent.parent / "_site"
DATA = OUT / "api"


def build_precisions():
    return [
        {
            "name": str(level),
            "value": level.value,
            "bits": level.bits,
            "speedup": level.speedup_factor,
            "memory_ratio": level.memory_ratio,
            "accuracy_drop": level.typical_accuracy_drop,
        }
        for level in PrecisionLevel
    ]


def build_strategies():
    monitor = MonitoringAgent()
    hw = monitor.collect()
    results = {}
    for strategy in Strategy:
        ctrl = PrecisionController(monitor, strategy=strategy)
        decision = ctrl.decide(hw)
        results[strategy.value] = {
            "selected": str(decision.selected),
            "reason": decision.reason,
        }
    return {"hardware": hw.to_dict(), "strategies": results}


def build_cost():
    results = {}
    hw_configs = [
        ("nvidia_a100", "NVIDIA A100"),
        ("nvidia_h100", "NVIDIA H100"),
        ("nvidia_t4", "NVIDIA T4"),
        ("nvidia_l4", "NVIDIA L4"),
        ("jetson_nano", "Jetson Nano"),
        ("cpu_generic", "Generic CPU"),
    ]
    for hw_type, hw_name in hw_configs:
        est = CostEstimator(hardware_type=hw_type)
        comp = est.compare_precisions(latency_fp32_ms=25.0, count=1000)
        results[hw_type] = {"name": hw_name, "precisions": comp}
    return results


def build_simulation():
    import random

    random.seed(42)
    scaler = AutoScaler(
        model=lambda x: [v * 2 for v in x],
        strategy=Strategy.BALANCED,
        hardware_type="nvidia_t4",
        auto_adjust=True,
        adjust_interval=3,
    )
    sim_results = []
    for i in range(200):
        inp = [random.uniform(0.1, 10.0) for _ in range(5)]
        result = scaler.predict(inp)
        sim_results.append({
            "index": i,
            "precision": result["precision"],
            "latency_ms": result["latency_ms"],
            "cost_usd": result["cost_usd"],
            "carbon_grams": result["carbon_grams"],
        })

    prec_stats = {}
    for r in sim_results:
        p = r["precision"]
        if p not in prec_stats:
            prec_stats[p] = {
                "count": 0,
                "total_latency": 0,
                "total_cost": 0,
                "total_carbon": 0,
            }
        prec_stats[p]["count"] += 1
        prec_stats[p]["total_latency"] += r["latency_ms"]
        prec_stats[p]["total_cost"] += r["cost_usd"]
        prec_stats[p]["total_carbon"] += r["carbon_grams"]

    for p, s in prec_stats.items():
        s["avg_latency"] = round(s["total_latency"] / s["count"], 4)
        s["avg_cost"] = s["total_cost"] / s["count"]

    return {
        "count": 200,
        "strategy": "balanced",
        "status": scaler.status(),
        "precision_stats": prec_stats,
        "samples": sim_results[:50],
    }


def build_benchmark():
    scaler = AutoScaler(
        model=lambda x: [v * 2 for v in x],
        hardware_type="nvidia_t4",
    )
    return scaler.benchmark([1.0, 2.0, 3.0, 4.0, 5.0], iterations=50)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str, indent=2))
    print(f"  {path.relative_to(OUT)}")


def main():
    print(f"Building static site → {OUT}/\n")

    # Clean output
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    DATA.mkdir(parents=True)

    # Generate JSON data
    print("Generating API data:")
    write_json(DATA / "health.json", {"status": "ok", "version": "0.1.0", "mode": "static"})
    write_json(DATA / "precisions.json", build_precisions())
    write_json(DATA / "strategies.json", build_strategies())
    write_json(DATA / "cost.json", build_cost())
    write_json(DATA / "simulation.json", build_simulation())
    write_json(DATA / "benchmark.json", build_benchmark())

    # Copy dashboard HTML
    src_dir = Path(__file__).parent.parent / "src" / "adaptive_precision"
    dashboard_src = src_dir / "dashboard.html"
    html = dashboard_src.read_text()

    # Inject static mode flag and data loader
    static_script = """
<script>
// Static mode — data pre-generated at build time, no live server
const STATIC_MODE = true;
const STATIC_DATA = {};

async function loadStaticData() {
  const files = ['health', 'precisions', 'strategies', 'cost', 'simulation', 'benchmark'];
  for (const f of files) {
    try {
      const r = await fetch(`api/${f}.json`);
      STATIC_DATA[f] = await r.json();
    } catch(e) { console.warn(`Failed to load ${f}:`, e); }
  }
}
</script>
"""
    html = html.replace("</head>", static_script + "</head>")

    # Replace the existing <script> block with static-aware version
    old_script_start = "<script>\nconst API = '';"
    new_script_start = """<script>
const API = '';
const IS_STATIC = typeof STATIC_MODE !== 'undefined' && STATIC_MODE;"""
    html = html.replace(old_script_start, new_script_start)

    # Add static banner injection into init
    old_init = """async function init() {
  pollHardware();
  hwInterval = setInterval(pollHardware, 2000);
  await Promise.all([loadStrategies(), loadPrecisions(), loadCost()]);
  setInterval(loadStrategies, 5000);
}"""
    new_init = """async function init() {
  if (IS_STATIC) {
    await loadStaticData();
    // Show static mode banner
    const banner = document.createElement('div');
    banner.style.cssText = 'background:#1c2333;border:1px solid #30363d;border-radius:8px;padding:12px 20px;margin-bottom:16px;font-size:13px;color:#8b949e;text-align:center';
    banner.innerHTML = '📄 <strong style="color:#e6edf3">Static Preview</strong> — You are viewing pre-generated data. ' +
      '<a href="https://github.com/jaswindercc/ai-adaptive-precision#-live-dashboard" style="color:#58a6ff">Run the server locally</a> for live hardware monitoring and interactive simulations.';
    document.querySelector('.container').prepend(banner);
  }
  pollHardware();
  if (!IS_STATIC) hwInterval = setInterval(pollHardware, 2000);
  await Promise.all([loadStrategies(), loadPrecisions(), loadCost()]);
  if (!IS_STATIC) setInterval(loadStrategies, 5000);
  if (IS_STATIC) {
    // Auto-load simulation and benchmark data
    loadStaticSimulation();
    loadStaticBenchmark();
  }
}

function loadStaticSimulation() {
  const d = STATIC_DATA.simulation;
  if (!d) return;
  const barsEl = document.getElementById('sim-bars');
  const total = d.count;
  let html = '';
  for (const [prec, stats] of Object.entries(d.precision_stats)) {
    const pct = (stats.count / total * 100).toFixed(1);
    html += '<div class="bar-group"><div class="bar-label">' +
      '<span>' + prec + ' — ' + stats.count + ' inferences (' + pct + '%)</span>' +
      '<span style="color:var(--muted)">avg ' + stats.avg_latency.toFixed(3) + 'ms</span></div>' +
      '<div class="bar-track"><div class="bar-fill ' + barClass(prec) + '" style="width:' + pct + '%">' + pct + '%</div></div></div>';
  }
  barsEl.innerHTML = html;
  const s = d.status;
  document.getElementById('sim-stats').innerHTML =
    '<table><tr><td style="color:var(--muted)">Total Inferences</td><td style="font-weight:600">' + s.total_inferences + '</td></tr>' +
    '<tr><td style="color:var(--muted)">Strategy</td><td style="font-weight:600">' + s.strategy + '</td></tr></table>';
  const logEl = document.getElementById('sim-log');
  logEl.innerHTML = d.samples.map(function(s) {
    return '<div><span style="color:var(--muted)">#' + String(s.index).padStart(4,'0') + '</span> ' +
      badgeFor(s.precision) + ' <span style="color:var(--muted)">' + s.latency_ms.toFixed(3) + 'ms</span> ' +
      '<span style="color:var(--green)">$' + s.cost_usd.toFixed(8) + '</span></div>';
  }).join('');
}

function loadStaticBenchmark() {
  const data = STATIC_DATA.benchmark;
  if (!data) return;
  const el = document.getElementById('bench-results');
  let html = '<table><thead><tr><th>Precision</th><th>Avg Latency</th><th>Min</th><th>Max</th>' +
    '<th>Throughput</th><th>Cost/1K</th><th>Energy/1K</th><th>CO₂/1K</th></tr></thead><tbody>';
  for (const [prec, d] of Object.entries(data)) {
    html += '<tr><td>' + badgeFor(prec) + '</td>' +
      '<td>' + d.avg_latency_ms.toFixed(3) + ' ms</td>' +
      '<td>' + d.min_latency_ms.toFixed(3) + ' ms</td>' +
      '<td>' + d.max_latency_ms.toFixed(3) + ' ms</td>' +
      '<td style="font-weight:600">' + d.throughput_per_sec.toLocaleString() + ' /s</td>' +
      '<td style="color:var(--green)">$' + d.cost_per_1k_usd.toFixed(6) + '</td>' +
      '<td>' + d.energy_kwh_per_1k.toFixed(8) + '</td>' +
      '<td>' + d.carbon_g_per_1k.toFixed(6) + ' g</td></tr>';
  }
  html += '</tbody></table>';
  el.innerHTML = html;
  document.getElementById('bench-note').style.display = 'block';
}"""
    html = html.replace(old_init, new_init)

    # Make API calls fall back to static data
    old_poll = """async function pollHardware() {
  try {
    const r = await fetch(API + '/api/hardware');
    const d = await r.json();"""
    new_poll = """async function pollHardware() {
  try {
    let d;
    if (IS_STATIC) {
      d = STATIC_DATA.strategies ? STATIC_DATA.strategies.hardware : {cpu_percent:0,memory_percent:0};
    } else {
      const r = await fetch(API + '/api/hardware');
      d = await r.json();
    }"""
    html = html.replace(old_poll, new_poll)

    old_strat_fetch = """    const r = await fetch(API + '/api/strategies');
    const d = await r.json();"""
    new_strat_fetch = """    let d;
    if (IS_STATIC) { d = STATIC_DATA.strategies; }
    else { const r = await fetch(API + '/api/strategies'); d = await r.json(); }"""
    html = html.replace(old_strat_fetch, new_strat_fetch)

    old_prec_fetch = """    const r = await fetch(API + '/api/precisions');
    const data = await r.json();"""
    new_prec_fetch = """    let data;
    if (IS_STATIC) { data = STATIC_DATA.precisions; }
    else { const r = await fetch(API + '/api/precisions'); data = await r.json(); }"""
    html = html.replace(old_prec_fetch, new_prec_fetch)

    old_cost_fetch = """    const r = await fetch(API + '/api/cost');
    const data = await r.json();"""
    new_cost_fetch = """    let data;
    if (IS_STATIC) { data = STATIC_DATA.cost; }
    else { const r = await fetch(API + '/api/cost'); data = await r.json(); }"""
    html = html.replace(old_cost_fetch, new_cost_fetch)

    # Write final HTML
    (OUT / "index.html").write_text(html)
    print(f"  index.html")

    # Copy paper PDF if it exists
    pdf = Path(__file__).parent.parent / "Low-Cost-AI-Infrastructure-Strategies-and-Optimizationv8.pdf"
    if pdf.exists():
        shutil.copy2(pdf, OUT / pdf.name)
        print(f"  {pdf.name}")

    print(f"\nDone! Static site built to {OUT}/")
    print(f"Test locally: cd _site && python -m http.server 9000")


if __name__ == "__main__":
    main()
