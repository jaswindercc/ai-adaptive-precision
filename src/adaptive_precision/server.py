"""
Web Dashboard Server for Adaptive Precision Scaling System.

Provides a real-time web UI and REST API to visualize hardware metrics,
precision decisions, cost analysis, and run live inference simulations.

Usage:
    python -m adaptive_precision.server          # starts on port 8080
    python -m adaptive_precision.server --port 3000

Author: Jaswinder Singh (https://jaswinder.cc/)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from adaptive_precision.controller import (
    PrecisionController,
    Strategy,
)
from adaptive_precision.cost_estimator import CostEstimator
from adaptive_precision.monitor import (
    MonitoringAgent,
)
from adaptive_precision.pipeline import AutoScaler
from adaptive_precision.precision import PrecisionLevel

logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────

monitor = MonitoringAgent(history_size=200)
scaler = AutoScaler(
    model=lambda x: [v * 2 for v in x] if isinstance(x, list) else x,
    strategy=Strategy.BALANCED,
    hardware_type="nvidia_t4",
    auto_adjust=True,
    adjust_interval=3,
)

simulation_running = False
simulation_log: List[Dict] = []
simulation_lock = threading.Lock()


# ── API Handlers ──────────────────────────────────────────────

def api_health() -> Dict:
    return {"status": "ok", "version": "0.1.0"}


def api_hardware() -> Dict:
    m = monitor.collect()
    return m.to_dict()


def api_hardware_history() -> List[Dict]:
    return [m.to_dict() for m in monitor.hardware_history[-50:]]


def api_strategies() -> Dict:
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


def api_precisions() -> List[Dict]:
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


def api_cost_compare() -> Dict:
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
        comp = est.compare_precisions(
            latency_fp32_ms=25.0, count=1000,
        )
        results[hw_type] = {"name": hw_name, "precisions": comp}
    return results


def api_simulate(params: Dict) -> Dict:
    count = min(int(params.get("count", [100])[0]), 5000)
    strategy_name = params.get("strategy", ["balanced"])[0]

    try:
        strategy = Strategy(strategy_name)
    except ValueError:
        strategy = Strategy.BALANCED

    scaler.controller.strategy = strategy

    sim_results: List[Dict] = []
    for i in range(count):
        inp = [random.uniform(0.1, 10.0) for _ in range(5)]
        result = scaler.predict(inp)
        sim_results.append({
            "index": i,
            "precision": result["precision"],
            "latency_ms": result["latency_ms"],
            "cost_usd": result["cost_usd"],
            "carbon_grams": result["carbon_grams"],
        })

    status = scaler.status()

    # Aggregate stats by precision
    prec_stats: Dict[str, Dict] = {}
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
        "count": count,
        "strategy": strategy.value,
        "status": status,
        "precision_stats": prec_stats,
        "samples": sim_results[:50],
    }


def api_status() -> Dict:
    return scaler.status()


def api_benchmark() -> Dict:
    return scaler.benchmark([1.0, 2.0, 3.0, 4.0, 5.0], iterations=50)


# ── Route dispatch ────────────────────────────────────────────

ROUTES = {
    "/api/health": ("GET", api_health),
    "/api/hardware": ("GET", api_hardware),
    "/api/hardware/history": ("GET", api_hardware_history),
    "/api/strategies": ("GET", api_strategies),
    "/api/precisions": ("GET", api_precisions),
    "/api/cost": ("GET", api_cost_compare),
    "/api/status": ("GET", api_status),
    "/api/benchmark": ("GET", api_benchmark),
}


# ── HTTP Handler ──────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Silence default logging

    def _json_response(self, data: Any, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # API routes
        if path in ROUTES:
            method, handler = ROUTES[path]
            try:
                self._json_response(handler())
            except Exception as e:
                self._json_response({"error": str(e)}, 500)
            return

        if path == "/api/simulate":
            try:
                self._json_response(api_simulate(params))
            except Exception as e:
                self._json_response({"error": str(e)}, 500)
            return

        # Serve dashboard
        if path == "/" or path == "/index.html":
            self._serve_dashboard()
            return

        self.send_error(404)

    def _serve_dashboard(self):
        html_path = Path(__file__).parent / "dashboard.html"
        if not html_path.exists():
            self.send_error(500, "Dashboard file not found")
            return
        content = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


# ── Main ──────────────────────────────────────────────────────

def run_server(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"\n{'=' * 60}")
    print("  Adaptive Precision Scaling — Live Dashboard")
    print("  Paper: Low-Cost AI Infrastructure by Jaswinder Singh")
    print(f"{'=' * 60}")
    print(f"\n  Dashboard:  http://localhost:{port}")
    print(f"  API:        http://localhost:{port}/api/health")
    print("\n  API Endpoints:")
    print("    GET /api/health          — Server health check")
    print("    GET /api/hardware        — Current hardware metrics")
    print("    GET /api/hardware/history — Hardware metrics history")
    print("    GET /api/strategies      — All strategy decisions")
    print("    GET /api/precisions      — Precision level details")
    print("    GET /api/cost            — Cost comparison")
    print("    GET /api/simulate?count=N&strategy=X — Run simulation")
    print("    GET /api/benchmark       — Benchmark all precisions")
    print("    GET /api/status          — System status")
    print("\n  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Precision Scaling Dashboard",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port to run the server on (default: 8080)",
    )
    args = parser.parse_args()
    run_server(args.port)


if __name__ == "__main__":
    main()
