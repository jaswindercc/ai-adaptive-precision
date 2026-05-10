<div align="center">

# ⚡ Adaptive Precision Scaling

**Dynamically adjust AI inference precision (FP32 → FP16 → INT8) based on real-time hardware metrics, cost, and energy constraints.**

[![CI](https://github.com/jaswindercc/ai-adaptive-precision/actions/workflows/ci.yml/badge.svg)](https://github.com/jaswindercc/ai-adaptive-precision/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PDF](https://img.shields.io/badge/PDF-Download%20Paper-red.svg)](./Low-Cost-AI-Infrastructure-Strategies-and-Optimizationv8.pdf)

*Based on the research paper **"Low-Cost AI Infrastructure: Strategies and Optimization"** by [Jaswinder Singh](https://jaswinder.cc/)*

[Quick Start](#-quick-start) · [Live Dashboard](#-live-dashboard) · [Architecture](#-architecture) · [Examples](#-examples) · [API Reference](#-api-reference) · [Paper](#-the-paper) · [Contributing](#contributing)

</div>

---
## Why Adaptive Precision?

AI inference is expensive. Running every model at FP32 wastes compute, energy, and money — especially when FP16 or INT8 can deliver near-identical accuracy at a fraction of the cost:

| Precision | Speedup | Memory | Cost Reduction | Typical Accuracy Drop |
|-----------|---------|--------|----------------|-----------------------|
| FP32      | 1.0x    | 100%   | Baseline       | 0%                    |
| FP16      | 1.8x    | 50%    | ~60%           | 0.1% – 0.5%          |
| INT8      | 2.5x    | 25%    | ~75%           | 1.0% – 3.0%          |

**This library automatically switches precision levels** based on real-time GPU temperature, utilization, latency, cost, and energy — so you don't have to choose statically.

## Key Features

- **4 Components:** Monitoring Agent, Precision Controller, Model Wrapper, Cost Estimator
- **5 Strategies:** Rule-based, Cost-optimized, Accuracy-first, Energy-efficient, Balanced
- **Framework Support:** PyTorch (AMP + dynamic quantization), ONNX Runtime, generic wrapper
- **Real-time Monitoring:** CPU/GPU utilization, temperature, power, memory via psutil + NVML
- **Cost Tracking:** Per-inference cost, energy (kWh), carbon footprint (g CO₂)
- **Live Dashboard:** Built-in web UI with real-time monitoring, simulation, and cost analysis
- **REST API:** 9 endpoints for hardware metrics, strategy decisions, simulations, benchmarks
- **Zero Dependencies:** Core runs on just `psutil` + `numpy` — no heavy ML frameworks required
- **Lightweight:** Pure Python, pip installable, zero-config dashboard

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

With PyTorch support:
```bash
pip install -e ".[torch]"
```

With ONNX Runtime support:
```bash
pip install -e ".[onnx]"
```

### 5-Line Usage

```python
from adaptive_precision.pipeline import AutoScaler

scaler = AutoScaler(model=your_model, hardware_type="nvidia_t4")
result = scaler.predict(input_data)

print(f"Precision: {result['precision']}")       # e.g., "INT8"
print(f"Cost: ${result['cost_usd']:.6f}")         # e.g., $0.000004
print(f"CO₂: {result['carbon_grams']:.4f}g")      # e.g., 0.0001g
```

### Full Pipeline Example

```python
from adaptive_precision import MonitoringAgent, PrecisionController, CostEstimator
from adaptive_precision.controller import Strategy
from adaptive_precision.precision import PrecisionLevel

# 1. Monitor hardware
agent = MonitoringAgent()
metrics = agent.collect()
print(f"CPU: {metrics.cpu_percent}% | GPU: {metrics.gpu_utilization}%")

# 2. Controller decides precision
controller = PrecisionController(agent, strategy=Strategy.BALANCED)
decision = controller.decide()
print(f"Use {decision.selected} — {decision.reason}")

# 3. Estimate costs
estimator = CostEstimator(hardware_type="nvidia_t4")
comparison = estimator.compare_precisions(latency_fp32_ms=25.0, count=1000)
for prec, data in comparison.items():
    print(f"{prec}: ${data['cost_per_1k']:.4f}/1K inferences")
```

### Example Output

```
📊 Collecting Hardware Metrics
   CPU Usage:    9.5%
   Memory:       2.9 / 7.8 GB

🧠 Precision Controller Decision
   Strategy:  balanced
   Selected:  INT8
   Reason:    Balanced score: FP32=0.48, FP16=3.24, INT8=16.47

💰 Cost Comparison Across Precisions
   Precision  Latency      Cost/1K        CO₂/1K
   FP32       25.00ms      $0.003878      0.1944g
   FP16       13.89ms      $0.001093      0.1080g
   INT8       10.00ms      $0.000405      0.0778g
```

## 🖥 Live Dashboard

The project includes a built-in web dashboard for real-time visualization. No extra dependencies required.

```bash
# Start the dashboard server
python -m adaptive_precision.server

# Or specify a port
python -m adaptive_precision.server --port 3000
```

Then open **http://localhost:8080** in your browser.

### Dashboard Features

- **Real-time Hardware Gauges** — CPU, memory, GPU utilization, temperature (auto-refreshes every 2s)
- **Strategy Comparison** — See how all 5 strategies respond to current hardware conditions
- **Precision Level Table** — Side-by-side comparison of FP32, FP16, BF16, INT8
- **Cost Analysis** — Cloud vs edge cost comparison across NVIDIA A100/H100/T4/L4, Jetson Nano, CPU
- **Live Simulation** — Run inference simulations with configurable strategy and batch size
- **Benchmarks** — Measure latency, throughput, cost, and CO₂ per precision level

### REST API

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Server health check |
| `GET /api/hardware` | Current CPU/GPU/memory metrics |
| `GET /api/hardware/history` | Last 50 hardware snapshots |
| `GET /api/strategies` | All 5 strategy decisions for current conditions |
| `GET /api/precisions` | Precision level properties |
| `GET /api/cost` | Cost comparison across 6 hardware types |
| `GET /api/simulate?count=N&strategy=X` | Run N inferences with given strategy |
| `GET /api/benchmark` | Benchmark all precision levels |
| `GET /api/status` | Full system status |

## 🏗 Architecture

The system implements the **Adaptive Precision Scaling System** from the paper, consisting of four components in a feedback loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    Adaptive Precision Scaling                │
│                                                             │
│  ┌──────────────┐    ┌───────────────────┐                  │
│  │  Monitoring   │───▶│    Precision      │                  │
│  │  Agent        │    │    Controller     │                  │
│  │              │    │                   │                  │
│  │ • CPU/GPU %  │    │ • Rule-based      │                  │
│  │ • Temperature│    │ • Cost-optimized  │                  │
│  │ • Power (W)  │    │ • Accuracy-first  │                  │
│  │ • Latency    │    │ • Energy-efficient│                  │
│  │ • Memory     │    │ • Balanced        │                  │
│  └──────────────┘    └────────┬──────────┘                  │
│         ▲                     │                             │
│         │                     ▼                             │
│  ┌──────┴───────┐    ┌───────────────────┐                  │
│  │  Cost         │    │   Model Wrapper    │                  │
│  │  Estimator    │◀───│                   │                  │
│  │              │    │ • PyTorch AMP     │                  │
│  │ • $/inference│    │ • ONNX Runtime    │                  │
│  │ • kWh/infer  │    │ • Dynamic quant   │                  │
│  │ • g CO₂      │    │ • Zero-downtime   │                  │
│  └──────────────┘    └───────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | Purpose | Key Features |
|-----------|---------|-------------|
| **MonitoringAgent** | Real-time hardware telemetry | CPU/GPU util, temp, power, memory; rolling history buffer; callbacks |
| **PrecisionController** | Decision engine | 5 strategies; composite scoring; accuracy guard; threshold config |
| **ModelWrapper** | Runtime precision switching | PyTorch AMP, ONNX Runtime, dynamic quantization; zero-downtime |
| **CostEstimator** | Per-inference cost tracking | Cloud pricing (A100/H100/T4/L4), energy, carbon footprint |

### Decision Formula

The balanced controller uses the composite performance score from the paper:

$$S = \frac{T_{put}}{C_{inf} + E_{per} + L_{avg}}$$

Where:
- $T_{put}$ = Throughput (inferences/second)
- $C_{inf}$ = Cost per inference (USD)
- $E_{per}$ = Energy per inference (kWh)
- $L_{avg}$ = Average latency (ms)

## 📖 Examples

Run any example directly:

```bash
# Full quickstart demo
python examples/quickstart.py

# Cost analysis across cloud vs edge hardware
python examples/cost_analysis.py

# Compare all 5 controller strategies
python examples/strategies.py
```

## 📚 API Reference

### `MonitoringAgent`

```python
agent = MonitoringAgent(history_size=500)
metrics = agent.collect()                    # HardwareMetrics snapshot
agent.record_inference(inference_metrics)     # Track inference stats
agent.avg_cpu_percent(last_n=10)             # Rolling averages
agent.avg_latency_ms(last_n=10)
agent.summary()                              # Full status dict
```

### `PrecisionController`

```python
from adaptive_precision.controller import Strategy, PrecisionThresholds

controller = PrecisionController(
    monitor=agent,
    strategy=Strategy.BALANCED,
    thresholds=PrecisionThresholds(max_gpu_temp_c=75.0),
    allowed_levels=[PrecisionLevel.FP32, PrecisionLevel.FP16, PrecisionLevel.INT8],
)

decision = controller.decide()  # PrecisionDecision
# decision.selected  → PrecisionLevel.INT8
# decision.reason    → "Balanced score: FP32=0.48, INT8=16.47"
# decision.score     → 16.47
```

### `ModelWrapper`

```python
# Generic wrapper (testing/prototyping)
wrapper = ModelWrapper(model=my_function)

# PyTorch wrapper (production)
from adaptive_precision.wrapper import PyTorchWrapper
wrapper = PyTorchWrapper(model=torch_model, device="cuda")
wrapper.set_precision(PrecisionLevel.FP16)
output, metrics = wrapper.timed_predict(input_tensor)

# ONNX wrapper
from adaptive_precision.wrapper import ONNXWrapper
wrapper = ONNXWrapper("model.onnx")
```

### `CostEstimator`

```python
estimator = CostEstimator(
    hardware_type="nvidia_t4",      # or nvidia_a100, jetson_nano, etc.
    energy_price_kwh=0.12,          # electricity rate
    carbon_intensity_g_kwh=400.0,   # grid carbon intensity
)

cost = estimator.estimate(latency_ms=15.0, precision=PrecisionLevel.INT8)
# cost.total_cost_usd, cost.energy_kwh, cost.carbon_grams

comparison = estimator.compare_precisions(latency_fp32_ms=25.0, count=1000)
```

### `AutoScaler` (Pipeline)

```python
from adaptive_precision.pipeline import AutoScaler

scaler = AutoScaler(
    model=your_model,
    strategy=Strategy.BALANCED,
    hardware_type="nvidia_t4",
    auto_adjust=True,          # Auto re-evaluate precision
    adjust_interval=5,         # Every N inferences
)

result = scaler.predict(input_data)
benchmark = scaler.benchmark(input_data, iterations=100)
status = scaler.status()
```

## Supported Hardware Profiles

| Profile | Typical Device | Cost/hr | TDP |
|---------|---------------|---------|-----|
| `nvidia_a100` | A100 40/80GB | $35 | 400W |
| `nvidia_h100` | H100 80GB | $45 | 700W |
| `nvidia_v100` | V100 16/32GB | $20 | 300W |
| `nvidia_t4` | T4 16GB | $0.55 | 70W |
| `nvidia_l4` | L4 24GB | $0.75 | 72W |
| `jetson_nano` | Jetson Nano | $0.01* | 10W |
| `cpu_generic` | Any CPU | $0.10 | 65W |

*\*Amortized hardware cost*

## 📄 The Paper

This project implements the **Adaptive Precision Scaling System** described in Section 7 of:

> **"Low-Cost AI Infrastructure: Strategies and Optimization"**
> By Jaswinder Singh · [jaswinder.cc](https://jaswinder.cc/)

The paper covers:
- Cost-efficient AI infrastructure strategies
- Edge AI deployment on Jetson Nano, Coral TPU, Raspberry Pi
- Model optimization: pruning, quantization, knowledge distillation
- GPU/TPU/ASIC hardware comparison and trade-offs
- The Adaptive Precision Scaling System (implemented here)
- Performance benchmarks and metrics

Read the full paper:
- [Download PDF](./Low-Cost-AI-Infrastructure-Strategies-and-Optimizationv8.pdf) — Original paper (PDF)

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all 47 tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=adaptive_precision --cov-report=term-missing

# Lint
ruff check src/ tests/
```

## Project Structure

```
adaptive-precision/
├── src/adaptive_precision/
│   ├── __init__.py          # Public API
│   ├── __main__.py          # python -m adaptive_precision.server
│   ├── precision.py         # PrecisionLevel enum
│   ├── monitor.py           # MonitoringAgent (hardware telemetry)
│   ├── controller.py        # PrecisionController (decision engine)
│   ├── wrapper.py           # ModelWrapper (PyTorch, ONNX, generic)
│   ├── cost_estimator.py    # CostEstimator ($/inference, kWh, CO₂)
│   ├── pipeline.py          # AutoScaler (end-to-end pipeline)
│   ├── server.py            # Web server + REST API
│   └── dashboard.html       # Live web dashboard UI
├── tests/                   # 47 tests
├── examples/
│   ├── quickstart.py        # Full demo
│   ├── cost_analysis.py     # Cloud vs edge cost comparison
│   └── strategies.py        # Controller strategy comparison
├── *.pdf                    # Original research paper (PDF)
├── pyproject.toml           # Package config
└── .github/workflows/ci.yml # CI pipeline
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas of interest:
- Real-world benchmarks with production models
- AMD/Intel GPU monitoring support
- Reinforcement learning-based controller
- Kubernetes operator for cluster-wide scaling
- Persistent metrics storage (InfluxDB/TimescaleDB)

## License

MIT License — see [LICENSE](LICENSE).

## Author

**Jaswinder Singh** — [jaswinder.cc](https://jaswinder.cc/)

---

<div align="center">

*If this project helps you save on AI infrastructure costs, consider giving it a ⭐*

</div>
