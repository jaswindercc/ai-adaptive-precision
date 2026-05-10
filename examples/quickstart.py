"""
Quick Start Example — Adaptive Precision Scaling System

Demonstrates the full pipeline: monitoring, precision control,
cost estimation, and automatic scaling.

Author: Jaswinder Singh (https://jaswinder.cc/)
"""

from adaptive_precision import MonitoringAgent, PrecisionController, CostEstimator, PrecisionLevel
from adaptive_precision.controller import Strategy
from adaptive_precision.pipeline import AutoScaler


def main():
    print("=" * 60)
    print("  Adaptive Precision Scaling System — Quick Start")
    print("  Paper: Low-Cost AI Infrastructure by Jaswinder Singh")
    print("=" * 60)

    # --- 1. Hardware Monitoring ---
    print("\n📊 Step 1: Collecting Hardware Metrics")
    agent = MonitoringAgent()
    metrics = agent.collect()
    print(f"   CPU Usage:    {metrics.cpu_percent}%")
    print(f"   Memory:       {metrics.memory_used_gb:.1f} / {metrics.memory_total_gb:.1f} GB")
    print(f"   GPU Available: {metrics.is_gpu_available}")

    # --- 2. Precision Controller ---
    print("\n🧠 Step 2: Precision Controller Decision")
    controller = PrecisionController(agent, strategy=Strategy.BALANCED)
    decision = controller.decide()
    print(f"   Strategy:  {decision.strategy.value}")
    print(f"   Selected:  {decision.selected}")
    print(f"   Reason:    {decision.reason}")

    # --- 3. Cost Estimation ---
    print("\n💰 Step 3: Cost Comparison Across Precisions")
    estimator = CostEstimator(hardware_type="nvidia_t4")
    comparison = estimator.compare_precisions(latency_fp32_ms=25.0, count=1000)

    print(f"   {'Precision':<10} {'Latency':<12} {'Cost/1K':<14} {'CO₂/1K'}")
    print(f"   {'─' * 48}")
    for prec, data in comparison.items():
        print(
            f"   {prec:<10} {data['latency_ms']:<12.2f} "
            f"${data['cost_per_1k']:<13.6f} {data['carbon_g_per_1k']:.4f}g"
        )

    # --- 4. AutoScaler Pipeline ---
    print("\n🚀 Step 4: AutoScaler Pipeline")
    scaler = AutoScaler(
        model=lambda x: [v * 2 for v in x],  # Simple demo model
        strategy=Strategy.BALANCED,
        hardware_type="nvidia_t4",
    )

    # Log precision switches
    scaler.on_precision_switch(
        lambda d: print(f"   ⚡ Precision switched: {d.previous} → {d.selected}")
    )

    # Run 20 inferences
    for i in range(20):
        result = scaler.predict([1.0, 2.0, 3.0])

    status = scaler.status()
    print(f"   Total inferences: {status['total_inferences']}")
    print(f"   Current precision: {status['precision']}")
    print(f"   Total cost: ${status['total_cost_usd']:.6f}")
    print(f"   Total CO₂: {status['total_carbon_grams']:.4f}g")

    # --- 5. Precision Properties ---
    print("\n📐 Step 5: Precision Level Properties")
    print(f"   {'Level':<8} {'Bits':<6} {'Speedup':<10} {'Memory':<10} {'Accuracy Drop'}")
    print(f"   {'─' * 52}")
    for level in PrecisionLevel:
        drop = level.typical_accuracy_drop
        print(
            f"   {str(level):<8} {level.bits:<6} {level.speedup_factor:<10.1f} "
            f"{level.memory_ratio:<10.2f} {drop['min']*100:.1f}%–{drop['max']*100:.1f}%"
        )

    print("\n✅ Done! Visit https://jaswinder.cc/ for the full paper.")
    print("=" * 60)


if __name__ == "__main__":
    main()
