"""
Cost Analysis Example — Compare inference costs across hardware & precisions.

Demonstrates the Cost Estimator module with detailed breakdowns for
cloud vs. edge deployment scenarios.

Author: Jaswinder Singh (https://jaswinder.cc/)
"""

from adaptive_precision.cost_estimator import CostEstimator
from adaptive_precision.precision import PrecisionLevel


def main():
    print("=" * 65)
    print("  Cost Analysis — Cloud vs Edge Deployment")
    print("  Paper: Low-Cost AI Infrastructure by Jaswinder Singh")
    print("=" * 65)

    hardware_configs = [
        ("nvidia_a100", "NVIDIA A100 (Cloud Training)"),
        ("nvidia_t4", "NVIDIA T4 (Cloud Inference)"),
        ("nvidia_l4", "NVIDIA L4 (Gen AI Inference)"),
        ("jetson_nano", "Jetson Nano (Edge)"),
        ("cpu_generic", "Generic CPU"),
    ]

    fp32_latency_ms = 30.0
    batch_size = 10_000

    for hw_type, hw_name in hardware_configs:
        print(f"\n{'─' * 65}")
        print(f"  {hw_name} ({hw_type})")
        print(f"{'─' * 65}")

        estimator = CostEstimator(hardware_type=hw_type)
        comparison = estimator.compare_precisions(fp32_latency_ms, count=batch_size)

        print(f"  {'Precision':<8} {'Latency':<10} {'Cost/10K':<14} {'Energy kWh':<14} {'CO₂ (g)'}")
        for prec, data in comparison.items():
            print(
                f"  {prec:<8} {data['latency_ms']:<10.2f} "
                f"${data['cost_per_1k']:<13.6f} "
                f"{data['energy_kwh_per_1k']:<14.8f} "
                f"{data['carbon_g_per_1k']:.4f}"
            )

        # Show savings from FP32 → INT8
        fp32 = comparison["FP32"]
        int8 = comparison["INT8"]
        cost_saving = (1 - int8["cost_per_1k"] / fp32["cost_per_1k"]) * 100
        energy_saving = (1 - int8["energy_kwh_per_1k"] / fp32["energy_kwh_per_1k"]) * 100
        speed_gain = (fp32["latency_ms"] / int8["latency_ms"] - 1) * 100

        print(f"\n  💡 FP32 → INT8 Savings:")
        print(f"     Cost:    {cost_saving:.1f}% reduction")
        print(f"     Energy:  {energy_saving:.1f}% reduction")
        print(f"     Speed:   {speed_gain:.1f}% faster")

    # Monthly cost projection
    print(f"\n{'=' * 65}")
    print("  Monthly Cost Projection (1M inferences/month)")
    print(f"{'=' * 65}")

    inferences_per_month = 1_000_000
    print(f"\n  {'Hardware':<20} {'FP32':<14} {'FP16':<14} {'INT8':<14} {'Savings'}")
    print(f"  {'─' * 70}")

    for hw_type, hw_name in hardware_configs:
        estimator = CostEstimator(hardware_type=hw_type)
        comp = estimator.compare_precisions(fp32_latency_ms, count=inferences_per_month)
        fp32_cost = comp["FP32"]["cost_per_1k"]
        fp16_cost = comp["FP16"]["cost_per_1k"]
        int8_cost = comp["INT8"]["cost_per_1k"]
        saving = fp32_cost - int8_cost

        short_name = hw_name.split("(")[0].strip()
        print(
            f"  {short_name:<20} ${fp32_cost:<13.2f} ${fp16_cost:<13.2f} "
            f"${int8_cost:<13.2f} ${saving:.2f}"
        )

    print(f"\n  Visit https://jaswinder.cc/ for the full paper.")


if __name__ == "__main__":
    main()
