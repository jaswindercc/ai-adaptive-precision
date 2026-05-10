"""
Controller Strategies Example — Compare all precision control strategies.

Demonstrates how different strategies (rule-based, cost-optimized,
accuracy-first, energy-efficient, balanced) make decisions.

Author: Jaswinder Singh (https://jaswinder.cc/)
"""

from adaptive_precision import MonitoringAgent, PrecisionController
from adaptive_precision.controller import Strategy


def main():
    print("=" * 60)
    print("  Precision Controller — Strategy Comparison")
    print("  Paper: Low-Cost AI Infrastructure by Jaswinder Singh")
    print("=" * 60)

    agent = MonitoringAgent()
    hw = agent.collect()

    print(f"\n  Current Hardware State:")
    print(f"    CPU:    {hw.cpu_percent}%")
    print(f"    Memory: {hw.memory_percent}%")
    print(f"    GPU:    {'Available' if hw.is_gpu_available else 'Not available'}")

    print(f"\n  {'Strategy':<20} {'Selected':<10} {'Reason'}")
    print(f"  {'─' * 60}")

    for strategy in Strategy:
        controller = PrecisionController(agent, strategy=strategy)
        decision = controller.decide(hw)
        print(f"  {strategy.value:<20} {str(decision.selected):<10} {decision.reason}")

    # Simulate high-load scenario
    print(f"\n  --- Simulated High-Load Scenario ---")
    print(f"  (Manually setting high CPU to simulate pressure)\n")

    # Create a modified metrics snapshot
    from adaptive_precision.monitor import HardwareMetrics
    import time

    high_load = HardwareMetrics(
        timestamp=time.time(),
        cpu_percent=92.0,
        memory_percent=85.0,
        memory_used_gb=13.6,
        memory_total_gb=16.0,
        gpu_utilization=95.0,
        gpu_temperature_c=82.0,
        gpu_power_watts=280.0,
    )

    print(f"  Simulated: CPU={high_load.cpu_percent}%, GPU={high_load.gpu_utilization}%, "
          f"Temp={high_load.gpu_temperature_c}°C")
    print()
    print(f"  {'Strategy':<20} {'Selected':<10} {'Reason'}")
    print(f"  {'─' * 60}")

    for strategy in Strategy:
        controller = PrecisionController(agent, strategy=strategy)
        decision = controller.decide(high_load)
        print(f"  {strategy.value:<20} {str(decision.selected):<10} {decision.reason}")

    print(f"\n  Visit https://jaswinder.cc/ for the full paper.")


if __name__ == "__main__":
    main()
