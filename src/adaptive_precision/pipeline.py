"""
Adaptive Precision Scaling Pipeline — ties all components together.

Provides a high-level AutoScaler that runs the full feedback loop:
Monitor → Controller → Wrapper → Cost Estimator.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from adaptive_precision.controller import PrecisionController, PrecisionDecision, Strategy
from adaptive_precision.cost_estimator import CostEstimator
from adaptive_precision.monitor import MonitoringAgent
from adaptive_precision.precision import PrecisionLevel
from adaptive_precision.wrapper import BaseModelWrapper, ModelWrapper

logger = logging.getLogger(__name__)


class AutoScaler:
    """
    High-level pipeline that runs the adaptive precision feedback loop.

    Connects MonitoringAgent, PrecisionController, ModelWrapper, and
    CostEstimator into a single predict() call that automatically
    adjusts precision based on real-time conditions.

    Example:
        >>> from adaptive_precision import AutoScaler
        >>> scaler = AutoScaler(hardware_type="nvidia_t4")
        >>> result = scaler.predict(input_data)
        >>> print(f"Precision: {result['precision']}, Cost: ${result['cost_usd']:.6f}")
    """

    def __init__(
        self,
        model: Any = None,
        wrapper: Optional[BaseModelWrapper] = None,
        strategy: Strategy = Strategy.BALANCED,
        hardware_type: str = "cpu_generic",
        auto_adjust: bool = True,
        adjust_interval: int = 5,
    ):
        self.monitor = MonitoringAgent()
        self.controller = PrecisionController(self.monitor, strategy=strategy)
        self.estimator = CostEstimator(hardware_type=hardware_type)
        self.wrapper = wrapper or ModelWrapper(model=model)

        self._auto_adjust = auto_adjust
        self._adjust_interval = adjust_interval
        self._call_count = 0
        self._on_switch: List[Callable[[PrecisionDecision], None]] = []

    def on_precision_switch(self, callback: Callable[[PrecisionDecision], None]) -> None:
        """Register a callback for precision switches."""
        self._on_switch.append(callback)

    def predict(self, input_data: Any) -> Dict:
        """
        Run inference with automatic precision scaling.

        Returns a dict with output, precision, metrics, and cost info.
        """
        self._call_count += 1

        # Periodically re-evaluate precision
        if self._auto_adjust and self._call_count % self._adjust_interval == 1:
            decision = self.controller.decide()
            if decision.selected != self.wrapper.precision:
                self.wrapper.set_precision(decision.selected)
                for cb in self._on_switch:
                    cb(decision)

        # Run inference with timing
        output, inf_metrics = self.wrapper.timed_predict(input_data)
        self.monitor.record_inference(inf_metrics)

        # Estimate cost
        cost = self.estimator.estimate(inf_metrics.latency_ms, self.wrapper.precision)

        return {
            "output": output,
            "precision": str(self.wrapper.precision),
            "latency_ms": round(inf_metrics.latency_ms, 3),
            "cost_usd": cost.total_cost_usd,
            "energy_kwh": cost.energy_kwh,
            "carbon_grams": cost.carbon_grams,
        }

    def benchmark(self, input_data: Any, iterations: int = 100) -> Dict:
        """
        Run a benchmark comparing all precision levels.

        Returns performance metrics for each precision level.
        """
        results = {}
        original = self.wrapper.precision

        for level in [PrecisionLevel.FP32, PrecisionLevel.FP16, PrecisionLevel.INT8]:
            try:
                self.wrapper.set_precision(level)
            except Exception as e:
                logger.warning("Cannot set %s: %s", level, e)
                continue

            latencies = []
            for _ in range(iterations):
                _, metrics = self.wrapper.timed_predict(input_data)
                latencies.append(metrics.latency_ms)

            avg_lat = sum(latencies) / len(latencies)
            cost = self.estimator.estimate(avg_lat, level)

            results[str(level)] = {
                "avg_latency_ms": round(avg_lat, 3),
                "min_latency_ms": round(min(latencies), 3),
                "max_latency_ms": round(max(latencies), 3),
                "throughput_per_sec": round(1000.0 / avg_lat, 1),
                "cost_per_1k_usd": round(cost.total_cost_usd * 1000, 6),
                "energy_kwh_per_1k": round(cost.energy_kwh * 1000, 8),
                "carbon_g_per_1k": round(cost.carbon_grams * 1000, 6),
            }

        # Restore original precision
        try:
            self.wrapper.set_precision(original)
        except Exception:
            pass

        return results

    def status(self) -> Dict:
        """Return current system status."""
        return {
            "precision": str(self.wrapper.precision),
            "strategy": self.controller.strategy.value,
            "total_inferences": self._call_count,
            "total_cost_usd": round(self.estimator.total_cost(), 6),
            "total_carbon_grams": round(self.estimator.total_carbon(), 4),
            "hardware": self.monitor.summary(),
        }
