"""
Cost Estimator — real-time inference cost calculation.

Computes cost per inference using cloud pricing or local energy metrics,
including carbon footprint estimation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from adaptive_precision.monitor import HardwareMetrics
from adaptive_precision.precision import PrecisionLevel


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an inference operation."""

    compute_cost_usd: float
    energy_cost_usd: float
    total_cost_usd: float
    energy_kwh: float
    carbon_grams: float
    precision: PrecisionLevel
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "compute_cost_usd": round(self.compute_cost_usd, 8),
            "energy_cost_usd": round(self.energy_cost_usd, 8),
            "total_cost_usd": round(self.total_cost_usd, 8),
            "energy_kwh": round(self.energy_kwh, 10),
            "carbon_grams": round(self.carbon_grams, 6),
            "precision": str(self.precision),
            "latency_ms": round(self.latency_ms, 3),
        }


# Default cloud GPU pricing (USD per hour) — approximate market rates
DEFAULT_CLOUD_PRICING: Dict[str, float] = {
    "nvidia_a100": 35.0,
    "nvidia_h100": 45.0,
    "nvidia_v100": 20.0,
    "nvidia_t4": 0.55,
    "nvidia_l4": 0.75,
    "cpu_generic": 0.10,
    "jetson_nano": 0.01,  # Amortized hardware cost
}


class CostEstimator:
    """
    Computes real-time cost metrics for each inference operation.

    Supports cloud pricing APIs, local energy metrics, and carbon footprint
    estimation. Feeds cost data into the Precision Controller.

    Example:
        >>> estimator = CostEstimator(hardware_type="nvidia_t4")
        >>> cost = estimator.estimate(latency_ms=15.0, precision=PrecisionLevel.FP16)
        >>> print(f"Cost: ${cost.total_cost_usd:.6f} | CO2: {cost.carbon_grams:.4f}g")

    From the paper:
        C_inf = C_hw × T_inf
        C_total = C_hw × T_inf + E_inf × P_kWh
    """

    def __init__(
        self,
        hardware_type: str = "cpu_generic",
        cost_per_hour: Optional[float] = None,
        energy_price_kwh: float = 0.12,
        carbon_intensity_g_kwh: float = 400.0,
        power_draw_watts: Optional[float] = None,
    ):
        self._hardware_type = hardware_type
        self._cost_per_hour = cost_per_hour or DEFAULT_CLOUD_PRICING.get(hardware_type, 0.10)
        self._energy_price_kwh = energy_price_kwh
        self._carbon_intensity = carbon_intensity_g_kwh
        self._power_draw_watts = power_draw_watts or self._default_power(hardware_type)
        self._history: List[CostBreakdown] = []

    @staticmethod
    def _default_power(hw: str) -> float:
        """Default TDP estimates in watts."""
        return {
            "nvidia_a100": 400.0,
            "nvidia_h100": 700.0,
            "nvidia_v100": 300.0,
            "nvidia_t4": 70.0,
            "nvidia_l4": 72.0,
            "cpu_generic": 65.0,
            "jetson_nano": 10.0,
        }.get(hw, 65.0)

    def estimate(
        self,
        latency_ms: float,
        precision: PrecisionLevel,
        hw_metrics: Optional[HardwareMetrics] = None,
    ) -> CostBreakdown:
        """
        Estimate the cost of a single inference.

        C_total = C_hw × T_inf + E_inf × P_kWh

        Args:
            latency_ms: Inference latency in milliseconds.
            precision: The precision level used.
            hw_metrics: Optional hardware metrics for power-aware estimation.

        Returns:
            CostBreakdown with compute, energy, and carbon costs.
        """
        t_seconds = latency_ms / 1000.0

        # Compute cost: C_hw × T_inf
        compute_cost = (self._cost_per_hour / 3600.0) * t_seconds

        # Adjust for precision efficiency
        compute_cost *= precision.memory_ratio

        # Energy estimation
        power_w = self._power_draw_watts
        if hw_metrics and hw_metrics.gpu_power_watts:
            power_w = hw_metrics.gpu_power_watts

        energy_kwh = (power_w * t_seconds) / (3600 * 1000)
        energy_cost = energy_kwh * self._energy_price_kwh

        # Carbon footprint
        carbon_g = energy_kwh * self._carbon_intensity

        breakdown = CostBreakdown(
            compute_cost_usd=compute_cost,
            energy_cost_usd=energy_cost,
            total_cost_usd=compute_cost + energy_cost,
            energy_kwh=energy_kwh,
            carbon_grams=carbon_g,
            precision=precision,
            latency_ms=latency_ms,
        )

        self._history.append(breakdown)
        return breakdown

    def estimate_batch(
        self,
        count: int,
        avg_latency_ms: float,
        precision: PrecisionLevel,
    ) -> Dict:
        """Estimate cost for a batch of N inferences."""
        single = self.estimate(avg_latency_ms, precision)
        return {
            "count": count,
            "per_inference": single.to_dict(),
            "total_cost_usd": round(single.total_cost_usd * count, 6),
            "total_energy_kwh": round(single.energy_kwh * count, 8),
            "total_carbon_grams": round(single.carbon_grams * count, 4),
        }

    def compare_precisions(self, latency_fp32_ms: float, count: int = 1000) -> Dict:
        """Compare costs across precision levels for a given workload."""
        results = {}
        for level in PrecisionLevel:
            adjusted_latency = latency_fp32_ms / level.speedup_factor
            batch = self.estimate_batch(count, adjusted_latency, level)
            results[str(level)] = {
                "latency_ms": round(adjusted_latency, 2),
                "cost_per_1k": batch["total_cost_usd"],
                "energy_kwh_per_1k": batch["total_energy_kwh"],
                "carbon_g_per_1k": batch["total_carbon_grams"],
            }
        return results

    @property
    def cost_history(self) -> List[CostBreakdown]:
        return list(self._history)

    def total_cost(self) -> float:
        """Total cost across all recorded inferences."""
        return sum(c.total_cost_usd for c in self._history)

    def total_carbon(self) -> float:
        """Total CO2 emissions in grams."""
        return sum(c.carbon_grams for c in self._history)
