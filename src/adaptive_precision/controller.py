"""
Precision Controller - the decision engine for adaptive precision scaling.

Dynamically selects optimal precision level (FP32, FP16, INT8) based on
real-time hardware metrics, accuracy, cost, and energy constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from adaptive_precision.monitor import HardwareMetrics, MonitoringAgent
from adaptive_precision.precision import PrecisionLevel

logger = logging.getLogger(__name__)


class Strategy(Enum):
    RULE_BASED = "rule_based"
    COST_OPTIMIZED = "cost_optimized"
    ACCURACY_FIRST = "accuracy_first"
    ENERGY_EFFICIENT = "energy_efficient"
    BALANCED = "balanced"


@dataclass
class PrecisionThresholds:
    max_gpu_temp_c: float = 80.0
    max_gpu_util_percent: float = 90.0
    max_cpu_percent: float = 85.0
    max_latency_ms: float = 100.0
    max_accuracy_drop: float = 0.02
    max_cost_per_inference: float = 0.001
    low_battery_threshold: float = 20.0


@dataclass
class PrecisionDecision:
    previous: PrecisionLevel
    selected: PrecisionLevel
    reason: str
    strategy: Strategy
    metrics_snapshot: Optional[Dict] = None
    score: Optional[float] = None


class PrecisionController:
    def __init__(
        self,
        monitor: MonitoringAgent,
        strategy: Strategy = Strategy.BALANCED,
        thresholds: Optional[PrecisionThresholds] = None,
        allowed_levels: Optional[List[PrecisionLevel]] = None,
    ):
        self._monitor = monitor
        self._strategy = strategy
        self._thresholds = thresholds or PrecisionThresholds()
        self._allowed = allowed_levels or [
            PrecisionLevel.FP32,
            PrecisionLevel.FP16,
            PrecisionLevel.INT8,
        ]
        self._current = PrecisionLevel.FP32
        self._history: List[PrecisionDecision] = []
        self._accuracy_baseline: Optional[float] = None
        self._accuracy_current: Optional[float] = None

    @property
    def current_precision(self) -> PrecisionLevel:
        return self._current

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: Strategy) -> None:
        self._strategy = value

    @property
    def decision_history(self) -> List[PrecisionDecision]:
        return list(self._history)

    def set_accuracy(self, baseline: float, current: float) -> None:
        self._accuracy_baseline = baseline
        self._accuracy_current = current

    def decide(
        self,
        hw_metrics: Optional[HardwareMetrics] = None,
    ) -> PrecisionDecision:
        if hw_metrics is None:
            hw_metrics = self._monitor.collect()

        dispatch = {
            Strategy.RULE_BASED: self._decide_rule_based,
            Strategy.COST_OPTIMIZED: self._decide_cost_optimized,
            Strategy.ACCURACY_FIRST: self._decide_accuracy_first,
            Strategy.ENERGY_EFFICIENT: self._decide_energy_efficient,
            Strategy.BALANCED: self._decide_balanced,
        }

        decision = dispatch[self._strategy](hw_metrics)
        decision.metrics_snapshot = hw_metrics.to_dict()
        self._history.append(decision)
        self._current = decision.selected
        return decision

    def _mk(self, prev, sel, reason, strat, score=None):
        return PrecisionDecision(
            prev, sel, reason, strat, score=score,
        )

    def _decide_rule_based(self, m: HardwareMetrics) -> PrecisionDecision:
        t = self._thresholds
        prev = self._current

        if m.gpu_temperature_c and m.gpu_temperature_c > t.max_gpu_temp_c:
            r = f"GPU temp {m.gpu_temperature_c}C > {t.max_gpu_temp_c}C"
            return self._mk(prev, self._lowest(), r, Strategy.RULE_BASED)

        if m.gpu_utilization and m.gpu_utilization > t.max_gpu_util_percent:
            tgt = self._step_down(prev)
            r = f"GPU util {m.gpu_utilization}% > {t.max_gpu_util_percent}%"
            return self._mk(prev, tgt, r, Strategy.RULE_BASED)

        if not m.is_gpu_available and m.cpu_percent > t.max_cpu_percent:
            tgt = self._step_down(prev)
            r = f"CPU {m.cpu_percent}% > {t.max_cpu_percent}%"
            return self._mk(prev, tgt, r, Strategy.RULE_BASED)

        avg_lat = self._monitor.avg_latency_ms(10)
        if avg_lat and avg_lat > t.max_latency_ms:
            tgt = self._step_down(prev)
            r = f"Avg latency {avg_lat:.1f}ms > {t.max_latency_ms}ms"
            return self._mk(prev, tgt, r, Strategy.RULE_BASED)

        if m.cpu_percent < 50 and (
            m.gpu_utilization is None or m.gpu_utilization < 50
        ):
            tgt = self._step_up(prev)
            r = "Resources underutilized, stepping up"
            return self._mk(prev, tgt, r, Strategy.RULE_BASED)

        return self._mk(
            prev, prev, "Conditions nominal", Strategy.RULE_BASED,
        )

    def _decide_cost_optimized(
        self, m: HardwareMetrics,
    ) -> PrecisionDecision:
        prev = self._current
        if self._accuracy_degraded():
            tgt = self._step_up(prev)
            r = "Accuracy degraded, stepping up for quality"
            return self._mk(prev, tgt, r, Strategy.COST_OPTIMIZED)
        r = "Minimizing cost with lowest precision"
        return self._mk(
            prev, self._lowest(), r, Strategy.COST_OPTIMIZED,
        )

    def _decide_accuracy_first(
        self, m: HardwareMetrics,
    ) -> PrecisionDecision:
        prev = self._current
        thresh = self._thresholds
        if m.gpu_temperature_c and m.gpu_temperature_c > thresh.max_gpu_temp_c:
            tgt = self._step_down(prev)
            r = "Thermal protection override"
            return self._mk(prev, tgt, r, Strategy.ACCURACY_FIRST)
        return self._mk(
            prev, self._highest(), "Prioritizing accuracy",
            Strategy.ACCURACY_FIRST,
        )

    def _decide_energy_efficient(
        self, m: HardwareMetrics,
    ) -> PrecisionDecision:
        prev = self._current
        if self._accuracy_degraded():
            tgt = self._step_up(prev)
            r = "Accuracy guard: stepping up"
            return self._mk(prev, tgt, r, Strategy.ENERGY_EFFICIENT)
        if m.gpu_power_watts and m.gpu_power_watts > 150:
            r = f"High power draw {m.gpu_power_watts}W"
            return self._mk(
                prev, self._lowest(), r, Strategy.ENERGY_EFFICIENT,
            )
        tgt = self._step_down(prev)
        r = "Optimizing for energy efficiency"
        return self._mk(prev, tgt, r, Strategy.ENERGY_EFFICIENT)

    def _decide_balanced(
        self, m: HardwareMetrics,
    ) -> PrecisionDecision:
        prev = self._current
        score_map: Dict[PrecisionLevel, float] = {}
        for level in self._allowed:
            score_map[level] = self._compute_score(level, m)

        best = max(score_map, key=lambda k: score_map[k])
        parts = ", ".join(
            f"{lvl}={sc:.2f}" for lvl, sc in score_map.items()
        )
        return self._mk(
            prev, best,
            f"Balanced score: {parts}",
            Strategy.BALANCED,
            score=score_map[best],
        )

    def _compute_score(
        self, level: PrecisionLevel, m: HardwareMetrics,
    ) -> float:
        speedup = level.speedup_factor
        mem_ratio = level.memory_ratio
        acc_drop = sum(level.typical_accuracy_drop.values()) / 2

        throughput_proxy = speedup * (1 / mem_ratio)
        cost_proxy = mem_ratio
        energy_proxy = level.bits / 32.0
        accuracy_penalty = acc_drop * 10

        util_factor = 1.0
        if m.gpu_utilization and m.gpu_utilization > 70:
            util_factor = 1.0 + (speedup - 1) * 0.5
        elif m.cpu_percent > 70:
            util_factor = 1.0 + (speedup - 1) * 0.5

        denom = cost_proxy + energy_proxy + 0.1
        score = (throughput_proxy * util_factor) / denom
        score -= accuracy_penalty
        return round(score, 4)

    def _accuracy_degraded(self) -> bool:
        if (
            self._accuracy_baseline is None
            or self._accuracy_current is None
        ):
            return False
        drop = self._accuracy_baseline - self._accuracy_current
        return drop > self._thresholds.max_accuracy_drop

    def _step_down(self, current: PrecisionLevel) -> PrecisionLevel:
        order = [
            p for p in [
                PrecisionLevel.FP32,
                PrecisionLevel.FP16,
                PrecisionLevel.INT8,
            ]
            if p in self._allowed
        ]
        idx = order.index(current) if current in order else 0
        return order[min(idx + 1, len(order) - 1)]

    def _step_up(self, current: PrecisionLevel) -> PrecisionLevel:
        order = [
            p for p in [
                PrecisionLevel.FP32,
                PrecisionLevel.FP16,
                PrecisionLevel.INT8,
            ]
            if p in self._allowed
        ]
        if current in order:
            idx = order.index(current)
        else:
            idx = len(order) - 1
        return order[max(idx - 1, 0)]

    def _lowest(self) -> PrecisionLevel:
        for p in [
            PrecisionLevel.INT8,
            PrecisionLevel.FP16,
            PrecisionLevel.FP32,
        ]:
            if p in self._allowed:
                return p
        return PrecisionLevel.FP32

    def _highest(self) -> PrecisionLevel:
        for p in [
            PrecisionLevel.FP32,
            PrecisionLevel.FP16,
            PrecisionLevel.INT8,
        ]:
            if p in self._allowed:
                return p
        return PrecisionLevel.FP32
