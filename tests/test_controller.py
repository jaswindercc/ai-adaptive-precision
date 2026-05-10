"""Tests for PrecisionController."""

import time

from adaptive_precision.controller import (
    PrecisionController,
    PrecisionThresholds,
    Strategy,
)
from adaptive_precision.monitor import HardwareMetrics, MonitoringAgent
from adaptive_precision.precision import PrecisionLevel


def _make_metrics(**kwargs) -> HardwareMetrics:
    defaults = dict(
        timestamp=time.time(),
        cpu_percent=30.0,
        memory_percent=40.0,
        memory_used_gb=6.4,
        memory_total_gb=16.0,
    )
    defaults.update(kwargs)
    return HardwareMetrics(**defaults)


class TestPrecisionController:
    def test_default_decision(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(agent, strategy=Strategy.BALANCED)
        decision = ctrl.decide()
        assert decision.selected in PrecisionLevel
        assert decision.strategy == Strategy.BALANCED

    def test_rule_based_nominal(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(agent, strategy=Strategy.RULE_BASED)
        m = _make_metrics(cpu_percent=30.0)
        decision = ctrl.decide(m)
        assert decision.selected in PrecisionLevel

    def test_rule_based_high_gpu_temp(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(agent, strategy=Strategy.RULE_BASED)
        m = _make_metrics(gpu_temperature_c=85.0, gpu_utilization=50.0)
        decision = ctrl.decide(m)
        assert decision.selected == PrecisionLevel.INT8
        assert "temp" in decision.reason.lower()

    def test_rule_based_high_gpu_util(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(agent, strategy=Strategy.RULE_BASED)
        m = _make_metrics(gpu_utilization=95.0)
        decision = ctrl.decide(m)
        # Should step down from FP32
        assert decision.selected in (PrecisionLevel.FP16, PrecisionLevel.INT8)

    def test_cost_optimized(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(agent, strategy=Strategy.COST_OPTIMIZED)
        decision = ctrl.decide(_make_metrics())
        assert decision.selected == PrecisionLevel.INT8

    def test_accuracy_first(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(agent, strategy=Strategy.ACCURACY_FIRST)
        decision = ctrl.decide(_make_metrics())
        assert decision.selected == PrecisionLevel.FP32

    def test_energy_efficient(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(agent, strategy=Strategy.ENERGY_EFFICIENT)
        decision = ctrl.decide(_make_metrics())
        assert decision.selected in (PrecisionLevel.FP16, PrecisionLevel.INT8)

    def test_accuracy_degradation_fallback(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(agent, strategy=Strategy.COST_OPTIMIZED)
        ctrl.set_accuracy(baseline=0.95, current=0.90)  # 5% drop > 2% threshold
        decision = ctrl.decide(_make_metrics())
        # Should step up from FP32 (stays FP32)
        assert decision.selected == PrecisionLevel.FP32

    def test_decision_history(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(agent, strategy=Strategy.BALANCED)
        ctrl.decide(_make_metrics())
        ctrl.decide(_make_metrics())
        assert len(ctrl.decision_history) == 2

    def test_custom_thresholds(self):
        agent = MonitoringAgent()
        thresholds = PrecisionThresholds(max_gpu_temp_c=60.0)
        ctrl = PrecisionController(agent, strategy=Strategy.RULE_BASED, thresholds=thresholds)
        m = _make_metrics(gpu_temperature_c=65.0, gpu_utilization=30.0)
        decision = ctrl.decide(m)
        assert decision.selected == PrecisionLevel.INT8

    def test_allowed_levels(self):
        agent = MonitoringAgent()
        ctrl = PrecisionController(
            agent, strategy=Strategy.COST_OPTIMIZED,
            allowed_levels=[PrecisionLevel.FP32, PrecisionLevel.FP16],
        )
        decision = ctrl.decide(_make_metrics())
        assert decision.selected in (PrecisionLevel.FP32, PrecisionLevel.FP16)
