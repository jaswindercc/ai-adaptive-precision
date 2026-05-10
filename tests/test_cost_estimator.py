"""Tests for CostEstimator."""

from adaptive_precision.cost_estimator import CostBreakdown, CostEstimator
from adaptive_precision.precision import PrecisionLevel


class TestCostEstimator:
    def test_estimate_basic(self):
        est = CostEstimator(hardware_type="nvidia_t4")
        cost = est.estimate(latency_ms=20.0, precision=PrecisionLevel.FP32)
        assert isinstance(cost, CostBreakdown)
        assert cost.total_cost_usd > 0
        assert cost.energy_kwh > 0
        assert cost.carbon_grams > 0

    def test_precision_reduces_cost(self):
        est = CostEstimator(hardware_type="nvidia_t4")
        fp32 = est.estimate(latency_ms=20.0, precision=PrecisionLevel.FP32)
        int8 = est.estimate(latency_ms=20.0, precision=PrecisionLevel.INT8)
        # INT8 should have lower compute cost due to memory_ratio
        assert int8.compute_cost_usd < fp32.compute_cost_usd

    def test_estimate_batch(self):
        est = CostEstimator(hardware_type="nvidia_t4")
        batch = est.estimate_batch(count=1000, avg_latency_ms=15.0, precision=PrecisionLevel.FP16)
        assert batch["count"] == 1000
        assert batch["total_cost_usd"] > 0

    def test_compare_precisions(self):
        est = CostEstimator(hardware_type="nvidia_a100")
        comp = est.compare_precisions(latency_fp32_ms=30.0, count=1000)
        assert "FP32" in comp
        assert "INT8" in comp
        # INT8 should cost less per 1K
        assert comp["INT8"]["cost_per_1k"] < comp["FP32"]["cost_per_1k"]

    def test_cost_history(self):
        est = CostEstimator()
        est.estimate(10.0, PrecisionLevel.FP32)
        est.estimate(10.0, PrecisionLevel.FP16)
        assert len(est.cost_history) == 2

    def test_total_cost(self):
        est = CostEstimator()
        est.estimate(10.0, PrecisionLevel.FP32)
        est.estimate(10.0, PrecisionLevel.FP16)
        assert est.total_cost() > 0

    def test_total_carbon(self):
        est = CostEstimator()
        est.estimate(10.0, PrecisionLevel.FP32)
        assert est.total_carbon() > 0

    def test_custom_pricing(self):
        est = CostEstimator(hardware_type="custom", cost_per_hour=100.0)
        cost = est.estimate(10.0, PrecisionLevel.FP32)
        assert cost.compute_cost_usd > 0

    def test_cost_breakdown_to_dict(self):
        est = CostEstimator()
        cost = est.estimate(10.0, PrecisionLevel.FP16)
        d = cost.to_dict()
        assert "compute_cost_usd" in d
        assert "carbon_grams" in d
        assert d["precision"] == "FP16"
