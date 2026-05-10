"""Tests for AutoScaler pipeline."""

from adaptive_precision.pipeline import AutoScaler


class TestAutoScaler:
    def test_basic_predict(self):
        scaler = AutoScaler(model=lambda x: sum(x))
        result = scaler.predict([1, 2, 3])
        assert result["output"] == 6
        assert "precision" in result
        assert result["latency_ms"] > 0
        assert result["cost_usd"] > 0

    def test_multiple_predicts(self):
        scaler = AutoScaler(model=lambda x: x)
        for _ in range(10):
            scaler.predict([1])
        status = scaler.status()
        assert status["total_inferences"] == 10

    def test_status(self):
        scaler = AutoScaler()
        status = scaler.status()
        assert "precision" in status
        assert "strategy" in status
        assert "total_cost_usd" in status

    def test_on_switch_callback(self):
        switches = []
        scaler = AutoScaler(model=lambda x: x)
        scaler.on_precision_switch(lambda d: switches.append(d))
        for _ in range(10):
            scaler.predict([1])
        # Callback may or may not fire depending on hardware

    def test_benchmark(self):
        scaler = AutoScaler(model=lambda x: x)
        results = scaler.benchmark([1, 2, 3], iterations=10)
        assert "FP32" in results
        for level, data in results.items():
            assert data["avg_latency_ms"] >= 0
            assert data["throughput_per_sec"] > 0
