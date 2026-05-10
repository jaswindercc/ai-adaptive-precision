"""Tests for MonitoringAgent."""

from adaptive_precision.monitor import HardwareMetrics, InferenceMetrics, MonitoringAgent


class TestHardwareMetrics:
    def test_to_dict(self):
        m = HardwareMetrics(
            timestamp=1000.0,
            cpu_percent=45.0,
            memory_percent=60.0,
            memory_used_gb=9.6,
            memory_total_gb=16.0,
        )
        d = m.to_dict()
        assert d["cpu_percent"] == 45.0
        assert d["memory_used_gb"] == 9.6
        assert d["gpu_utilization"] is None

    def test_gpu_availability(self):
        m = HardwareMetrics(
            timestamp=1000.0, cpu_percent=10.0, memory_percent=30.0,
            memory_used_gb=4.8, memory_total_gb=16.0,
        )
        assert not m.is_gpu_available

        m.gpu_utilization = 50.0
        assert m.is_gpu_available


class TestInferenceMetrics:
    def test_to_dict(self):
        m = InferenceMetrics(latency_ms=15.5, precision="fp16", accuracy=0.95)
        d = m.to_dict()
        assert d["latency_ms"] == 15.5
        assert d["precision"] == "fp16"
        assert d["accuracy"] == 0.95


class TestMonitoringAgent:
    def test_collect(self):
        agent = MonitoringAgent()
        metrics = agent.collect()
        assert isinstance(metrics, HardwareMetrics)
        assert 0 <= metrics.cpu_percent <= 100
        assert metrics.memory_total_gb > 0

    def test_history(self):
        agent = MonitoringAgent(history_size=10)
        for _ in range(5):
            agent.collect()
        assert len(agent.hardware_history) == 5

    def test_record_inference(self):
        agent = MonitoringAgent()
        inf = InferenceMetrics(latency_ms=10.0, precision="fp32")
        agent.record_inference(inf)
        assert len(agent.inference_history) == 1

    def test_avg_cpu(self):
        agent = MonitoringAgent()
        agent.collect()
        agent.collect()
        avg = agent.avg_cpu_percent(2)
        assert avg >= 0

    def test_avg_latency_empty(self):
        agent = MonitoringAgent()
        assert agent.avg_latency_ms() is None

    def test_callback(self):
        results = []
        agent = MonitoringAgent()
        agent.on_collect(lambda m: results.append(m))
        agent.collect()
        assert len(results) == 1

    def test_summary(self):
        agent = MonitoringAgent()
        s = agent.summary()
        assert "hardware" in s
        assert "hw_samples" in s
