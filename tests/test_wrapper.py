"""Tests for ModelWrapper."""

from adaptive_precision.precision import PrecisionLevel
from adaptive_precision.wrapper import ModelWrapper


class TestModelWrapper:
    def test_default_precision(self):
        w = ModelWrapper()
        assert w.precision == PrecisionLevel.FP32

    def test_set_precision(self):
        w = ModelWrapper()
        w.set_precision(PrecisionLevel.INT8)
        assert w.precision == PrecisionLevel.INT8

    def test_predict_passthrough(self):
        w = ModelWrapper()
        out = w.predict([1, 2, 3])
        assert out == [1, 2, 3]

    def test_predict_with_model(self):
        w = ModelWrapper(model=lambda x: sum(x))
        out = w.predict([1, 2, 3])
        assert out == 6

    def test_timed_predict(self):
        w = ModelWrapper(model=lambda x: x)
        out, metrics = w.timed_predict([1, 2])
        assert out == [1, 2]
        assert metrics.latency_ms > 0
        assert metrics.precision == "fp32"

    def test_timed_predict_records_precision(self):
        w = ModelWrapper()
        w.set_precision(PrecisionLevel.FP16)
        _, metrics = w.timed_predict("hello")
        assert metrics.precision == "fp16"
