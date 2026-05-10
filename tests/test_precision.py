"""Tests for PrecisionLevel."""

from adaptive_precision.precision import PrecisionLevel


class TestPrecisionLevel:
    def test_values(self):
        assert PrecisionLevel.FP32.value == "fp32"
        assert PrecisionLevel.FP16.value == "fp16"
        assert PrecisionLevel.INT8.value == "int8"

    def test_bits(self):
        assert PrecisionLevel.FP32.bits == 32
        assert PrecisionLevel.FP16.bits == 16
        assert PrecisionLevel.INT8.bits == 8

    def test_speedup(self):
        assert PrecisionLevel.FP32.speedup_factor == 1.0
        assert PrecisionLevel.FP16.speedup_factor > 1.0
        assert PrecisionLevel.INT8.speedup_factor > PrecisionLevel.FP16.speedup_factor

    def test_memory_ratio(self):
        assert PrecisionLevel.FP32.memory_ratio == 1.0
        assert PrecisionLevel.FP16.memory_ratio == 0.5
        assert PrecisionLevel.INT8.memory_ratio == 0.25

    def test_accuracy_drop(self):
        drop = PrecisionLevel.INT8.typical_accuracy_drop
        assert drop["min"] > 0
        assert drop["max"] >= drop["min"]
        fp32_drop = PrecisionLevel.FP32.typical_accuracy_drop
        assert fp32_drop["min"] == 0.0

    def test_str(self):
        assert str(PrecisionLevel.FP32) == "FP32"
        assert str(PrecisionLevel.INT8) == "INT8"
