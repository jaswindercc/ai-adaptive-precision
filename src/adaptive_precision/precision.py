"""Precision level definitions and utilities."""

from __future__ import annotations

from enum import Enum
from typing import Dict


class PrecisionLevel(Enum):
    """Supported numerical precision levels for inference."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"

    @property
    def bits(self) -> int:
        return {"fp32": 32, "fp16": 16, "bf16": 16, "int8": 8}[self.value]

    @property
    def speedup_factor(self) -> float:
        """Approximate speedup relative to FP32."""
        return {"fp32": 1.0, "fp16": 1.8, "bf16": 1.7, "int8": 2.5}[self.value]

    @property
    def memory_ratio(self) -> float:
        """Memory usage relative to FP32."""
        return {"fp32": 1.0, "fp16": 0.5, "bf16": 0.5, "int8": 0.25}[self.value]

    @property
    def typical_accuracy_drop(self) -> Dict[str, float]:
        """Typical accuracy degradation ranges (min, max) relative to FP32."""
        ranges = {
            "fp32": (0.0, 0.0),
            "fp16": (0.001, 0.005),
            "bf16": (0.001, 0.005),
            "int8": (0.01, 0.03),
        }
        mn, mx = ranges[self.value]
        return {"min": mn, "max": mx}

    def __str__(self) -> str:
        return self.value.upper()
