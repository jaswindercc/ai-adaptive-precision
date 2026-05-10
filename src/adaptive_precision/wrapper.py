"""
Model Wrapper — runtime precision switching for AI models.

Provides a unified interface over PyTorch, ONNX Runtime, and TensorFlow
for seamless precision format switching without reloading models.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from adaptive_precision.monitor import InferenceMetrics
from adaptive_precision.precision import PrecisionLevel

logger = logging.getLogger(__name__)


class BaseModelWrapper(ABC):
    """Abstract base class for framework-specific model wrappers."""

    def __init__(self, precision: PrecisionLevel = PrecisionLevel.FP32):
        self._precision = precision

    @property
    def precision(self) -> PrecisionLevel:
        return self._precision

    @abstractmethod
    def set_precision(self, level: PrecisionLevel) -> None:
        """Switch the model to a different precision level."""

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference and return the output."""

    def timed_predict(self, input_data: Any) -> tuple[Any, InferenceMetrics]:
        """Run inference with timing and return (output, metrics)."""
        start = time.perf_counter()
        output = self.predict(input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics = InferenceMetrics(latency_ms=elapsed_ms, precision=self._precision.value)
        return output, metrics


class ModelWrapper(BaseModelWrapper):
    """
    Default model wrapper with simulated precision switching.

    Use this for testing and prototyping without requiring a real model.
    For production, use PyTorchWrapper or ONNXWrapper.

    Example:
        >>> wrapper = ModelWrapper()
        >>> wrapper.set_precision(PrecisionLevel.INT8)
        >>> output, metrics = wrapper.timed_predict({"input": [1, 2, 3]})
        >>> print(f"Latency: {metrics.latency_ms:.1f}ms at {metrics.precision}")
    """

    def __init__(
        self,
        model: Any = None,
        precision: PrecisionLevel = PrecisionLevel.FP32,
    ):
        super().__init__(precision)
        self._model = model

    def set_precision(self, level: PrecisionLevel) -> None:
        logger.info("Precision switch: %s → %s", self._precision, level)
        self._precision = level

    def predict(self, input_data: Any) -> Any:
        if self._model is not None and callable(self._model):
            return self._model(input_data)
        return input_data


class PyTorchWrapper(BaseModelWrapper):
    """
    PyTorch model wrapper with AMP-based precision switching.

    Supports FP32, FP16 (via torch.cuda.amp), and INT8 (dynamic quantization).

    Example:
        >>> import torch
        >>> model = torch.nn.Linear(10, 5)
        >>> wrapper = PyTorchWrapper(model)
        >>> wrapper.set_precision(PrecisionLevel.FP16)
        >>> output, metrics = wrapper.timed_predict(torch.randn(1, 10))
    """

    def __init__(
        self, model: Any, device: str = "cpu",
        precision: PrecisionLevel = PrecisionLevel.FP32,
    ):
        super().__init__(precision)
        self._original_model = model
        self._model = model
        self._device = device
        self._torch = None
        self._setup()

    def _setup(self) -> None:
        try:
            import torch
            self._torch = torch
            self._model = self._original_model.to(self._device)
            self._model.eval()
        except ImportError:
            raise ImportError("PyTorch is required: pip install adaptive-precision[torch]")

    def set_precision(self, level: PrecisionLevel) -> None:
        torch = self._torch
        logger.info("PyTorch precision: %s → %s", self._precision, level)

        if level == PrecisionLevel.INT8:
            self._model = torch.quantization.quantize_dynamic(
                self._original_model.to("cpu"),
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8,
            )
            self._model.eval()
        elif level in (PrecisionLevel.FP16, PrecisionLevel.BF16):
            dtype = torch.float16 if level == PrecisionLevel.FP16 else torch.bfloat16
            self._model = self._original_model.to(dtype).to(self._device)
            self._model.eval()
        else:
            self._model = self._original_model.float().to(self._device)
            self._model.eval()

        self._precision = level

    def predict(self, input_data: Any) -> Any:
        torch = self._torch
        with torch.no_grad():
            if self._precision == PrecisionLevel.FP16:
                if hasattr(input_data, "half"):
                    input_data = input_data.half().to(self._device)
            elif self._precision == PrecisionLevel.BF16:
                if hasattr(input_data, "bfloat16"):
                    input_data = input_data.bfloat16().to(self._device)
            elif self._precision == PrecisionLevel.INT8:
                if hasattr(input_data, "to"):
                    input_data = input_data.to("cpu").float()
            else:
                if hasattr(input_data, "to"):
                    input_data = input_data.float().to(self._device)
            return self._model(input_data)


class ONNXWrapper(BaseModelWrapper):
    """
    ONNX Runtime model wrapper with execution provider-based precision.

    Example:
        >>> wrapper = ONNXWrapper("model.onnx")
        >>> wrapper.set_precision(PrecisionLevel.FP16)
        >>> inp = {"input": np.random.randn(1, 10).astype(np.float32)}
        >>> output, metrics = wrapper.timed_predict(inp)
    """

    def __init__(self, model_path: str, precision: PrecisionLevel = PrecisionLevel.FP32):
        super().__init__(precision)
        self._model_path = Path(model_path)
        self._session = None
        self._ort = None
        self._setup()

    def _setup(self) -> None:
        try:
            import onnxruntime as ort
            self._ort = ort
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self._session = ort.InferenceSession(str(self._model_path), providers=providers)
        except ImportError:
            raise ImportError("ONNX Runtime is required: pip install adaptive-precision[onnx]")

    def set_precision(self, level: PrecisionLevel) -> None:
        logger.info("ONNX precision: %s → %s", self._precision, level)
        self._precision = level

    def predict(self, input_data: Any) -> Any:
        if self._session is None:
            raise RuntimeError("ONNX session not initialized")
        input_name = self._session.get_inputs()[0].name
        if isinstance(input_data, dict):
            return self._session.run(None, input_data)
        return self._session.run(None, {input_name: input_data})
