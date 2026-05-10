"""
Monitoring Agent — lightweight telemetry module for hardware metrics.

Collects CPU/GPU utilization, memory, temperature, and power metrics
for real-time precision scaling decisions.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

import psutil


@dataclass
class HardwareMetrics:
    """Snapshot of hardware metrics at a point in time."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    cpu_temperature_c: Optional[float] = None

    @property
    def is_gpu_available(self) -> bool:
        return self.gpu_utilization is not None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": round(self.memory_used_gb, 2),
            "memory_total_gb": round(self.memory_total_gb, 2),
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "gpu_temperature_c": self.gpu_temperature_c,
            "gpu_power_watts": self.gpu_power_watts,
            "cpu_temperature_c": self.cpu_temperature_c,
        }


@dataclass
class InferenceMetrics:
    """Metrics from a single inference run."""

    latency_ms: float
    precision: str
    accuracy: Optional[float] = None
    throughput: Optional[float] = None
    energy_joules: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "latency_ms": round(self.latency_ms, 3),
            "precision": self.precision,
            "accuracy": self.accuracy,
            "throughput": self.throughput,
            "energy_joules": self.energy_joules,
            "timestamp": self.timestamp,
        }


class MonitoringAgent:
    """
    Lightweight telemetry module that collects hardware and inference metrics.

    Interfaces with system sensors (psutil, NVML) and maintains a rolling
    buffer of metrics for the Precision Controller to consume.

    Example:
        >>> agent = MonitoringAgent(history_size=100)
        >>> metrics = agent.collect()
        >>> print(f"CPU: {metrics.cpu_percent}%")
    """

    def __init__(self, history_size: int = 500):
        self._history_size = history_size
        self._hw_history: Deque[HardwareMetrics] = deque(maxlen=history_size)
        self._inf_history: Deque[InferenceMetrics] = deque(maxlen=history_size)
        self._gpu_available = False
        self._nvml_handle = None
        self._callbacks: List[Callable[[HardwareMetrics], None]] = []
        self._init_gpu()

    def _init_gpu(self) -> None:
        """Try to initialize NVIDIA GPU monitoring via pynvml."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_available = True
        except Exception:
            self._gpu_available = False

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    def on_collect(self, callback: Callable[[HardwareMetrics], None]) -> None:
        """Register a callback invoked after each metric collection."""
        self._callbacks.append(callback)

    def collect(self) -> HardwareMetrics:
        """Collect a snapshot of current hardware metrics."""
        mem = psutil.virtual_memory()

        metrics = HardwareMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=mem.percent,
            memory_used_gb=mem.used / (1024**3),
            memory_total_gb=mem.total / (1024**3),
        )

        # CPU temperature (Linux)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name in ("coretemp", "k10temp", "cpu_thermal", "cpu-thermal"):
                    if name in temps and temps[name]:
                        metrics.cpu_temperature_c = temps[name][0].current
                        break
        except Exception:
            pass

        # GPU metrics via NVML
        if self._gpu_available and self._nvml_handle:
            try:
                import pynvml

                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                metrics.gpu_utilization = util.gpu
                metrics.gpu_memory_used_gb = mem_info.used / (1024**3)
                metrics.gpu_memory_total_gb = mem_info.total / (1024**3)
                try:
                    metrics.gpu_temperature_c = pynvml.nvmlDeviceGetTemperature(
                        self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    pass
                try:
                    metrics.gpu_power_watts = (
                        pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                    )
                except Exception:
                    pass
            except Exception:
                pass

        self._hw_history.append(metrics)

        for cb in self._callbacks:
            cb(metrics)

        return metrics

    def record_inference(self, metrics: InferenceMetrics) -> None:
        """Record inference metrics for historical tracking."""
        self._inf_history.append(metrics)

    @property
    def hardware_history(self) -> List[HardwareMetrics]:
        return list(self._hw_history)

    @property
    def inference_history(self) -> List[InferenceMetrics]:
        return list(self._inf_history)

    def avg_cpu_percent(self, last_n: int = 10) -> float:
        """Average CPU usage over the last N samples."""
        samples = list(self._hw_history)[-last_n:]
        if not samples:
            return 0.0
        return sum(s.cpu_percent for s in samples) / len(samples)

    def avg_gpu_utilization(self, last_n: int = 10) -> Optional[float]:
        """Average GPU utilization over the last N samples."""
        samples = [s for s in list(self._hw_history)[-last_n:] if s.gpu_utilization is not None]
        if not samples:
            return None
        return sum(s.gpu_utilization for s in samples) / len(samples)

    def avg_latency_ms(self, last_n: int = 10) -> Optional[float]:
        """Average inference latency over the last N samples."""
        samples = list(self._inf_history)[-last_n:]
        if not samples:
            return None
        return sum(s.latency_ms for s in samples) / len(samples)

    def summary(self) -> Dict:
        """Return a summary of current hardware state."""
        latest = self.collect()
        return {
            "hardware": latest.to_dict(),
            "avg_cpu_10": round(self.avg_cpu_percent(10), 1),
            "avg_gpu_10": self.avg_gpu_utilization(10),
            "avg_latency_10": self.avg_latency_ms(10),
            "hw_samples": len(self._hw_history),
            "inf_samples": len(self._inf_history),
        }
