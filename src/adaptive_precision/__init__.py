"""
Adaptive Precision Scaling System for AI Inference.

Dynamically adjusts numerical precision (FP32, FP16, INT8) of AI model
inference based on real-time hardware metrics, cost, and energy constraints.

Author: Jaswinder Singh (https://jaswinder.cc/)
Paper: "Low-Cost AI Infrastructure: Strategies and Optimization"
"""

from adaptive_precision.controller import PrecisionController
from adaptive_precision.cost_estimator import CostEstimator
from adaptive_precision.monitor import MonitoringAgent
from adaptive_precision.precision import PrecisionLevel
from adaptive_precision.wrapper import ModelWrapper

__version__ = "0.1.0"
__author__ = "Jaswinder Singh"
__url__ = "https://jaswinder.cc/"

__all__ = [
    "MonitoringAgent",
    "PrecisionController",
    "ModelWrapper",
    "CostEstimator",
    "PrecisionLevel",
]
