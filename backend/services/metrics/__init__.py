"""
Metric Plugin System for RAG Evaluation.

This module provides a registry-based architecture for evaluation metrics,
allowing easy extension with custom metrics without modifying core code.

Usage:
    from services.metrics import get_metric, list_metrics, compute_metric

    # Get a specific metric
    metric = get_metric("faithfulness")
    score = await metric.compute(question, answer, reference, chunks)

    # List all available metrics
    available = list_metrics()

    # Compute metric with automatic discovery
    result = await compute_metric("answer_correctness", ...)

    # Get metric configuration
    from services.metrics.config import MetricsConfig, get_default_config
    config = get_default_config()
"""

from .registry import (
    register_metric,
    get_metric,
    list_metrics,
    get_metric_info,
    get_metric_versions,
    compute_metric,
)
from .base import BaseMetric, MetricResult
from .config import (
    MetricsConfig,
    FallbackBehavior,
    get_default_config,
    set_default_config,
    STRICT_CONFIG,
    LENIENT_CONFIG,
    FAST_CONFIG,
)

__all__ = [
    # Registry functions
    "register_metric",
    "get_metric",
    "list_metrics",
    "get_metric_info",
    "get_metric_versions",
    "compute_metric",
    # Base classes
    "BaseMetric",
    "MetricResult",
    # Configuration
    "MetricsConfig",
    "FallbackBehavior",
    "get_default_config",
    "set_default_config",
    "STRICT_CONFIG",
    "LENIENT_CONFIG",
    "FAST_CONFIG",
]

# Import metric implementations to register them
from . import faithfulness
from . import correctness
from . import relevancy
from . import context_precision
