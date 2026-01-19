"""
Metric Registry for dynamic metric discovery and management.

The registry allows metrics to be registered at import time using decorators,
enabling a plugin-like architecture where new metrics can be added without
modifying core evaluation code.
"""

import logging
from typing import Any, Optional, Type

from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)

# Global registry of metrics
_METRICS: dict[str, Type[BaseMetric]] = {}


def register_metric(name: str):
    """
    Decorator to register a metric class in the global registry.

    Usage:
        @register_metric("my_metric")
        class MyMetric(BaseMetric):
            ...

    Args:
        name: Unique identifier for the metric

    Returns:
        Decorator function that registers the class
    """

    def decorator(cls: Type[BaseMetric]) -> Type[BaseMetric]:
        if name in _METRICS:
            logger.warning(f"Metric '{name}' is being re-registered")
        _METRICS[name] = cls
        # Also set the name attribute on the class
        cls.name = name
        logger.debug(f"Registered metric: {name}")
        return cls

    return decorator


def get_metric(name: str) -> BaseMetric:
    """
    Get an instance of a registered metric by name.

    Args:
        name: The metric identifier

    Returns:
        An instance of the metric class

    Raises:
        KeyError: If the metric is not registered
    """
    if name not in _METRICS:
        available = ", ".join(_METRICS.keys())
        raise KeyError(f"Metric '{name}' not found. Available: {available}")
    return _METRICS[name]()


def list_metrics() -> list[str]:
    """
    List all registered metric names.

    Returns:
        List of metric identifiers
    """
    return list(_METRICS.keys())


def get_metric_info(name: str) -> dict[str, Any]:
    """
    Get information about a registered metric.

    Args:
        name: The metric identifier

    Returns:
        Dictionary with metric metadata
    """
    if name not in _METRICS:
        raise KeyError(f"Metric '{name}' not found")

    cls = _METRICS[name]
    return {
        "name": name,
        "description": getattr(cls, "description", "No description"),
        "requires_context": getattr(cls, "requires_context", False),
        "requires_reference": getattr(cls, "requires_reference", False),
        "version": getattr(cls, "version", "1.0.0"),
    }


def get_metric_versions() -> dict[str, str]:
    """
    Get versions of all registered metrics.

    Returns:
        Dictionary mapping metric names to their versions
    """
    return {
        name: getattr(cls, "version", "1.0.0")
        for name, cls in _METRICS.items()
    }


def get_all_metric_info() -> list[dict[str, Any]]:
    """
    Get information about all registered metrics.

    Returns:
        List of metric info dictionaries
    """
    return [get_metric_info(name) for name in _METRICS.keys()]


async def compute_metric(
    name: str,
    question: str,
    generated_answer: str,
    reference_answer: Optional[str] = None,
    retrieved_chunks: Optional[list[str]] = None,
    **kwargs: Any,
) -> MetricResult:
    """
    Convenience function to compute a metric by name.

    Args:
        name: The metric identifier
        question: The original question
        generated_answer: The RAG-generated answer
        reference_answer: The ground truth answer (optional)
        retrieved_chunks: Retrieved context chunks (optional)
        **kwargs: Additional metric-specific parameters

    Returns:
        MetricResult with score and details
    """
    metric = get_metric(name)
    metric.validate_inputs(question, generated_answer, reference_answer, retrieved_chunks)
    return await metric.compute(
        question=question,
        generated_answer=generated_answer,
        reference_answer=reference_answer,
        retrieved_chunks=retrieved_chunks,
        **kwargs,
    )
