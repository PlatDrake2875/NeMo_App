"""
Base classes for RAG evaluation metrics.

All metrics should inherit from BaseMetric and implement the compute method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MetricResult:
    """Result of a metric computation."""

    score: float
    details: dict[str, Any] = field(default_factory=dict)
    used_fallback: bool = False  # Whether heuristic fallback was used
    metric_version: Optional[str] = None  # Version of the metric that computed this

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "score": self.score,
            "details": self.details,
        }
        if self.used_fallback:
            result["used_fallback"] = self.used_fallback
        if self.metric_version:
            result["metric_version"] = self.metric_version
        return result


class BaseMetric(ABC):
    """
    Abstract base class for all RAG evaluation metrics.

    Each metric must implement:
    - name: Unique identifier for the metric
    - description: Human-readable description
    - compute(): Async method to calculate the metric

    Metrics can optionally specify:
    - requires_context: Whether retrieved chunks are needed
    - requires_reference: Whether ground truth answer is needed
    - version: Semantic version string for reproducibility
    """

    name: str = "base"
    description: str = "Base metric"
    requires_context: bool = False
    requires_reference: bool = False
    version: str = "2.0.0"  # Metric version for reproducibility tracking

    @abstractmethod
    async def compute(
        self,
        question: str,
        generated_answer: str,
        reference_answer: Optional[str] = None,
        retrieved_chunks: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute the metric score.

        Args:
            question: The original question
            generated_answer: The RAG-generated answer
            reference_answer: The ground truth/expected answer (if available)
            retrieved_chunks: List of retrieved context chunks (if available)
            **kwargs: Additional metric-specific parameters

        Returns:
            MetricResult with score (0-1) and optional details
        """
        pass

    def validate_inputs(
        self,
        question: str,
        generated_answer: str,
        reference_answer: Optional[str] = None,
        retrieved_chunks: Optional[list[str]] = None,
    ) -> None:
        """
        Validate inputs before computation.

        Raises:
            ValueError: If required inputs are missing
        """
        if not question:
            raise ValueError(f"Metric '{self.name}' requires a question")
        if not generated_answer:
            raise ValueError(f"Metric '{self.name}' requires a generated answer")
        if self.requires_reference and not reference_answer:
            raise ValueError(f"Metric '{self.name}' requires a reference answer")
        if self.requires_context and not retrieved_chunks:
            raise ValueError(f"Metric '{self.name}' requires retrieved chunks")

    def __repr__(self) -> str:
        return f"<Metric: {self.name}>"
