"""
Evaluation Scorer - Computes metrics from inference results.

Separates scoring from inference, enabling:
- Re-scoring historical runs with new metrics
- Computing subset of metrics
- Adding custom metrics without re-running inference
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .eval_runner import EvalRunResult, InferenceResult
from .metrics import compute_metric, get_metric, list_metrics

logger = logging.getLogger(__name__)


@dataclass
class ScoredResult:
    """Result with computed metric scores."""

    query: str
    ground_truth: str
    predicted_answer: str
    retrieved_chunks: list[dict[str, Any]]
    latency: float
    scores: dict[str, float] = field(default_factory=dict)
    score_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "predicted_answer": self.predicted_answer,
            "retrieved_chunks": self.retrieved_chunks,
            "latency": self.latency,
            "scores": self.scores,
            "score_details": self.score_details,
            "error": self.error,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all results."""

    context_precision: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    avg_latency: float = 0.0
    total_pairs: int = 0
    successful_pairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "context_precision": self.context_precision,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "avg_latency": self.avg_latency,
            "total_pairs": self.total_pairs,
            "successful_pairs": self.successful_pairs,
        }


@dataclass
class ScoredRunResult:
    """Complete scored evaluation run."""

    run_id: str
    config: dict[str, Any]
    results: list[ScoredResult]
    metrics: AggregateMetrics
    created_at: str
    scored_at: str
    dataset_name: str = "Unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "run_id": self.run_id,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at,
            "scored_at": self.scored_at,
            "dataset_name": self.dataset_name,
        }


class EvalScorer:
    """
    Computes metrics from inference results.

    Responsibilities:
    - Calculate individual metric scores for each result
    - Aggregate metrics across all results
    - Support re-scoring with different/new metrics
    """

    def __init__(
        self,
        runs_dir: Path = Path("data/evaluation_runs"),
        default_metrics: Optional[list[str]] = None,
    ):
        self.runs_dir = runs_dir
        self.default_metrics = default_metrics or [
            "context_precision",
            "precision_at_k",
            "recall_at_k",
        ]

    async def score_result(
        self,
        result: InferenceResult,
        metrics: Optional[list[str]] = None,
    ) -> ScoredResult:
        """
        Compute metrics for a single inference result.

        Args:
            result: The inference result to score
            metrics: List of metric names to compute (defaults to all)

        Returns:
            ScoredResult with computed scores
        """
        metrics_to_compute = metrics or self.default_metrics
        scores = {}
        score_details = {}

        # Extract chunk contents for metrics that need them
        chunk_contents = [c.get("content", "") for c in result.retrieved_chunks]

        # Compute each metric
        for metric_name in metrics_to_compute:
            try:
                metric = get_metric(metric_name)

                # Skip metrics that require unavailable data
                if metric.requires_reference and not result.ground_truth:
                    continue
                if metric.requires_context and not chunk_contents:
                    continue

                metric_result = await compute_metric(
                    name=metric_name,
                    question=result.query,
                    generated_answer=result.predicted_answer,
                    reference_answer=result.ground_truth,
                    retrieved_chunks=chunk_contents if chunk_contents else None,
                )

                scores[metric_name] = metric_result.score
                score_details[metric_name] = metric_result.details

            except Exception as e:
                logger.warning(f"Failed to compute metric '{metric_name}': {e}")
                scores[metric_name] = 0.0
                score_details[metric_name] = {"error": str(e)}

        return ScoredResult(
            query=result.query,
            ground_truth=result.ground_truth,
            predicted_answer=result.predicted_answer,
            retrieved_chunks=result.retrieved_chunks,
            latency=result.latency,
            scores=scores,
            score_details=score_details,
            error=result.error,
        )

    async def score_run(
        self,
        run_result: EvalRunResult,
        metrics: Optional[list[str]] = None,
    ) -> ScoredRunResult:
        """
        Compute metrics for an entire evaluation run.

        Args:
            run_result: The evaluation run to score
            metrics: List of metric names to compute

        Returns:
            ScoredRunResult with all scores and aggregates
        """
        from datetime import datetime

        metrics_to_compute = metrics or self.default_metrics

        # Score all results
        scored_results = []
        for result in run_result.results:
            scored = await self.score_result(result, metrics_to_compute)
            scored_results.append(scored)

        # Compute aggregate metrics
        aggregate = self._compute_aggregates(scored_results)

        return ScoredRunResult(
            run_id=run_result.run_id,
            config=run_result.config.to_dict(),
            results=scored_results,
            metrics=aggregate,
            created_at=run_result.created_at,
            scored_at=datetime.now().isoformat(),
            dataset_name=run_result.dataset_name,
        )

    def _compute_aggregates(self, results: list[ScoredResult]) -> AggregateMetrics:
        """Compute aggregate metrics from scored results."""
        if not results:
            return AggregateMetrics()

        # Filter successful results
        successful = [r for r in results if r.error is None]

        if not successful:
            return AggregateMetrics(
                total_pairs=len(results),
                successful_pairs=0,
            )

        # Calculate averages
        def avg(metric_name: str) -> float:
            values = [r.scores.get(metric_name, 0.0) for r in successful]
            return sum(values) / len(values) if values else 0.0

        return AggregateMetrics(
            context_precision=avg("context_precision"),
            precision_at_k=avg("precision_at_k"),
            recall_at_k=avg("recall_at_k"),
            avg_latency=sum(r.latency for r in results) / len(results),
            total_pairs=len(results),
            successful_pairs=len(successful),
        )

    async def rescore_run(
        self,
        run_id: str,
        metrics: Optional[list[str]] = None,
    ) -> ScoredRunResult:
        """
        Re-score a historical run with new/different metrics.

        Args:
            run_id: ID of the run to re-score
            metrics: List of metric names to compute

        Returns:
            ScoredRunResult with new scores
        """
        # Load the run
        run_path = self.runs_dir / f"{run_id}.json"
        if not run_path.exists():
            raise ValueError(f"Run {run_id} not found")

        with open(run_path) as f:
            run_data = json.load(f)

        # Convert to InferenceResults
        inference_results = []
        for r in run_data.get("results", []):
            inference_results.append(
                InferenceResult(
                    query=r.get("query", ""),
                    ground_truth=r.get("ground_truth", ""),
                    predicted_answer=r.get("predicted_answer", ""),
                    retrieved_chunks=r.get("retrieved_chunks", []),
                    latency=r.get("latency", 0.0),
                    error=r.get("error"),
                )
            )

        # Create a mock EvalRunResult
        from .eval_runner import EvalConfig

        config = EvalConfig.from_dict(run_data.get("config", {}))
        run_result = EvalRunResult(
            run_id=run_id,
            config=config,
            results=inference_results,
            created_at=run_data.get("created_at", ""),
            dataset_name=run_data.get("name", run_data.get("dataset_name", "Unknown")),
        )

        return await self.score_run(run_result, metrics)

    def save_scored_run(self, scored_run: ScoredRunResult) -> Path:
        """
        Save a scored run to disk.

        Args:
            scored_run: The scored run to save

        Returns:
            Path to the saved file
        """
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        run_path = self.runs_dir / f"{scored_run.run_id}.json"

        # Format for backward compatibility with existing UI
        save_data = {
            "id": scored_run.run_id,
            "name": scored_run.dataset_name,
            "created_at": scored_run.created_at,
            "scored_at": scored_run.scored_at,
            "config": scored_run.config,
            "metrics": scored_run.metrics.to_dict(),
            "results": [r.to_dict() for r in scored_run.results],
        }

        with open(run_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        logger.info(f"Saved scored run to {run_path}")
        return run_path


def get_available_metrics() -> list[dict[str, Any]]:
    """
    Get information about all available metrics.

    Returns:
        List of metric info dictionaries
    """
    from .metrics.registry import get_all_metric_info

    return get_all_metric_info()
