"""Output handling utilities for JSON and CSV export."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.benchmark.config import (
    EvaluateResult,
    EvalMetrics,
    GenerateQAResult,
    PipelineResult,
    PreprocessResult,
)

logger = logging.getLogger(__name__)


def export_json(
    data: dict | PipelineResult | EvaluateResult,
    output_path: Path,
    *,
    indent: int = 2,
) -> Path:
    """
    Export data to JSON file.

    Args:
        data: Data to export (dict or Pydantic model)
        output_path: Output file path
        indent: JSON indentation level

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")

    # Convert Path objects to strings
    data = _serialize_paths(data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=str)

    logger.info(f"Exported JSON to {output_path}")
    return output_path


def _serialize_paths(obj: Any) -> Any:
    """Recursively convert Path objects to strings."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_paths(item) for item in obj]
    return obj


def export_csv(
    results: list[dict],
    output_path: Path,
    *,
    metrics: EvalMetrics | None = None,
    include_summary: bool = True,
) -> Path:
    """
    Export evaluation results to CSV.

    Args:
        results: List of evaluation result dicts
        output_path: Output file path
        metrics: Optional aggregate metrics for summary row
        include_summary: Whether to include summary row

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to export")
        return output_path

    # Standard columns
    fieldnames = [
        "Query",
        "Predicted Answer",
        "Ground Truth",
        "Jaccard (%)",
        "Context Precision (%)",
        "Precision@K (%)",
        "Recall@K (%)",
        "Latency (s)",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            scores = result.get("scores", {}) or {}
            row = {
                "Query": result.get("query", ""),
                "Predicted Answer": result.get("predicted_answer", ""),
                "Ground Truth": result.get("ground_truth", ""),
                "Jaccard (%)": _format_percent(result.get("score")),
                "Context Precision (%)": _format_percent(scores.get("context_precision")),
                "Precision@K (%)": _format_percent(scores.get("precision_at_k")),
                "Recall@K (%)": _format_percent(scores.get("recall_at_k")),
                "Latency (s)": _format_float(result.get("latency")),
            }
            writer.writerow(row)

        if include_summary and metrics:
            # Write empty row and summary
            writer.writerow({})
            summary_row = {
                "Query": "SUMMARY",
                "Predicted Answer": "",
                "Ground Truth": "",
                "Jaccard (%)": "",
                "Context Precision (%)": _format_percent(metrics.context_precision),
                "Precision@K (%)": _format_percent(metrics.precision_at_k),
                "Recall@K (%)": _format_percent(metrics.recall_at_k),
                "Latency (s)": _format_float(metrics.avg_latency),
            }
            writer.writerow(summary_row)

    logger.info(f"Exported CSV to {output_path}")
    return output_path


def _format_percent(value: float | None) -> str:
    """Format a 0-1 float as percentage string."""
    if value is None:
        return ""
    return f"{value * 100:.1f}"


def _format_float(value: float | None) -> str:
    """Format a float to 3 decimal places."""
    if value is None:
        return ""
    return f"{value:.3f}"


def create_result_summary(
    result: PipelineResult | EvaluateResult | PreprocessResult | GenerateQAResult,
) -> dict[str, Any]:
    """
    Create a summary dict suitable for console output.

    Args:
        result: Any result model

    Returns:
        Summary dict with key information
    """
    if isinstance(result, PipelineResult):
        return {
            "name": result.name,
            "success": result.success,
            "duration_seconds": round(result.duration_seconds, 2),
            "raw_dataset_id": result.preprocess.raw_dataset_id,
            "processed_dataset_id": result.preprocess.processed_dataset_id,
            "eval_dataset_id": result.qa_generation.eval_dataset_id,
            "eval_run_id": result.evaluation.run_id,
            "metrics": {
                "context_precision": round(result.evaluation.metrics.context_precision, 4),
                "precision_at_k": round(result.evaluation.metrics.precision_at_k, 4),
                "recall_at_k": round(result.evaluation.metrics.recall_at_k, 4),
                "avg_latency": round(result.evaluation.metrics.avg_latency, 4),
            },
            "error": result.error,
        }
    elif isinstance(result, EvaluateResult):
        return {
            "run_id": result.run_id,
            "result_count": result.result_count,
            "metrics": {
                "context_precision": round(result.metrics.context_precision, 4),
                "precision_at_k": round(result.metrics.precision_at_k, 4),
                "recall_at_k": round(result.metrics.recall_at_k, 4),
                "avg_latency": round(result.metrics.avg_latency, 4),
            },
        }
    elif isinstance(result, PreprocessResult):
        return {
            "raw_dataset_id": result.raw_dataset_id,
            "processed_dataset_id": result.processed_dataset_id,
            "document_count": result.document_count,
            "chunk_count": result.chunk_count,
        }
    elif isinstance(result, GenerateQAResult):
        return {
            "eval_dataset_id": result.eval_dataset_id,
            "name": result.name,
            "pair_count": result.pair_count,
            "chunks_processed": result.chunks_processed,
        }
    else:
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)


def generate_output_path(
    base_dir: Path,
    prefix: str,
    extension: str = "json",
) -> Path:
    """
    Generate a timestamped output path.

    Args:
        base_dir: Base directory for output
        prefix: File name prefix
        extension: File extension (without dot)

    Returns:
        Path with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{prefix}_{timestamp}.{extension}"
