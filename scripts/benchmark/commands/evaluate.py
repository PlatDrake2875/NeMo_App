"""Evaluate command - run evaluation on Q&A dataset."""

import logging
from collections.abc import Callable
from pathlib import Path

from scripts.benchmark.client import BenchmarkAPIClient
from scripts.benchmark.config import (
    ChunkingConfig,
    EvaluateResult,
    EvalMetrics,
)
from scripts.benchmark.utils.polling import poll_task

logger = logging.getLogger(__name__)


async def evaluate(
    client: BenchmarkAPIClient,
    eval_dataset_id: str,
    processed_dataset_id: int,
    preset: str | None = None,
    chunking: ChunkingConfig | None = None,
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_rag: bool = True,
    use_colbert: bool = False,
    top_k: int = 5,
    temperature: float = 0.1,
    experiment_name: str | None = None,
    output_path: Path | None = None,
    timeout: float = 3600.0,
    progress_callback: Callable[[dict], None] | None = None,
) -> EvaluateResult:
    """
    Run evaluation on a Q&A dataset using a processed dataset.

    This is Stage 3 of the benchmark pipeline. It:
    1. Starts an evaluation task
    2. Polls until completion
    3. Retrieves results and metrics
    4. Optionally exports to CSV

    Args:
        client: API client instance
        eval_dataset_id: Evaluation dataset ID (Q&A pairs)
        processed_dataset_id: Processed dataset ID (document chunks)
        preset: Evaluation preset (quick|balanced|high_quality)
        chunking: Custom chunking config (overrides preset)
        embedder: Embedder model name
        use_rag: Whether to use RAG
        use_colbert: Whether to use ColBERT reranking
        top_k: Number of chunks to retrieve
        temperature: Generation temperature
        experiment_name: Optional experiment name
        output_path: Optional CSV output path
        timeout: Maximum time to wait for evaluation
        progress_callback: Optional callback for progress updates

    Returns:
        EvaluateResult with run_id and metrics
    """
    logger.info(
        f"Starting evaluation: eval_dataset={eval_dataset_id}, "
        f"processed_dataset={processed_dataset_id}"
    )

    if preset:
        logger.info(f"Using preset: {preset}")
    elif chunking:
        logger.info(
            f"Using custom chunking: method={chunking.method}, "
            f"size={chunking.chunk_size}, overlap={chunking.chunk_overlap}"
        )

    # Step 1: Start evaluation task
    task_id = await client.start_eval_task(
        eval_dataset_id=eval_dataset_id,
        preprocessed_dataset_id=processed_dataset_id,
        preset=preset,
        chunking=chunking,
        embedder=embedder,
        use_rag=use_rag,
        use_colbert=use_colbert,
        top_k=top_k,
        temperature=temperature,
        experiment_name=experiment_name,
    )
    logger.info(f"Started evaluation task: {task_id}")

    # Step 2: Poll until complete
    def task_callback(status: dict) -> None:
        task_status = status.get("status", "unknown")
        progress = status.get("progress", {})
        if progress_callback:
            progress_callback(status)
        logger.debug(f"Task status: {task_status}, progress: {progress}")

    task = await poll_task(
        client.get_eval_task,
        task_id,
        timeout=timeout,
        interval=2.0,
        progress_callback=task_callback,
    )

    # Step 3: Get run results
    run_id = task.get("result_run_id")
    if not run_id:
        raise RuntimeError(f"Task completed but no run_id found: {task}")

    logger.info(f"Evaluation completed, run ID: {run_id}")

    run = await client.get_eval_run(run_id)
    metrics_data = run.get("metrics", {})
    metrics = EvalMetrics(
        context_precision=metrics_data.get("context_precision", 0.0),
        precision_at_k=metrics_data.get("precision_at_k", 0.0),
        recall_at_k=metrics_data.get("recall_at_k", 0.0),
        avg_latency=metrics_data.get("avg_latency", 0.0),
        answer_relevancy=metrics_data.get("answer_relevancy"),
        faithfulness=metrics_data.get("faithfulness"),
        answer_correctness=metrics_data.get("answer_correctness"),
    )

    logger.info(
        f"Metrics: context_precision={metrics.context_precision:.4f}, "
        f"precision@k={metrics.precision_at_k:.4f}, "
        f"recall@k={metrics.recall_at_k:.4f}, "
        f"avg_latency={metrics.avg_latency:.4f}s"
    )

    # Step 4: Export CSV if requested
    if output_path:
        await client.export_run_csv(run_id, output_path)
        logger.info(f"Exported results to {output_path}")

    return EvaluateResult(
        run_id=run_id,
        metrics=metrics,
        result_count=len(run.get("results", [])),
    )
