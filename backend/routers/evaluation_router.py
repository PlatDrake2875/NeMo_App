"""
Evaluation router - API endpoints for RAG evaluation.

Provides endpoints for:
- Creating and managing evaluation datasets (Q&A pairs)
- Running evaluations with configurable parameters
- Retrieving evaluation results and metrics
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import csv
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas.evaluation import (
    AnswerCorrectnessDetail,
    ChunkingConfig,
    ClaimVerificationDetail,
    CreateEvalDatasetRequest,
    EvalConfig,
    EvalDatasetResponse,
    EvalMetrics,
    EvalResult,
    EvalResultScores,
    EXPERIMENT_PRESETS,
    FaithfulnessDetail,
    GeneratedPair,
    GenerateQARequest,
    GenerateQAResponse,
    GenerateSampleRequest,
    GenerateSampleResponse,
    RetrievedChunk,
    RunEvaluationRequest,
    RunEvaluationResponse,
)
from services.qa_generator import QAGeneratorService
from services.embedding_service import embedding_service
from database_models import get_db_session
from services.evaluation_metrics import RAGEvaluationMetrics
from rag_components import get_rag_context_prefix, format_docs
from config import VLLM_BASE_URL, VLLM_MODEL
import httpx

logger = logging.getLogger(__name__)

# Directory for storing evaluation datasets
EVAL_DATASETS_DIR = Path("data/evaluation_datasets")
EVAL_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Directory for storing evaluation runs (results)
EVAL_RUNS_DIR = Path("data/evaluation_runs")
EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(
    prefix="/api/evaluation",
    tags=["evaluation"],
)


def _get_dataset_path(dataset_id: str) -> Path:
    """Get the file path for a dataset."""
    return EVAL_DATASETS_DIR / f"{dataset_id}.json"


def _load_dataset(dataset_id: str) -> dict:
    """Load a dataset from disk."""
    path = _get_dataset_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    with open(path) as f:
        return json.load(f)


def _save_dataset(dataset_id: str, data: dict):
    """Save a dataset to disk."""
    path = _get_dataset_path(dataset_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@router.get("/datasets", response_model=list[EvalDatasetResponse])
async def list_eval_datasets():
    """List all evaluation datasets."""
    datasets = []
    for path in EVAL_DATASETS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                datasets.append(
                    EvalDatasetResponse(
                        id=data["id"],
                        name=data["name"],
                        pair_count=len(data["pairs"]),
                        created_at=data["created_at"],
                        source_collection=data.get("source_collection"),  # May be None for old datasets
                    )
                )
        except Exception as e:
            logger.error(f"Error loading dataset {path}: {e}")
    return datasets


@router.post("/datasets", response_model=EvalDatasetResponse)
async def create_eval_dataset(request: CreateEvalDatasetRequest):
    """Create a new evaluation dataset with Q&A pairs."""
    dataset_id = str(uuid.uuid4())[:8]
    created_at = datetime.now().isoformat()

    data = {
        "id": dataset_id,
        "name": request.name,
        "pairs": [p.model_dump() for p in request.pairs],
        "created_at": created_at,
    }

    _save_dataset(dataset_id, data)
    logger.info(f"Created evaluation dataset '{request.name}' with {len(request.pairs)} pairs")

    return EvalDatasetResponse(
        id=dataset_id,
        name=request.name,
        pair_count=len(request.pairs),
        created_at=created_at,
    )


@router.get("/datasets/{dataset_id}")
async def get_eval_dataset(dataset_id: str):
    """Get a specific evaluation dataset with all Q&A pairs."""
    return _load_dataset(dataset_id)


@router.delete("/datasets/{dataset_id}")
async def delete_eval_dataset(dataset_id: str):
    """Delete an evaluation dataset."""
    path = _get_dataset_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    path.unlink()
    return {"status": "deleted", "id": dataset_id}


@router.post("/run", response_model=RunEvaluationResponse)
async def run_evaluation(request: RunEvaluationRequest):
    """
    Run an evaluation using the specified dataset and configuration.

    Supports two modes:
    1. Legacy mode: Use existing collection_name (already chunked/embedded)
    2. New mode: Use preprocessed_dataset_id + chunking/embedder config (on-demand processing)

    Uses RAGAS-style metrics:
    - Answer Correctness: F1 factual + semantic similarity
    - Faithfulness: LLM-verified claim support from context
    - Response Relevancy: Embedding-based query-answer similarity
    - Context Precision: Mean precision@k for retrieval ranking

    This will:
    1. Load the evaluation dataset (or use test query if no dataset)
    2. For each Q&A pair:
       - Run the RAG pipeline with the query
       - Record the predicted answer, retrieved chunks, and latency
       - Calculate RAGAS-style metrics
    3. Compute aggregate metrics
    """
    # Determine effective configuration (apply preset if specified)
    effective_chunking = request.chunking
    effective_embedder = request.embedder
    effective_top_k = request.top_k
    effective_colbert = request.use_colbert
    effective_temperature = request.temperature
    preset_name = None

    if request.preset and request.preset in EXPERIMENT_PRESETS:
        preset = EXPERIMENT_PRESETS[request.preset]
        preset_name = preset.name
        # Use preset values unless explicitly overridden
        if not request.chunking:
            effective_chunking = preset.chunking
        if request.embedder == "sentence-transformers/all-MiniLM-L6-v2":  # Default value
            effective_embedder = preset.embedder
        if request.top_k == 5:  # Default value
            effective_top_k = preset.top_k
        if not request.use_colbert:  # Default is False
            effective_colbert = preset.reranker == "colbert"
        if request.temperature == 0.1:  # Default value
            effective_temperature = preset.temperature

        logger.info(f"Using preset '{preset_name}': {preset.description}")

    # Determine collection name
    collection_name = request.collection_name
    was_cached = False

    # New mode: Use preprocessed_dataset_id with on-demand chunking/embedding
    # Only use new flow if dataset actually has preprocessed documents
    if request.preprocessed_dataset_id and effective_chunking:
        from database_models import PreprocessedDocument, ProcessedDataset
        from sqlalchemy import func

        # Get database session
        db_gen = get_db_session()
        db = next(db_gen)

        try:
            # Check if dataset has preprocessed documents (new flow)
            preprocessed_count = db.query(func.count(PreprocessedDocument.id)).filter(
                PreprocessedDocument.processed_dataset_id == request.preprocessed_dataset_id
            ).scalar()

            if preprocessed_count > 0:
                # New flow: dataset has preprocessed documents, use on-demand embedding
                logger.info(
                    f"New flow: Creating/getting collection for dataset {request.preprocessed_dataset_id} "
                    f"with chunking {effective_chunking.method}/{effective_chunking.chunk_size} "
                    f"and embedder {effective_embedder}"
                )

                # Get or create collection using EmbeddingService
                collection_name, was_cached = await embedding_service.get_or_create_collection(
                    db=db,
                    preprocessed_dataset_id=request.preprocessed_dataset_id,
                    chunking_config=effective_chunking,
                    embedder_model=effective_embedder,
                )

                logger.info(
                    f"Using collection '{collection_name}' (cached={was_cached})"
                )
            else:
                # Legacy flow: dataset was processed with direct indexing
                # Get collection name from ProcessedDataset table
                processed_ds = db.query(ProcessedDataset).filter(
                    ProcessedDataset.id == request.preprocessed_dataset_id
                ).first()

                if processed_ds and processed_ds.collection_name:
                    collection_name = processed_ds.collection_name
                    logger.info(
                        f"Dataset {request.preprocessed_dataset_id} has no preprocessed documents, "
                        f"using legacy collection '{collection_name}'"
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Dataset {request.preprocessed_dataset_id} has no preprocessed documents and no collection. "
                               "Please re-process the dataset."
                    )
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass

    # Get pairs from dataset or use test query
    if request.eval_dataset_id:
        dataset = _load_dataset(request.eval_dataset_id)
        pairs = dataset["pairs"]
        dataset_name = dataset["name"]
    else:
        # Use test query or default
        test_query = request.test_query or "What information is available in the documents?"
        pairs = [{"query": test_query, "ground_truth": "Based on retrieved documents."}]
        dataset_name = "Quick Test"

    logger.info(
        f"Starting evaluation: dataset='{dataset_name}', "
        f"pairs={len(pairs)}, collection={collection_name}, "
        f"rag={request.use_rag}, colbert={effective_colbert}"
    )

    # Initialize RAGAS-style metrics evaluator
    metrics_evaluator = RAGEvaluationMetrics()

    results = []
    total_relevancy = 0.0
    total_faithfulness = 0.0
    total_correctness = 0.0
    total_context_precision = 0.0
    total_latency = 0.0

    for i, pair in enumerate(pairs):
        query = pair["query"]
        ground_truth = pair["ground_truth"]

        logger.info(f"Evaluating pair {i+1}/{len(pairs)}: {query[:50]}...")

        start_time = time.time()

        try:
            # Get RAG context and generate answer
            if request.use_rag:
                # Get RAG context prefix
                rag_prefix = await get_rag_context_prefix(
                    query,
                    collection_name=collection_name,
                    use_colbert=effective_colbert,
                    embedder=effective_embedder,
                )
                prompt_content = rag_prefix if rag_prefix else query
            else:
                prompt_content = query

            # Call vLLM to generate answer
            predicted_answer, retrieved_chunks = await _generate_answer(
                prompt_content,
                query,
                effective_temperature,
                request.use_rag,
            )

            generation_latency = time.time() - start_time

            # Extract chunk contents for metrics
            chunk_contents = [c.content for c in retrieved_chunks] if retrieved_chunks else []

            # Calculate Jaccard similarity (original metric - kept for reference)
            jaccard_score = _calculate_jaccard(predicted_answer, ground_truth)

            # Calculate RAGAS-style metrics
            logger.info(f"  Computing metrics for pair {i+1}...")

            # Answer Correctness (F1 + semantic)
            correctness_result = await metrics_evaluator.answer_correctness(
                predicted_answer, ground_truth
            )

            # Faithfulness (LLM claim verification)
            faithfulness_result = await metrics_evaluator.faithfulness(
                predicted_answer, chunk_contents
            )

            # Response Relevancy (embedding similarity)
            relevancy = metrics_evaluator.response_relevancy(query, predicted_answer)

            # Context Precision (retrieval ranking quality)
            precision_result = await metrics_evaluator.context_precision(
                query, chunk_contents, ground_truth
            )

            total_latency += generation_latency
            total_relevancy += relevancy
            total_faithfulness += faithfulness_result.score
            total_correctness += correctness_result.score
            total_context_precision += precision_result.score

            # Build detailed score breakdown
            faithfulness_detail = FaithfulnessDetail(
                total_claims=faithfulness_result.total_claims,
                supported_claims=faithfulness_result.supported_claims,
                claims=[
                    ClaimVerificationDetail(
                        claim=c.claim,
                        supported=c.supported,
                        evidence=c.evidence,
                    )
                    for c in faithfulness_result.claim_details
                ],
            )

            correctness_detail = AnswerCorrectnessDetail(
                factual_score=correctness_result.factual_score,
                semantic_score=correctness_result.semantic_score,
                true_positives=correctness_result.true_positives,
                false_positives=correctness_result.false_positives,
                false_negatives=correctness_result.false_negatives,
            )

            results.append(
                EvalResult(
                    query=query,
                    ground_truth=ground_truth,
                    predicted_answer=predicted_answer,
                    retrieved_chunks=retrieved_chunks,
                    score=jaccard_score,  # Keep Jaccard as main score for backwards compat
                    scores=EvalResultScores(
                        jaccard=jaccard_score,
                        relevancy=relevancy,
                        faithfulness=faithfulness_result.score,
                        answer_correctness=correctness_result.score,
                        context_precision=precision_result.score,
                        faithfulness_detail=faithfulness_detail,
                        answer_correctness_detail=correctness_detail,
                    ),
                    latency=generation_latency,
                )
            )

        except Exception as e:
            logger.error(f"Error evaluating pair {i+1}: {e}", exc_info=True)
            results.append(
                EvalResult(
                    query=query,
                    ground_truth=ground_truth,
                    predicted_answer=f"Error: {str(e)}",
                    retrieved_chunks=None,
                    score=0.0,
                    scores=None,
                    latency=time.time() - start_time,
                )
            )

    # Calculate aggregate metrics
    n = len([r for r in results if r.scores is not None])  # Only count successful evaluations
    metrics = EvalMetrics(
        answer_relevancy=total_relevancy / n if n > 0 else 0.0,
        faithfulness=total_faithfulness / n if n > 0 else 0.0,
        context_precision=total_context_precision / n if n > 0 else 0.0,
        answer_correctness=total_correctness / n if n > 0 else 0.0,
        avg_latency=total_latency / len(results) if results else 0.0,
    )

    logger.info(
        f"Evaluation complete: "
        f"answer_correctness={metrics.answer_correctness:.2f}, "
        f"faithfulness={metrics.faithfulness:.2f}, "
        f"relevancy={metrics.answer_relevancy:.2f}, "
        f"context_precision={metrics.context_precision:.2f}, "
        f"avg_latency={metrics.avg_latency:.2f}s"
    )

    # Build response
    response = RunEvaluationResponse(
        eval_dataset_id=request.eval_dataset_id,
        results=results,
        metrics=metrics,
        config=EvalConfig(
            collection=collection_name,
            use_rag=request.use_rag,
            use_colbert=effective_colbert,
            top_k=effective_top_k,
            temperature=effective_temperature,
            embedder=effective_embedder,
            llm_model=VLLM_MODEL,
            preprocessed_dataset_id=request.preprocessed_dataset_id,
            chunking=effective_chunking,
            preset_name=preset_name,
        ),
    )

    # Save evaluation run to disk for persistence
    run_id = str(uuid.uuid4())[:8]
    run_data = {
        "id": run_id,
        "name": dataset_name,
        "created_at": datetime.now().isoformat(),
        "eval_dataset_id": request.eval_dataset_id,
        "config": response.config.model_dump(),
        "metrics": response.metrics.model_dump(),
        "results": [r.model_dump() for r in response.results],
    }
    run_path = EVAL_RUNS_DIR / f"{run_id}.json"
    with open(run_path, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    logger.info(f"Saved evaluation run to {run_path}")

    return response


async def _generate_answer(
    prompt_content: str,
    original_query: str,
    temperature: float,
    use_rag: bool,
) -> tuple[str, Optional[list[RetrievedChunk]]]:
    """Generate an answer using vLLM."""
    messages = [{"role": "user", "content": prompt_content}]

    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            raise Exception(f"vLLM error: {response.status_code}")

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        # Extract retrieved chunks from the prompt if RAG was used
        retrieved_chunks = None
        if use_rag and "Context:" in prompt_content:
            # Parse context from the prompt
            try:
                context_start = prompt_content.find("Context:") + len("Context:")
                question_start = prompt_content.find("Question:")
                if question_start > context_start:
                    context_text = prompt_content[context_start:question_start].strip()
                    # Split into chunks (simplified - assumes double newline separation)
                    chunks = [c.strip() for c in context_text.split("\n\n") if c.strip()]
                    retrieved_chunks = [
                        RetrievedChunk(content=chunk[:500], source="retrieved")
                        for chunk in chunks[:5]
                    ]
            except Exception:
                pass

        return answer, retrieved_chunks


def _calculate_jaccard(predicted: str, ground_truth: str) -> float:
    """
    Calculate Jaccard similarity between predicted answer and ground truth.
    Simple word overlap metric (kept for backwards compatibility/reference).
    """
    pred_words = set(predicted.lower().split())
    truth_words = set(ground_truth.lower().split())

    if not pred_words or not truth_words:
        return 0.0

    intersection = pred_words & truth_words
    union = pred_words | truth_words

    return len(intersection) / len(union) if union else 0.0


# --- Evaluation Runs (History & Export) ---

@router.get("/runs")
async def list_evaluation_runs():
    """List all past evaluation runs."""
    runs = []
    for path in sorted(EVAL_RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
                config = data.get("config", {})
                # Backfill llm_model for existing runs that don't have it
                if "llm_model" not in config:
                    config["llm_model"] = VLLM_MODEL
                runs.append({
                    "id": data["id"],
                    "name": data.get("name", "Unknown"),
                    "created_at": data["created_at"],
                    "pair_count": len(data.get("results", [])),
                    "metrics": data.get("metrics", {}),
                    "config": config,
                })
        except Exception as e:
            logger.error(f"Error loading run {path}: {e}")
    return runs


@router.get("/runs/{run_id}")
async def get_evaluation_run(run_id: str):
    """Get a specific evaluation run with all results."""
    path = EVAL_RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    with open(path) as f:
        data = json.load(f)
    # Backfill llm_model for existing runs that don't have it
    if "config" in data and "llm_model" not in data["config"]:
        data["config"]["llm_model"] = VLLM_MODEL
    return data


@router.delete("/runs/{run_id}")
async def delete_evaluation_run(run_id: str):
    """Delete an evaluation run."""
    path = EVAL_RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    path.unlink()
    return {"status": "deleted", "id": run_id}


@router.get("/runs/{run_id}/csv")
async def export_evaluation_run_csv(run_id: str):
    """Export an evaluation run as CSV."""
    path = EVAL_RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    with open(path) as f:
        data = json.load(f)

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "Query",
        "Predicted Answer",
        "Ground Truth",
        "Jaccard",
        "Answer Correctness",
        "Faithfulness",
        "Context Precision",
        "Relevancy",
        "Latency (s)",
    ])

    # Data rows
    for result in data.get("results", []):
        scores = result.get("scores") or {}
        writer.writerow([
            result.get("query", ""),
            result.get("predicted_answer", ""),
            result.get("ground_truth", ""),
            f"{(scores.get('jaccard') or result.get('score', 0)) * 100:.1f}%",
            f"{(scores.get('answer_correctness') or 0) * 100:.1f}%",
            f"{(scores.get('faithfulness') or 0) * 100:.1f}%",
            f"{(scores.get('context_precision') or 0) * 100:.1f}%",
            f"{(scores.get('relevancy') or 0) * 100:.1f}%",
            f"{result.get('latency', 0):.2f}",
        ])

    # Add summary row
    metrics = data.get("metrics", {})
    writer.writerow([])
    writer.writerow(["SUMMARY METRICS"])
    writer.writerow(["Answer Correctness", f"{metrics.get('answer_correctness', 0) * 100:.1f}%"])
    writer.writerow(["Faithfulness", f"{metrics.get('faithfulness', 0) * 100:.1f}%"])
    writer.writerow(["Context Precision", f"{metrics.get('context_precision', 0) * 100:.1f}%"])
    writer.writerow(["Answer Relevancy", f"{metrics.get('answer_relevancy', 0) * 100:.1f}%"])
    writer.writerow(["Avg Latency", f"{metrics.get('avg_latency', 0):.2f}s"])

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=evaluation_{run_id}.csv"}
    )


# --- Q&A Generation Endpoints ---

@router.post("/datasets/generate", response_model=GenerateQAResponse)
async def generate_qa_dataset(request: GenerateQARequest):
    """
    Generate Q&A pairs from a processed dataset using LLM.

    Supports two LLM providers:
    - OpenRouter (default): Cloud-based, uses GPT-4o-mini or other models
    - Local vLLM: Uses your local vLLM instance

    This will:
    1. Retrieve chunks from the processed dataset
    2. Use the selected LLM to generate Q&A pairs from each chunk
    3. Save the generated pairs as an evaluation dataset
    """
    try:
        generator = QAGeneratorService(
            model=request.model,
            use_vllm=request.use_vllm,
            temperature=request.temperature,
        )

        result = await generator.generate_pairs_for_dataset(
            processed_dataset_id=request.processed_dataset_id,
            name=request.name,
            pairs_per_chunk=request.pairs_per_chunk,
            max_chunks=request.max_chunks,
            seed=request.seed,
        )

        return GenerateQAResponse(
            id=result["id"],
            name=result["name"],
            pair_count=result["pair_count"],
            chunks_processed=result["chunks_processed"],
            created_at=result["created_at"],
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating Q&A dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/datasets/generate/sample", response_model=GenerateSampleResponse)
async def generate_sample_pairs(request: GenerateSampleRequest):
    """
    Generate a small sample of Q&A pairs for preview.

    Use this to preview what generated Q&A pairs will look like
    before running full generation.
    """
    try:
        generator = QAGeneratorService()

        pairs = await generator.generate_sample_pairs(
            processed_dataset_id=request.processed_dataset_id,
            sample_size=request.sample_size,
        )

        return GenerateSampleResponse(
            pairs=[GeneratedPair(**pair) for pair in pairs],
            chunks_sampled=request.sample_size,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating sample pairs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sample generation failed: {str(e)}")


# --- New Batch, Dry-Run, and Import Endpoints ---

@router.post("/batch")
async def run_batch_evaluation(configs: list[dict], dataset_id: Optional[str] = None):
    """
    Run batch evaluations with multiple configurations.

    Useful for comparing different embedder/reranker combinations.
    """
    from services.eval_runner import BatchEvalRunner, EvalConfig

    try:
        eval_configs = [EvalConfig.from_dict(c) for c in configs]

        batch_runner = BatchEvalRunner()
        results = await batch_runner.run_batch(eval_configs, dataset_id, parallel=True)

        return {
            "batch_id": str(uuid.uuid4())[:8],
            "run_ids": [r.run_id for r in results],
            "config_hashes": [r.config.config_hash() for r in results],
        }
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@router.post("/dry-run")
async def dry_run_evaluation(
    collection_name: str,
    eval_dataset_id: Optional[str] = None,
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_colbert: bool = False,
    top_k: int = 5,
    temperature: float = 0.1,
    use_rag: bool = True,
):
    """
    Validate evaluation configuration without running inference.

    Returns:
    - config_hash: Unique hash for this configuration
    - would_create_new_run: Whether this config has been run before
    - validation_errors: Any configuration issues
    """
    from services.eval_runner import EvalConfig, EvalRunner

    config = EvalConfig(
        embedder_model=embedder,
        reranker_strategy="colbert" if use_colbert else "none",
        top_k=top_k,
        temperature=temperature,
        dataset_id=eval_dataset_id,
        collection_name=collection_name,
        use_rag=use_rag,
    )

    runner = EvalRunner()
    result = runner.dry_run(config, eval_dataset_id)

    return {
        "config_hash": result["config_hash"],
        "would_create_new_run": "existing_run_id" not in result,
        "existing_run_id": result.get("existing_run_id"),
        "valid": result["valid"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "dataset_name": result.get("dataset_name"),
        "pair_count": result.get("pair_count"),
    }


@router.get("/metrics")
async def list_available_metrics():
    """
    List all available evaluation metrics.

    Returns metadata about each metric including:
    - name: Metric identifier
    - description: What the metric measures
    - requires_context: Whether it needs retrieved chunks
    - requires_reference: Whether it needs ground truth
    """
    from services.metrics.registry import get_all_metric_info

    return get_all_metric_info()


@router.post("/runs/{run_id}/rescore")
async def rescore_evaluation_run(run_id: str, metrics: Optional[list[str]] = None):
    """
    Re-score an evaluation run with new or different metrics.

    Useful when new metrics are added or you want to recompute
    specific metrics without re-running inference.
    """
    from services.eval_scorer import EvalScorer

    try:
        scorer = EvalScorer()
        scored_result = await scorer.rescore_run(run_id, metrics)
        scorer.save_scored_run(scored_result)

        return {
            "run_id": scored_result.run_id,
            "metrics": scored_result.metrics.to_dict(),
            "scored_at": scored_result.scored_at,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Re-scoring failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Re-scoring failed: {str(e)}")


@router.get("/datasets/import/formats")
async def list_import_formats():
    """List available dataset import formats."""
    from services.dataset_importers import list_importers

    return list_importers()


@router.post("/datasets/import")
async def import_dataset(
    format: str,
    name: str,
    file: bytes,
    max_pairs: Optional[int] = None,
):
    """
    Import an evaluation dataset from a standard format.

    Supported formats:
    - squad: Stanford Question Answering Dataset
    - natural_questions: Google Natural Questions
    - msmarco: Microsoft MARCO

    The file should be uploaded as raw bytes.
    """
    from services.dataset_importers import get_importer
    from services.dataset_validator import validate_qa_pairs
    from io import BytesIO

    try:
        importer = get_importer(format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = importer.import_from_stream(BytesIO(file), max_pairs)

        if not result.success or not result.pairs:
            raise HTTPException(
                status_code=400,
                detail=f"Import failed: {result.errors[0] if result.errors else 'No pairs found'}",
            )

        # Validate imported pairs
        validation = validate_qa_pairs([p.to_dict() for p in result.pairs])

        if not validation.valid:
            logger.warning(f"Validation issues: {validation.errors}")

        # Convert to standard format and save
        dataset_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        pairs = [
            {
                "query": p.question,
                "ground_truth": p.expected_answer,
                "alternative_answers": p.alternative_answers,
                "answer_type": p.answer_type,
                "difficulty": p.difficulty,
                "is_answerable": p.is_answerable,
                "metadata": p.metadata,
            }
            for p in result.pairs
        ]

        data = {
            "id": dataset_id,
            "name": name,
            "pairs": pairs,
            "created_at": created_at,
            "source_format": result.source_format,
            "import_stats": {
                "total_processed": result.total_processed,
                "skipped": result.skipped,
                "imported": len(result.pairs),
            },
        }

        _save_dataset(dataset_id, data)

        return {
            "id": dataset_id,
            "name": name,
            "pair_count": len(result.pairs),
            "source_format": result.source_format,
            "created_at": created_at,
            "validation": validation.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset import failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.post("/datasets/{dataset_id}/validate")
async def validate_dataset(dataset_id: str):
    """
    Validate an evaluation dataset.

    Checks for:
    - Empty questions/answers
    - Duplicate questions
    - Format consistency
    """
    from services.dataset_validator import validate_qa_pairs

    try:
        dataset = _load_dataset(dataset_id)
        pairs = dataset.get("pairs", [])

        result = validate_qa_pairs(pairs)

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/runs/compare")
async def compare_evaluation_runs(
    run_id_a: str,
    run_id_b: str,
    metrics: Optional[list[str]] = None,
    alpha: float = 0.05,
):
    """
    Statistically compare two evaluation runs.

    Computes:
    - Paired t-test / Wilcoxon signed-rank test
    - Cohen's d effect size
    - Significance determination with multiple testing correction

    Args:
        run_id_a: First run ID
        run_id_b: Second run ID
        metrics: Metrics to compare (default: all available)
        alpha: Significance level (default: 0.05)

    Returns:
        Statistical comparison results with p-values and effect sizes
    """
    from services.statistics import (
        compare_runs,
        bootstrap_ci,
        benjamini_hochberg_correction,
        summarize_comparison,
    )

    # Load both runs
    path_a = EVAL_RUNS_DIR / f"{run_id_a}.json"
    path_b = EVAL_RUNS_DIR / f"{run_id_b}.json"

    if not path_a.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id_a} not found")
    if not path_b.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id_b} not found")

    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    results_a = data_a.get("results", [])
    results_b = data_b.get("results", [])

    if len(results_a) != len(results_b):
        raise HTTPException(
            status_code=400,
            detail=f"Runs have different number of results ({len(results_a)} vs {len(results_b)}). "
                   "Cannot perform paired comparison."
        )

    # Default metrics to compare
    if metrics is None:
        metrics = ["answer_correctness", "faithfulness", "context_precision", "relevancy"]

    # Extract scores for each metric
    comparisons = []
    for metric_name in metrics:
        scores_a = []
        scores_b = []

        for r_a, r_b in zip(results_a, results_b):
            # Get scores from the results
            score_a = r_a.get("scores", {}).get(metric_name, 0.0) if r_a.get("scores") else 0.0
            score_b = r_b.get("scores", {}).get(metric_name, 0.0) if r_b.get("scores") else 0.0
            scores_a.append(score_a)
            scores_b.append(score_b)

        if scores_a and scores_b:
            comparison = compare_runs(scores_a, scores_b, metric_name, alpha=alpha)
            comparisons.append(comparison)

    # Apply multiple testing correction
    if comparisons:
        p_values = [c.p_value for c in comparisons if c.p_value is not None]
        if p_values:
            significant, adjusted_p = benjamini_hochberg_correction(p_values, alpha)
            # Update significance based on correction
            for i, comp in enumerate(comparisons):
                if i < len(significant):
                    comp.is_significant = significant[i]

    # Generate summary
    summary = summarize_comparison(comparisons, alpha)

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "comparisons": [c.to_dict() for c in comparisons],
        "alpha": alpha,
        "summary": summary,
        "n_samples": len(results_a),
    }


# --- Dataset Contamination & Split Endpoints ---


@router.post("/datasets/{dataset_id}/check-contamination")
async def check_dataset_contamination(
    dataset_id: str,
    training_dataset_id: Optional[str] = None,
    check_ngram: bool = True,
    check_semantic: bool = False,
):
    """
    Check for contamination between evaluation dataset and training data.

    Args:
        dataset_id: Evaluation dataset to check
        training_dataset_id: Training dataset to check against (optional)
        check_ngram: Enable n-gram overlap detection
        check_semantic: Enable semantic similarity detection (slower)

    Returns:
        Contamination detection results
    """
    from services.contamination_checker import ContaminationChecker

    try:
        dataset = _load_dataset(dataset_id)
        pairs = dataset.get("pairs", [])

        checker = ContaminationChecker()

        if training_dataset_id:
            # Check against another dataset
            training_dataset = _load_dataset(training_dataset_id)
            training_pairs = training_dataset.get("pairs", [])
            # Extract text content from training pairs
            training_corpus = []
            for p in training_pairs:
                q = p.get("question", p.get("query", ""))
                a = p.get("expected_answer", p.get("ground_truth", ""))
                training_corpus.extend([q, a])

            result = checker.check_contamination(
                pairs,
                training_corpus,
                check_exact=True,
                check_ngram=check_ngram,
                check_semantic=check_semantic,
            )
        else:
            # Check for internal duplicates only
            result = checker.check_internal_contamination(pairs)

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contamination check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Contamination check failed: {str(e)}")


@router.post("/datasets/{dataset_id}/split")
async def split_dataset(
    dataset_id: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    method: str = "hash",
):
    """
    Split a dataset into train/val/test splits.

    Args:
        dataset_id: Dataset to split
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for testing (default 0.15)
        seed: Random seed for reproducibility
        method: "hash" (deterministic) or "random"

    Returns:
        Split information and updated dataset
    """
    from services.dataset_splitter import DatasetSplitter

    try:
        dataset = _load_dataset(dataset_id)
        pairs = dataset.get("pairs", [])

        splitter = DatasetSplitter()

        if method == "hash":
            result = splitter.split_by_hash(pairs, train_ratio, val_ratio, test_ratio, seed)
        else:
            result = splitter.split_by_ratio(pairs, train_ratio, val_ratio, test_ratio, seed)

        # Add split labels to pairs and update dataset
        labeled_pairs = splitter.assign_split_labels(
            pairs, train_ratio, val_ratio, test_ratio, seed, method
        )

        # Update dataset with labeled pairs
        dataset["pairs"] = labeled_pairs
        dataset["split_info"] = {
            "method": method,
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "train_count": result.train.pair_count,
            "val_count": result.val.pair_count,
            "test_count": result.test.pair_count,
        }

        _save_dataset(dataset_id, dataset)

        return {
            "dataset_id": dataset_id,
            "splits": result.to_dict(),
            "message": f"Split {len(pairs)} pairs into train/val/test",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset split failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Split failed: {str(e)}")


@router.post("/datasets/{dataset_id}/deduplicate")
async def deduplicate_dataset(
    dataset_id: str,
    method: str = "exact",
    similarity_threshold: float = 0.85,
):
    """
    Remove duplicate pairs from a dataset.

    Args:
        dataset_id: Dataset to deduplicate
        method: "exact" (hash-based) or "semantic" (embedding-based)
        similarity_threshold: For semantic method, similarity threshold

    Returns:
        Deduplication results and updated dataset
    """
    from services.dataset_validator import deduplicate_pairs, add_content_hashes

    try:
        dataset = _load_dataset(dataset_id)
        pairs = dataset.get("pairs", [])

        original_count = len(pairs)
        deduplicated, removed = deduplicate_pairs(pairs, method, similarity_threshold)

        # Add content hashes to deduplicated pairs
        deduplicated = add_content_hashes(deduplicated)

        # Update dataset
        dataset["pairs"] = deduplicated
        dataset["deduplication_info"] = {
            "method": method,
            "original_count": original_count,
            "deduplicated_count": len(deduplicated),
            "removed_count": len(removed),
        }

        _save_dataset(dataset_id, dataset)

        return {
            "dataset_id": dataset_id,
            "original_count": original_count,
            "deduplicated_count": len(deduplicated),
            "removed_count": len(removed),
            "removed_pairs": [
                {
                    "question": p.get("question", p.get("query", ""))[:100],
                    "duplicate_of": p.get("_duplicate_of"),
                }
                for p in removed[:20]  # Limit output
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deduplication failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deduplication failed: {str(e)}")


# =============================================================================
# Background Evaluation Task Endpoints
# =============================================================================

@router.post("/tasks/start")
async def start_evaluation_task(request: RunEvaluationRequest):
    """
    Start an evaluation as a background task.

    Returns immediately with a task ID that can be used to track progress.
    The evaluation runs in the background and results are saved when complete.

    This is the recommended way to run evaluations as it:
    - Survives browser disconnections
    - Provides real-time progress updates
    - Can be cancelled if needed

    Supports two modes:
    1. Legacy mode: collection_name provided (use existing indexed collection)
    2. New mode: preprocessed_dataset_id + chunking config (on-demand embedding)
    """
    from services.evaluation_task_service import get_evaluation_task_service

    service = get_evaluation_task_service()

    # Prepare chunking config dict if provided
    chunking_dict = None
    if request.chunking:
        chunking_dict = {
            "method": request.chunking.method,
            "chunk_size": request.chunking.chunk_size,
            "chunk_overlap": request.chunking.chunk_overlap,
        }

    task_id = await service.create_task(
        experiment_name=request.experiment_name,
        eval_dataset_id=request.eval_dataset_id,
        collection_name=request.collection_name,
        use_rag=request.use_rag,
        use_colbert=request.use_colbert,
        top_k=request.top_k,
        temperature=request.temperature,
        embedder=request.embedder,
        # New flow parameters
        preprocessed_dataset_id=request.preprocessed_dataset_id,
        preset=request.preset,
        chunking=chunking_dict,
    )

    return {"task_id": task_id, "message": "Evaluation started"}


@router.get("/tasks/{task_id}")
async def get_evaluation_task(task_id: str):
    """
    Get the status and progress of an evaluation task.

    Returns:
        Task details including:
        - status: pending, running, completed, failed, cancelled
        - current_pair: Current pair being evaluated
        - total_pairs: Total pairs to evaluate
        - progress_percent: Completion percentage
        - current_step: Human-readable current action
        - result_run_id: ID of saved run (when completed)
    """
    from services.evaluation_task_service import get_evaluation_task_service

    service = get_evaluation_task_service()
    task = service.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return task


@router.get("/tasks")
async def list_evaluation_tasks(limit: int = 20):
    """
    List recent evaluation tasks.

    Returns the most recent tasks ordered by creation time.
    """
    from services.evaluation_task_service import get_evaluation_task_service

    service = get_evaluation_task_service()
    return service.list_tasks(limit=limit)


@router.post("/tasks/{task_id}/cancel")
async def cancel_evaluation_task(task_id: str):
    """
    Cancel a running evaluation task.

    Only running tasks can be cancelled. Completed or failed tasks
    cannot be cancelled.
    """
    from services.evaluation_task_service import get_evaluation_task_service

    service = get_evaluation_task_service()
    success = await service.cancel_task(task_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} cannot be cancelled (not running or not found)"
        )

    return {"message": f"Task {task_id} cancelled"}


# =============================================================================
# Experiment Presets & Embedding Cache Endpoints
# =============================================================================

@router.get("/presets")
async def list_experiment_presets():
    """
    List available experiment presets.

    Returns predefined configurations for quick setup:
    - Quick Test: Fast iteration with smaller chunks
    - Balanced: Good balance of quality and speed (recommended)
    - High Quality: Best accuracy with larger chunks and better embedder
    """
    return {
        name: {
            "name": preset.name,
            "description": preset.description,
            "chunking": {
                "method": preset.chunking.method,
                "chunk_size": preset.chunking.chunk_size,
                "chunk_overlap": preset.chunking.chunk_overlap,
            },
            "embedder": preset.embedder,
            "top_k": preset.top_k,
            "reranker": preset.reranker,
            "temperature": preset.temperature,
        }
        for name, preset in EXPERIMENT_PRESETS.items()
    }


@router.get("/embedding-cache")
async def list_embedding_caches(preprocessed_dataset_id: int = None):
    """
    List all cached embedding collections.

    Cached collections can be reused to avoid re-embedding when running
    experiments with the same configuration.
    """
    db_gen = get_db_session()
    db = next(db_gen)

    try:
        caches = embedding_service.list_cached_collections(
            db, preprocessed_dataset_id=preprocessed_dataset_id
        )
        return caches
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


@router.delete("/embedding-cache/{cache_id}")
async def delete_embedding_cache(cache_id: int):
    """
    Delete a cached embedding collection.

    This removes the cache entry. The vector store collection may still
    exist and should be cleaned up separately if needed.
    """
    db_gen = get_db_session()
    db = next(db_gen)

    try:
        deleted = embedding_service.delete_cached_collection(db, cache_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Cache {cache_id} not found")
        return {"status": "deleted", "id": cache_id}
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
