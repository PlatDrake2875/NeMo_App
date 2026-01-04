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
    ClaimVerificationDetail,
    CreateEvalDatasetRequest,
    EvalConfig,
    EvalDatasetResponse,
    EvalMetrics,
    EvalResult,
    EvalResultScores,
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
        f"pairs={len(pairs)}, collection={request.collection_name}, "
        f"rag={request.use_rag}, colbert={request.use_colbert}"
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
                    collection_name=request.collection_name,
                    use_colbert=request.use_colbert,
                    embedder=request.embedder,
                )
                prompt_content = rag_prefix if rag_prefix else query
            else:
                prompt_content = query

            # Call vLLM to generate answer
            predicted_answer, retrieved_chunks = await _generate_answer(
                prompt_content,
                query,
                request.temperature,
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
            collection=request.collection_name,
            use_rag=request.use_rag,
            use_colbert=request.use_colbert,
            top_k=request.top_k,
            temperature=request.temperature,
            embedder=request.embedder,
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
                runs.append({
                    "id": data["id"],
                    "name": data.get("name", "Unknown"),
                    "created_at": data["created_at"],
                    "pair_count": len(data.get("results", [])),
                    "metrics": data.get("metrics", {}),
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
        return json.load(f)


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

    This will:
    1. Retrieve chunks from the processed dataset
    2. Use OpenRouter LLM to generate Q&A pairs from each chunk
    3. Save the generated pairs as an evaluation dataset
    """
    try:
        generator = QAGeneratorService(model=request.model)

        result = await generator.generate_pairs_for_dataset(
            processed_dataset_id=request.processed_dataset_id,
            name=request.name,
            pairs_per_chunk=request.pairs_per_chunk,
            max_chunks=request.max_chunks,
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
