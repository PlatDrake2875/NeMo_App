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

from fastapi import APIRouter, HTTPException

from schemas.evaluation import (
    CreateEvalDatasetRequest,
    EvalConfig,
    EvalDatasetResponse,
    EvalMetrics,
    EvalResult,
    EvalResultScores,
    RetrievedChunk,
    RunEvaluationRequest,
    RunEvaluationResponse,
)
from rag_components import get_rag_context_prefix, format_docs
from config import VLLM_BASE_URL, VLLM_MODEL
import httpx

logger = logging.getLogger(__name__)

# Directory for storing evaluation datasets
EVAL_DATASETS_DIR = Path("data/evaluation_datasets")
EVAL_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

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

    This will:
    1. Load the evaluation dataset
    2. For each Q&A pair:
       - Run the RAG pipeline with the query
       - Record the predicted answer, retrieved chunks, and latency
       - Calculate similarity score with ground truth
    3. Compute aggregate metrics
    """
    # Load the dataset
    dataset = _load_dataset(request.eval_dataset_id)
    pairs = dataset["pairs"]

    logger.info(
        f"Starting evaluation: dataset='{dataset['name']}', "
        f"pairs={len(pairs)}, collection={request.collection_name}, "
        f"rag={request.use_rag}, colbert={request.use_colbert}"
    )

    results = []
    total_relevancy = 0.0
    total_faithfulness = 0.0
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

            latency = time.time() - start_time

            # Calculate similarity score (simple word overlap for now)
            score = _calculate_similarity(predicted_answer, ground_truth)

            # Estimate relevancy and faithfulness (simplified)
            relevancy = _estimate_relevancy(query, predicted_answer)
            faithfulness = _estimate_faithfulness(predicted_answer, retrieved_chunks) if retrieved_chunks else 0.5

            total_relevancy += relevancy
            total_faithfulness += faithfulness
            total_latency += latency

            results.append(
                EvalResult(
                    query=query,
                    ground_truth=ground_truth,
                    predicted_answer=predicted_answer,
                    retrieved_chunks=retrieved_chunks,
                    score=score,
                    scores=EvalResultScores(
                        relevancy=relevancy,
                        faithfulness=faithfulness,
                    ),
                    latency=latency,
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
    n = len(results)
    metrics = EvalMetrics(
        answer_relevancy=total_relevancy / n if n > 0 else 0.0,
        faithfulness=total_faithfulness / n if n > 0 else 0.0,
        context_precision=sum(r.score for r in results) / n if n > 0 else 0.0,
        avg_latency=total_latency / n if n > 0 else 0.0,
    )

    logger.info(
        f"Evaluation complete: avg_relevancy={metrics.answer_relevancy:.2f}, "
        f"avg_faithfulness={metrics.faithfulness:.2f}, avg_latency={metrics.avg_latency:.2f}s"
    )

    return RunEvaluationResponse(
        eval_dataset_id=request.eval_dataset_id,
        results=results,
        metrics=metrics,
        config=EvalConfig(
            collection=request.collection_name,
            use_rag=request.use_rag,
            use_colbert=request.use_colbert,
            top_k=request.top_k,
            temperature=request.temperature,
        ),
    )


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


def _calculate_similarity(predicted: str, ground_truth: str) -> float:
    """
    Calculate similarity between predicted answer and ground truth.
    Uses simple word overlap (Jaccard similarity) as a baseline.
    """
    pred_words = set(predicted.lower().split())
    truth_words = set(ground_truth.lower().split())

    if not pred_words or not truth_words:
        return 0.0

    intersection = pred_words & truth_words
    union = pred_words | truth_words

    return len(intersection) / len(union) if union else 0.0


def _estimate_relevancy(query: str, answer: str) -> float:
    """
    Estimate how relevant the answer is to the query.
    Simple heuristic: check if key query terms appear in the answer.
    """
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())

    # Remove common words
    stop_words = {"what", "is", "the", "a", "an", "how", "does", "do", "are", "was", "were", "can", "could", "would", "should"}
    query_words -= stop_words

    if not query_words:
        return 0.5

    overlap = query_words & answer_words
    return min(1.0, len(overlap) / len(query_words) + 0.3)  # Baseline of 0.3


def _estimate_faithfulness(answer: str, chunks: Optional[list[RetrievedChunk]]) -> float:
    """
    Estimate if the answer is grounded in the retrieved context.
    Simple heuristic: check word overlap between answer and context.
    """
    if not chunks:
        return 0.5

    context_text = " ".join(c.content for c in chunks)
    context_words = set(context_text.lower().split())
    answer_words = set(answer.lower().split())

    # Remove common words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "and", "but", "if", "or", "because", "until", "while", "this", "that", "these", "those", "it", "its"}
    answer_words -= stop_words
    context_words -= stop_words

    if not answer_words:
        return 0.5

    overlap = answer_words & context_words
    return min(1.0, len(overlap) / len(answer_words) + 0.2)  # Baseline of 0.2
