"""Schemas for RAG evaluation API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field


class QAPair(BaseModel):
    """A single question-answer pair for evaluation."""

    query: str = Field(..., description="The test question")
    ground_truth: str = Field(..., description="The expected answer")


class CreateEvalDatasetRequest(BaseModel):
    """Request to create a new evaluation dataset."""

    name: str = Field(..., description="Name of the evaluation dataset")
    pairs: list[QAPair] = Field(..., description="List of Q&A pairs", min_length=1)


class EvalDatasetResponse(BaseModel):
    """Response containing evaluation dataset info."""

    id: str
    name: str
    pair_count: int
    created_at: str


class RunEvaluationRequest(BaseModel):
    """Request to run an evaluation."""

    eval_dataset_id: str = Field(..., description="ID of the evaluation dataset to use")
    collection_name: str = Field(..., description="Vector store collection to query")
    use_rag: bool = Field(True, description="Whether to enable RAG")
    use_colbert: bool = Field(True, description="Whether to enable ColBERT reranking")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=50)
    temperature: float = Field(0.1, description="LLM temperature", ge=0, le=1)


class RetrievedChunk(BaseModel):
    """Information about a retrieved document chunk."""

    content: str
    source: Optional[str] = None
    score: Optional[float] = None


class EvalResultScores(BaseModel):
    """Detailed scores for a single evaluation result."""

    relevancy: Optional[float] = None
    faithfulness: Optional[float] = None


class EvalResult(BaseModel):
    """Result of evaluating a single Q&A pair."""

    query: str
    ground_truth: str
    predicted_answer: str
    retrieved_chunks: Optional[list[RetrievedChunk]] = None
    score: float  # Overall similarity score
    scores: Optional[EvalResultScores] = None
    latency: float  # Response time in seconds


class EvalMetrics(BaseModel):
    """Aggregate metrics across all evaluation results."""

    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    context_precision: float = 0.0
    avg_latency: float = 0.0


class EvalConfig(BaseModel):
    """Configuration used for the evaluation run."""

    collection: str
    use_rag: bool
    use_colbert: bool
    top_k: int
    temperature: float


class RunEvaluationResponse(BaseModel):
    """Response from running an evaluation."""

    eval_dataset_id: str
    results: list[EvalResult]
    metrics: EvalMetrics
    config: EvalConfig
