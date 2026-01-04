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

    eval_dataset_id: Optional[str] = Field(None, description="ID of the evaluation dataset to use (optional)")
    collection_name: str = Field(..., description="Vector store collection to query")
    use_rag: bool = Field(True, description="Whether to enable RAG")
    use_colbert: bool = Field(True, description="Whether to enable ColBERT reranking")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=50)
    temperature: float = Field(0.1, description="LLM temperature", ge=0, le=1)
    embedder: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Embedding model to use for retrieval")
    # Optional: inline test query when no dataset is provided
    test_query: Optional[str] = Field(None, description="Single test query to run (used when no dataset)")


class RetrievedChunk(BaseModel):
    """Information about a retrieved document chunk."""

    content: str
    source: Optional[str] = None
    score: Optional[float] = None


class ClaimVerificationDetail(BaseModel):
    """Details about a single claim verification."""

    claim: str
    supported: bool
    evidence: Optional[str] = None


class FaithfulnessDetail(BaseModel):
    """Detailed faithfulness evaluation breakdown."""

    total_claims: int = 0
    supported_claims: int = 0
    claims: list[ClaimVerificationDetail] = []


class AnswerCorrectnessDetail(BaseModel):
    """Detailed answer correctness breakdown."""

    factual_score: float = 0.0
    semantic_score: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


class EvalResultScores(BaseModel):
    """Detailed scores for a single evaluation result."""

    # Original metric (kept for backwards compatibility)
    jaccard: Optional[float] = None  # Word overlap similarity

    # RAGAS-style metrics
    relevancy: Optional[float] = None  # Embedding similarity: query <-> answer
    faithfulness: Optional[float] = None  # Claims supported by context
    answer_correctness: Optional[float] = None  # F1 factual + semantic similarity
    context_precision: Optional[float] = None  # Retrieval ranking quality

    # Detailed breakdowns (optional, for expanded view)
    faithfulness_detail: Optional[FaithfulnessDetail] = None
    answer_correctness_detail: Optional[AnswerCorrectnessDetail] = None


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
    answer_correctness: float = 0.0
    avg_latency: float = 0.0


class EvalConfig(BaseModel):
    """Configuration used for the evaluation run."""

    collection: str
    use_rag: bool
    use_colbert: bool
    top_k: int
    temperature: float
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"


class RunEvaluationResponse(BaseModel):
    """Response from running an evaluation."""

    eval_dataset_id: Optional[str] = None
    results: list[EvalResult]
    metrics: EvalMetrics
    config: EvalConfig


# --- Q&A Generation Schemas ---

class GenerateQARequest(BaseModel):
    """Request to generate Q&A pairs from a processed dataset."""

    processed_dataset_id: int = Field(..., description="ID of the processed dataset to generate pairs from")
    name: str = Field(..., description="Name for the evaluation dataset")
    model: str = Field("openai/gpt-4o-mini", description="OpenRouter model to use")
    pairs_per_chunk: int = Field(2, description="Number of Q&A pairs per chunk", ge=1, le=5)
    max_chunks: Optional[int] = Field(50, description="Maximum chunks to process (None for all)", ge=1)


class GenerateQAResponse(BaseModel):
    """Response from Q&A generation."""

    id: str = Field(..., description="ID of the created evaluation dataset")
    name: str = Field(..., description="Name of the evaluation dataset")
    pair_count: int = Field(..., description="Number of Q&A pairs generated")
    chunks_processed: int = Field(..., description="Number of chunks processed")
    created_at: str = Field(..., description="Creation timestamp")


class GenerateSampleRequest(BaseModel):
    """Request to generate sample Q&A pairs for preview."""

    processed_dataset_id: int = Field(..., description="ID of the processed dataset")
    sample_size: int = Field(3, description="Number of chunks to sample", ge=1, le=10)


class GeneratedPair(BaseModel):
    """A generated Q&A pair."""

    query: str
    ground_truth: str
    difficulty: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None


class GenerateSampleResponse(BaseModel):
    """Response with sample Q&A pairs."""

    pairs: list[GeneratedPair]
    chunks_sampled: int
