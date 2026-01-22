"""Schemas for RAG evaluation API endpoints."""

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class QAPair(BaseModel):
    """A single question-answer pair for evaluation."""

    query: str = Field(..., description="The test question")
    ground_truth: str = Field(..., description="The expected answer")


class EnhancedQAPair(BaseModel):
    """Enhanced Q&A pair with additional metadata for evaluation."""

    question: str = Field(..., description="The test question")
    expected_answer: str = Field(..., description="The expected/reference answer")
    alternative_answers: list[str] = Field(default_factory=list, description="Other valid answers")
    answer_type: Literal["extractive", "abstractive", "yes_no", "multi_hop"] = Field(
        default="abstractive", description="Type of answer expected"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Question difficulty level"
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of source chunks containing the answer"
    )
    is_answerable: bool = Field(default=True, description="Whether the question is answerable from context")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvalConfigSchema(BaseModel):
    """Configuration for an evaluation run - first-class entity for reproducibility."""

    embedder_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for retrieval"
    )
    reranker_strategy: Literal["none", "colbert"] = Field(
        default="none",
        description="Reranking strategy"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of documents to retrieve")
    temperature: float = Field(default=0.1, ge=0, le=1, description="LLM temperature")
    dataset_id: Optional[str] = Field(default=None, description="Evaluation dataset ID")
    collection_name: str = Field(default="rag_documents", description="Vector store collection")
    use_rag: bool = Field(default=True, description="Whether to use RAG")

    def config_hash(self) -> str:
        """Generate SHA-256 hash for deduplication/reproducibility."""
        config_dict = {
            "embedder_model": self.embedder_model,
            "reranker_strategy": self.reranker_strategy,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "dataset_id": self.dataset_id,
            "collection_name": self.collection_name,
            "use_rag": self.use_rag,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


class BatchEvalRequest(BaseModel):
    """Request to run batch evaluations with multiple configurations."""

    configs: list[EvalConfigSchema] = Field(..., min_length=1, description="List of configs to evaluate")
    dataset_id: Optional[str] = Field(None, description="Dataset ID to use for all configs")


class BatchEvalResponse(BaseModel):
    """Response from batch evaluation."""

    batch_id: str
    run_ids: list[str]
    config_hashes: list[str]


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
    llm_model: Optional[str] = None  # LLM model used for answer generation


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


# --- Reproducibility & Statistical Rigor Schemas ---


class EvalMetadata(BaseModel):
    """
    Metadata for reproducibility tracking in evaluation runs.

    Captures all information needed to reproduce an evaluation exactly.
    """

    # LLM configuration
    llm_model: str = Field(..., description="LLM model used for generation")
    llm_model_version: Optional[str] = Field(None, description="Specific model version if available")
    llm_temperature: float = Field(..., description="Temperature used for LLM")
    llm_seed: Optional[int] = Field(None, description="Random seed for LLM (if supported)")

    # Embedder configuration
    embedder_model: str = Field(..., description="Embedder model used for retrieval")
    embedder_model_version: Optional[str] = Field(None, description="Specific embedder version")

    # Metric versions for reproducibility
    metric_versions: dict[str, str] = Field(
        default_factory=dict, description="Mapping of metric names to their versions"
    )

    # Dataset fingerprinting
    dataset_content_hash: Optional[str] = Field(
        None, description="SHA-256 hash of dataset content for integrity"
    )

    # Determinism tracking
    determinism_guaranteed: bool = Field(
        False, description="Whether this run is fully deterministic"
    )
    determinism_warnings: list[str] = Field(
        default_factory=list, description="Warnings about non-deterministic elements"
    )

    # Timestamps
    run_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When the evaluation was run"
    )


class ConfidenceInterval(BaseModel):
    """Statistical confidence interval for a metric."""

    lower: float = Field(..., description="Lower bound of the interval")
    upper: float = Field(..., description="Upper bound of the interval")
    confidence_level: float = Field(0.95, description="Confidence level (e.g., 0.95 for 95%)")
    method: str = Field("bootstrap", description="Method used to compute CI")
    n_bootstrap: Optional[int] = Field(None, description="Number of bootstrap samples if applicable")


class MetricWithCI(BaseModel):
    """A metric score with confidence interval."""

    score: float = Field(..., description="Point estimate of the metric")
    confidence_interval: Optional[ConfidenceInterval] = Field(
        None, description="Confidence interval for the score"
    )
    std_error: Optional[float] = Field(None, description="Standard error of the estimate")
    sample_size: int = Field(..., description="Number of samples used to compute the metric")


class AggregateMetricsWithCI(BaseModel):
    """Aggregate metrics with confidence intervals for statistical rigor."""

    answer_relevancy: MetricWithCI
    faithfulness: MetricWithCI
    context_precision: MetricWithCI
    answer_correctness: MetricWithCI
    avg_latency: MetricWithCI


class ComparisonResult(BaseModel):
    """Result of comparing two evaluation runs."""

    metric_name: str = Field(..., description="Name of the metric being compared")
    run_a_mean: float = Field(..., description="Mean score for run A")
    run_b_mean: float = Field(..., description="Mean score for run B")
    difference: float = Field(..., description="Difference (B - A)")
    p_value: Optional[float] = Field(None, description="P-value from statistical test")
    effect_size: Optional[float] = Field(None, description="Cohen's d effect size")
    is_significant: bool = Field(..., description="Whether difference is statistically significant")
    test_method: str = Field("paired_t_test", description="Statistical test used")


class RunComparisonResponse(BaseModel):
    """Response from comparing two evaluation runs."""

    run_id_a: str
    run_id_b: str
    comparisons: list[ComparisonResult]
    alpha: float = Field(0.05, description="Significance level used")
    summary: str = Field(..., description="Human-readable summary of comparison")


# --- Dataset Split & Contamination Schemas ---


class DatasetSplit(BaseModel):
    """Information about a dataset split."""

    split: Literal["train", "val", "test"] = Field(..., description="Split type")
    pair_count: int = Field(..., description="Number of pairs in this split")
    content_hash: str = Field(..., description="Hash of this split's content")


class SplitDatasetRequest(BaseModel):
    """Request to split a dataset into train/val/test."""

    train_ratio: float = Field(0.7, ge=0, le=1, description="Proportion for training")
    val_ratio: float = Field(0.15, ge=0, le=1, description="Proportion for validation")
    test_ratio: float = Field(0.15, ge=0, le=1, description="Proportion for testing")
    seed: int = Field(42, description="Random seed for reproducible splits")
    method: Literal["random", "hash"] = Field(
        "hash", description="Split method: 'random' or 'hash' (deterministic)"
    )


class SplitDatasetResponse(BaseModel):
    """Response from splitting a dataset."""

    dataset_id: str
    splits: list[DatasetSplit]
    seed: int
    method: str


class ContaminationResult(BaseModel):
    """Result of contamination checking."""

    contaminated_pairs: int = Field(..., description="Number of potentially contaminated pairs")
    contamination_rate: float = Field(..., description="Proportion of contaminated pairs")
    exact_matches: int = Field(0, description="Pairs with exact hash matches")
    near_duplicates: int = Field(0, description="Pairs with high similarity")
    flagged_pairs: list[dict[str, Any]] = Field(
        default_factory=list, description="Details of flagged pairs"
    )


class DeduplicationResult(BaseModel):
    """Result of deduplication."""

    original_count: int
    deduplicated_count: int
    removed_count: int
    removed_pairs: list[dict[str, Any]] = Field(
        default_factory=list, description="Pairs that were removed"
    )


# --- Enhanced Q&A Pair with Lineage ---


class QAPairLineage(BaseModel):
    """Lineage information for a Q&A pair."""

    source_chunk_id: Optional[str] = Field(None, description="ID of source chunk")
    generation_model: Optional[str] = Field(None, description="Model that generated this pair")
    chunk_content_hash: Optional[str] = Field(None, description="Hash of source chunk content")
    created_from: Literal["generation", "import", "manual"] = Field(
        "manual", description="How this pair was created"
    )
    created_at: Optional[str] = Field(None, description="When this pair was created")
    parent_pair_id: Optional[str] = Field(None, description="ID of parent pair if derived")


class EnhancedQAPairV2(BaseModel):
    """
    Enhanced Q&A pair with versioning and lineage for reproducibility.

    This extends EnhancedQAPair with fields needed for publication-grade research.
    """

    question: str = Field(..., description="The test question")
    expected_answer: str = Field(..., description="The expected/reference answer")
    alternative_answers: list[str] = Field(default_factory=list)
    answer_type: Literal["extractive", "abstractive", "yes_no", "multi_hop"] = "abstractive"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    source_chunk_ids: list[str] = Field(default_factory=list)
    is_answerable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    # New fields for reproducibility
    split: Optional[Literal["train", "val", "test"]] = Field(
        None, description="Dataset split this pair belongs to"
    )
    content_hash: Optional[str] = Field(
        None, description="Hash of question+answer for deduplication"
    )
    version: int = Field(1, description="Version number of this pair")
    lineage: Optional[QAPairLineage] = Field(
        None, description="Lineage/provenance information"
    )


# --- Request/Response with Reproducibility ---


class RunEvaluationRequestV2(BaseModel):
    """Enhanced evaluation request with reproducibility features."""

    eval_dataset_id: Optional[str] = None
    collection_name: str = Field(...)
    use_rag: bool = True
    use_colbert: bool = True
    top_k: int = Field(5, ge=1, le=50)
    temperature: float = Field(0.1, ge=0, le=1)
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    test_query: Optional[str] = None

    # Reproducibility options
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    required_split: Optional[Literal["train", "val", "test"]] = Field(
        None, description="Only evaluate pairs from this split"
    )
    enforce_split_isolation: bool = Field(
        True, description="Reject evaluation on non-test splits by default"
    )


class RunEvaluationResponseV2(BaseModel):
    """Enhanced evaluation response with metadata and CIs."""

    eval_dataset_id: Optional[str] = None
    results: list[EvalResult]
    metrics: EvalMetrics
    config: EvalConfig

    # New reproducibility fields
    metadata: Optional[EvalMetadata] = Field(
        None, description="Reproducibility metadata"
    )
    metrics_with_ci: Optional[AggregateMetricsWithCI] = Field(
        None, description="Metrics with confidence intervals"
    )
