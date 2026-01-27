"""Schemas for RAG evaluation API endpoints."""

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# --- Chunking Configuration (moved from preprocessing to evaluation) ---

class ChunkingConfig(BaseModel):
    """Configuration for document chunking during evaluation."""

    method: Literal["recursive", "fixed", "semantic"] = Field(
        default="recursive",
        description="Chunking method"
    )
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, ge=0, le=2000, description="Overlap between chunks")

    @model_validator(mode='after')
    def validate_overlap_less_than_size(self) -> 'ChunkingConfig':
        """Validate that chunk_overlap is less than chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


# --- Experiment Presets ---

class ExperimentPreset(BaseModel):
    """Predefined experiment configuration for quick setup."""

    name: str
    description: str
    chunking: ChunkingConfig
    embedder: str
    top_k: int
    reranker: Literal["none", "colbert"]
    temperature: float


# Predefined presets
EXPERIMENT_PRESETS: dict[str, ExperimentPreset] = {
    "quick": ExperimentPreset(
        name="Quick Test",
        description="Fast iteration with smaller chunks and lightweight embedder",
        chunking=ChunkingConfig(method="recursive", chunk_size=500, chunk_overlap=50),
        embedder="sentence-transformers/all-MiniLM-L6-v2",
        top_k=3,
        reranker="none",
        temperature=0.1,
    ),
    "balanced": ExperimentPreset(
        name="Balanced",
        description="Good balance of quality and speed (recommended)",
        chunking=ChunkingConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        embedder="sentence-transformers/all-MiniLM-L6-v2",
        top_k=5,
        reranker="none",
        temperature=0.1,
    ),
    "high_quality": ExperimentPreset(
        name="High Quality",
        description="Best accuracy with larger chunks and better embedder",
        chunking=ChunkingConfig(method="recursive", chunk_size=1500, chunk_overlap=300),
        embedder="BAAI/bge-small-en-v1.5",
        top_k=7,
        reranker="colbert",
        temperature=0.1,
    ),
}


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
    collection_name: Optional[str] = Field(default=None, description="Vector store collection (legacy mode)")
    use_rag: bool = Field(default=True, description="Whether to use RAG")

    # New fields for experiment flow
    preprocessed_dataset_id: Optional[int] = Field(default=None, description="Preprocessed dataset ID (new mode)")
    chunking: Optional[ChunkingConfig] = Field(default=None, description="Chunking configuration")
    preset: Optional[str] = Field(default=None, description="Preset name if used")

    def config_hash(self) -> str:
        """Generate SHA-256 hash for deduplication/reproducibility."""
        config_dict = {
            "embedder_model": self.embedder_model,
            "reranker_strategy": self.reranker_strategy,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "dataset_id": self.dataset_id,
            "collection_name": self.collection_name,
            "preprocessed_dataset_id": self.preprocessed_dataset_id,
            "use_rag": self.use_rag,
        }
        # Include chunking in hash if present
        if self.chunking:
            config_dict["chunking"] = {
                "method": self.chunking.method,
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
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
    source_collection: Optional[str] = None  # The collection this dataset was generated from


class RunEvaluationRequest(BaseModel):
    """
    Request to run an evaluation experiment.

    Supports two modes:
    1. Legacy mode: Use existing collection_name (already chunked/embedded)
    2. New mode: Use preprocessed_dataset_id + chunking/embedder config (on-demand processing)
    """

    experiment_name: Optional[str] = Field(None, description="Custom name for this experiment")
    eval_dataset_id: Optional[str] = Field(None, description="ID of the evaluation dataset to use (optional)")

    # --- Source selection (one of these required) ---
    # Legacy: use existing vector store collection
    collection_name: Optional[str] = Field(None, description="Vector store collection to query (legacy mode)")
    # New: use preprocessed dataset with on-demand chunking/embedding
    preprocessed_dataset_id: Optional[int] = Field(None, description="Preprocessed dataset ID (new mode)")

    # --- Experiment preset (optional shortcut) ---
    preset: Optional[Literal["quick", "balanced", "high_quality"]] = Field(
        None, description="Use a predefined preset instead of custom config"
    )

    # --- Chunking config (new mode, or override preset) ---
    chunking: Optional[ChunkingConfig] = Field(
        None, description="Chunking configuration (required for new mode if no preset)"
    )

    # --- RAG config ---
    use_rag: bool = Field(True, description="Whether to enable RAG")
    use_colbert: bool = Field(False, description="Whether to enable ColBERT reranking")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=50)
    temperature: float = Field(0.1, description="LLM temperature", ge=0, le=1)
    embedder: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Embedding model to use for retrieval")

    # Optional: inline test query when no dataset is provided
    test_query: Optional[str] = Field(None, description="Single test query to run (used when no dataset)")

    @model_validator(mode='after')
    def validate_source(self) -> 'RunEvaluationRequest':
        """Validate that either collection_name or preprocessed_dataset_id is provided."""
        if not self.collection_name and not self.preprocessed_dataset_id:
            raise ValueError("Either collection_name or preprocessed_dataset_id must be provided")
        if self.preprocessed_dataset_id and not self.chunking and not self.preset:
            raise ValueError("When using preprocessed_dataset_id, either chunking config or preset is required")
        return self


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

    # New fields for experiment tracking
    preprocessed_dataset_id: Optional[int] = None  # Source dataset (new mode)
    chunking: Optional[ChunkingConfig] = None  # Chunking config used
    preset_name: Optional[str] = None  # Preset used, if any


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
    use_vllm: bool = Field(False, description="Use local vLLM instead of OpenRouter")
    model: Optional[str] = Field(None, description="Model to use (defaults based on provider)")
    pairs_per_chunk: int = Field(2, description="Number of Q&A pairs per chunk", ge=1, le=5)
    max_chunks: Optional[int] = Field(50, description="Maximum chunks to process (None for all)", ge=1)
    temperature: float = Field(0.3, description="Temperature for generation", ge=0, le=1)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


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
