"""Pydantic configuration schemas matching backend exactly."""

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# CLEANING CONFIG
# =============================================================================


class CleaningConfig(BaseModel):
    """Document cleaning configuration."""

    enabled: bool = False
    remove_headers_footers: bool = False
    remove_page_numbers: bool = False
    normalize_whitespace: bool = True
    custom_patterns: list[str] = Field(default_factory=list)
    remove_html_markup: bool = False
    remove_urls: bool = False
    remove_citations: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    normalize_unicode: bool = False
    preserve_code_blocks: bool = True


# =============================================================================
# METADATA CONFIGS
# =============================================================================


class LightweightMetadataConfig(BaseModel):
    """Lightweight (non-LLM) metadata extraction config."""

    enabled: bool = False
    extract_rake_keywords: bool = False
    extract_statistics: bool = False
    detect_language: bool = False
    extract_spacy_entities: bool = False


class LLMMetadataConfig(BaseModel):
    """LLM-based metadata extraction config."""

    enabled: bool = False
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    extract_summary: bool = True
    extract_keywords: bool = True
    extract_entities: bool = True
    extract_categories: bool = True
    max_summary_length: int = Field(default=200, ge=50, le=1000)
    max_keywords: int = Field(default=10, ge=1, le=50)


# =============================================================================
# CHUNKING CONFIG
# =============================================================================


class ChunkingConfig(BaseModel):
    """Chunking configuration for preprocessing or evaluation."""

    method: Literal["recursive", "fixed", "semantic"] = "recursive"
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    embedder_model: Optional[str] = None
    breakpoint_threshold_type: str = "percentile"

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


# =============================================================================
# EMBEDDER CONFIG
# =============================================================================


class EmbedderConfig(BaseModel):
    """Embedder model configuration."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_type: Literal["huggingface", "openai", "custom"] = "huggingface"
    dimensions: Optional[int] = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# PREPROCESSING CONFIG
# =============================================================================


class PreprocessingConfig(BaseModel):
    """Full preprocessing configuration."""

    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    lightweight_metadata: LightweightMetadataConfig = Field(
        default_factory=LightweightMetadataConfig
    )
    llm_metadata: LLMMetadataConfig = Field(default_factory=LLMMetadataConfig)
    chunking: Optional[ChunkingConfig] = None  # None = defer to eval time


# =============================================================================
# Q&A GENERATION CONFIG
# =============================================================================


class QAGenerationConfig(BaseModel):
    """Q&A pair generation configuration."""

    name: str = Field(..., min_length=1, max_length=100)
    use_vllm: bool = False
    model: Optional[str] = None
    pairs_per_chunk: int = Field(default=2, ge=1, le=5)
    max_chunks: Optional[int] = Field(default=50, ge=1)
    temperature: float = Field(default=0.3, ge=0, le=1)
    seed: Optional[int] = None
    system_prompt: Optional[str] = None


# =============================================================================
# EVALUATION CONFIG
# =============================================================================


class EvaluationConfig(BaseModel):
    """Evaluation run configuration."""

    preset: Optional[Literal["quick", "balanced", "high_quality", "romanian"]] = None
    chunking: Optional[ChunkingConfig] = None
    use_rag: bool = True
    use_colbert: bool = False
    top_k: int = Field(default=5, ge=1, le=50)
    temperature: float = Field(default=0.1, ge=0, le=1)
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    experiment_name: Optional[str] = None

    @model_validator(mode="after")
    def validate_chunking_or_preset(self) -> "EvaluationConfig":
        # Either preset or chunking must be provided for new-style evaluation
        # (validation is optional - backend handles this too)
        return self


# =============================================================================
# DATA SOURCE CONFIG
# =============================================================================


class FilesSourceConfig(BaseModel):
    """Source from local files."""

    type: Literal["files"] = "files"
    file_paths: list[Path]
    dataset_name: str = Field(..., min_length=1, max_length=100)


class ExistingRawSourceConfig(BaseModel):
    """Source from existing raw dataset."""

    type: Literal["existing_raw"] = "existing_raw"
    raw_dataset_id: int


class ExistingProcessedSourceConfig(BaseModel):
    """Source from existing processed dataset."""

    type: Literal["existing_processed"] = "existing_processed"
    processed_dataset_id: int


DatasetSourceConfig = FilesSourceConfig | ExistingRawSourceConfig | ExistingProcessedSourceConfig


# =============================================================================
# RESULT MODELS
# =============================================================================


class PreprocessResult(BaseModel):
    """Result from preprocessing stage."""

    raw_dataset_id: int
    processed_dataset_id: int
    chunk_count: int = 0
    document_count: int = 0


class GenerateQAResult(BaseModel):
    """Result from Q&A generation stage."""

    eval_dataset_id: str
    name: str
    pair_count: int
    chunks_processed: int


class EvalMetrics(BaseModel):
    """Evaluation metrics."""

    context_precision: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    avg_latency: float = 0.0
    # Legacy
    answer_relevancy: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_correctness: Optional[float] = None


class EvaluateResult(BaseModel):
    """Result from evaluation stage."""

    run_id: str
    metrics: EvalMetrics
    result_count: int = 0


class PipelineResult(BaseModel):
    """Result from full pipeline run."""

    name: str
    preprocess: PreprocessResult
    qa_generation: GenerateQAResult
    evaluation: EvaluateResult
    duration_seconds: float
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# FULL BENCHMARK CONFIG (YAML)
# =============================================================================


class FullBenchmarkConfig(BaseModel):
    """Full benchmark configuration for pipeline command."""

    name: str = Field(..., min_length=1, max_length=100)

    # Stage 1: Source and preprocessing
    source: DatasetSourceConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    embedder_config: EmbedderConfig = Field(default_factory=EmbedderConfig)
    vector_backend: Literal["pgvector", "qdrant"] = "pgvector"

    # Stage 2: Q&A generation
    qa_generation: QAGenerationConfig

    # Stage 3: Evaluation
    evaluation: EvaluationConfig

    # Output
    output_dir: Path = Path("benchmark_results")

    @field_validator("source", mode="before")
    @classmethod
    def parse_source(cls, v: Any) -> DatasetSourceConfig:
        if isinstance(v, dict):
            source_type = v.get("type")
            if source_type == "files":
                return FilesSourceConfig(**v)
            elif source_type == "existing_raw":
                return ExistingRawSourceConfig(**v)
            elif source_type == "existing_processed":
                return ExistingProcessedSourceConfig(**v)
        return v


# =============================================================================
# PRESETS
# =============================================================================


PRESETS = {
    "quick": {
        "chunking": ChunkingConfig(method="recursive", chunk_size=500, chunk_overlap=50),
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 3,
        "use_colbert": False,
    },
    "balanced": {
        "chunking": ChunkingConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 5,
        "use_colbert": False,
    },
    "high_quality": {
        "chunking": ChunkingConfig(method="recursive", chunk_size=1500, chunk_overlap=300),
        "embedder": "BAAI/bge-small-en-v1.5",
        "top_k": 7,
        "use_colbert": True,
    },
    "romanian": {
        "chunking": ChunkingConfig(method="recursive", chunk_size=1500, chunk_overlap=300),
        "embedder": "BlackKakapo/stsb-xlm-r-multilingual-ro",
        "top_k": 5,
        "use_colbert": False,
    },
}
