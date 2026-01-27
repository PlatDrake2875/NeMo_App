"""Schemas for preprocessing configuration in the RAG Benchmark Hub."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from enums import ProcessingStatus

# Backward compatibility alias
ProcessingStatusEnum = ProcessingStatus


class CleaningConfig(BaseModel):
    """Configuration for document cleaning."""

    enabled: bool = False
    # Original options
    remove_headers_footers: bool = False
    remove_page_numbers: bool = False
    normalize_whitespace: bool = True
    custom_patterns: List[str] = Field(default_factory=list)

    # Content removal
    remove_html_markup: bool = False
    remove_urls: bool = False
    remove_citations: bool = False

    # Privacy/PII
    remove_emails: bool = False
    remove_phone_numbers: bool = False

    # Formatting
    normalize_unicode: bool = False
    preserve_code_blocks: bool = True

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "remove_headers_footers": True,
                "remove_page_numbers": True,
                "normalize_whitespace": True,
                "custom_patterns": [],
                "remove_html_markup": False,
                "remove_urls": False,
                "remove_citations": False,
                "remove_emails": False,
                "remove_phone_numbers": False,
                "normalize_unicode": False,
                "preserve_code_blocks": True
            }
        }
    )


class LLMMetadataConfig(BaseModel):
    """Configuration for LLM metadata extraction."""

    enabled: bool = False
    model: str = Field(default="meta-llama/Llama-3.2-3B-Instruct")
    extract_summary: bool = True
    extract_keywords: bool = True
    extract_entities: bool = True
    extract_categories: bool = True
    max_summary_length: int = Field(default=200, ge=50, le=1000)
    max_keywords: int = Field(default=10, ge=1, le=50)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "extract_summary": True,
                "extract_keywords": True,
                "extract_entities": True,
                "extract_categories": True,
                "max_summary_length": 200,
                "max_keywords": 10
            }
        }
    )


class LightweightMetadataConfig(BaseModel):
    """Configuration for lightweight metadata extraction (no LLM required)."""

    enabled: bool = False
    extract_rake_keywords: bool = False  # RAKE algorithm for keyword extraction
    extract_statistics: bool = False  # Word count, readability scores
    detect_language: bool = False  # Language detection
    extract_spacy_entities: bool = False  # Named entity recognition (requires spaCy)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "extract_rake_keywords": True,
                "extract_statistics": True,
                "detect_language": True,
                "extract_spacy_entities": False
            }
        }
    )


class ChunkingConfigSchema(BaseModel):
    """Configuration for document chunking."""

    method: str = Field(default="recursive", pattern="^(recursive|fixed|semantic)$")
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    # Semantic chunking specific
    embedder_model: Optional[str] = None
    breakpoint_threshold_type: str = Field(default="percentile")

    @model_validator(mode='after')
    def validate_overlap_less_than_size(self) -> 'ChunkingConfigSchema':
        """Validate that chunk_overlap is less than chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "method": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "embedder_model": None,
                "breakpoint_threshold_type": "percentile"
            }
        }
    )


class PreprocessingConfig(BaseModel):
    """
    Complete preprocessing configuration.

    In the new flow, preprocessing stops at cleaning/metadata extraction.
    Chunking and embedding happen during evaluation, allowing experimentation
    with different chunking/embedding strategies on the same cleaned data.

    Set `chunking` to None (or omit) to use the new flow.
    Legacy datasets may still have chunking config for backward compatibility.
    """

    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    lightweight_metadata: LightweightMetadataConfig = Field(
        default_factory=LightweightMetadataConfig
    )
    llm_metadata: LLMMetadataConfig = Field(default_factory=LLMMetadataConfig)

    # Chunking is now optional - if None, preprocessing stops at cleaning
    # and chunking happens during evaluation (new flow)
    chunking: Optional[ChunkingConfigSchema] = Field(
        default=None,
        description="Chunking config. If None, chunking happens at evaluation time (recommended)."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cleaning": {"enabled": True, "normalize_whitespace": True},
                "lightweight_metadata": {"enabled": True, "extract_rake_keywords": True},
                "llm_metadata": {"enabled": False},
                "chunking": None  # Chunking at evaluation time
            }
        }
    )


class ProcessingStatusResponse(BaseModel):
    """Response for processing status check."""

    status: ProcessingStatusEnum
    progress_percent: Optional[float] = None
    current_step: Optional[str] = None
    error: Optional[str] = None
