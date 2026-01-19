"""
Centralized configuration for RAG evaluation metrics.

This module provides a single source of truth for all metric parameters,
replacing hardcoded magic numbers throughout the metrics codebase.

All metrics should accept a MetricsConfig instance and use its values
instead of hardcoded defaults.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FallbackBehavior(Enum):
    """Behavior when LLM calls fail."""

    USE_HEURISTIC = "use_heuristic"  # Fall back to simple heuristics
    RAISE_ERROR = "raise_error"  # Raise exception
    RETURN_ZERO = "return_zero"  # Return 0 score


@dataclass
class MetricsConfig:
    """
    Centralized configuration for all RAG evaluation metrics.

    This configuration allows tuning metric behavior without modifying code.
    All parameters have sensible defaults based on empirical testing.

    Attributes:
        claim_similarity_threshold: Minimum cosine similarity for claims to be
            considered matching in F1 calculation. Range: 0-1. Default: 0.7
        factual_weight: Weight for factual F1 score in answer correctness.
            Semantic weight = 1 - factual_weight. Range: 0-1. Default: 0.5
        word_overlap_threshold: Minimum word overlap ratio for heuristic
            relevance fallback. Range: 0-1. Default: 0.3
        embedding_fallback_threshold: Minimum embedding similarity for
            heuristic relevance fallback. Range: 0-1. Default: 0.3
        max_text_length: Maximum characters for claim extraction input.
            Longer texts are truncated. Default: 3000
        max_context_length: Maximum characters for context in verification.
            Longer contexts are truncated. Default: 4000
        max_chunk_length: Maximum characters per chunk for relevance judging.
            Default: 2000
        max_ground_truth_length: Maximum characters for ground truth in prompts.
            Default: 1000
        ensemble_size: Number of LLM calls for ensemble scoring (for future use).
            Default: 1
        fallback_behavior: What to do when LLM calls fail.
            Default: USE_HEURISTIC
        llm_temperature: Temperature for LLM calls. Range: 0-1. Default: 0.1
        llm_max_tokens_claims: Max tokens for claim extraction. Default: 1024
        llm_max_tokens_verify: Max tokens for verification. Default: 256
        llm_max_tokens_relevance: Max tokens for relevance judgment. Default: 128
        version: Configuration version for tracking/reproducibility.
            Default: "2.0.0"
    """

    # Similarity thresholds
    claim_similarity_threshold: float = 0.7
    word_overlap_threshold: float = 0.3
    embedding_fallback_threshold: float = 0.3

    # Weighting
    factual_weight: float = 0.5

    # Text length limits
    max_text_length: int = 3000
    max_context_length: int = 4000
    max_chunk_length: int = 2000
    max_ground_truth_length: int = 1000

    # Ensemble/reliability
    ensemble_size: int = 1
    fallback_behavior: FallbackBehavior = FallbackBehavior.USE_HEURISTIC

    # LLM parameters
    llm_temperature: float = 0.1
    llm_max_tokens_claims: int = 1024
    llm_max_tokens_verify: int = 256
    llm_max_tokens_relevance: int = 128

    # Versioning
    version: str = "2.0.0"

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "claim_similarity_threshold": self.claim_similarity_threshold,
            "word_overlap_threshold": self.word_overlap_threshold,
            "embedding_fallback_threshold": self.embedding_fallback_threshold,
            "factual_weight": self.factual_weight,
            "max_text_length": self.max_text_length,
            "max_context_length": self.max_context_length,
            "max_chunk_length": self.max_chunk_length,
            "max_ground_truth_length": self.max_ground_truth_length,
            "ensemble_size": self.ensemble_size,
            "fallback_behavior": self.fallback_behavior.value,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens_claims": self.llm_max_tokens_claims,
            "llm_max_tokens_verify": self.llm_max_tokens_verify,
            "llm_max_tokens_relevance": self.llm_max_tokens_relevance,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsConfig":
        """Create config from dictionary."""
        fallback = data.get("fallback_behavior", "use_heuristic")
        if isinstance(fallback, str):
            fallback = FallbackBehavior(fallback)

        return cls(
            claim_similarity_threshold=data.get("claim_similarity_threshold", 0.7),
            word_overlap_threshold=data.get("word_overlap_threshold", 0.3),
            embedding_fallback_threshold=data.get("embedding_fallback_threshold", 0.3),
            factual_weight=data.get("factual_weight", 0.5),
            max_text_length=data.get("max_text_length", 3000),
            max_context_length=data.get("max_context_length", 4000),
            max_chunk_length=data.get("max_chunk_length", 2000),
            max_ground_truth_length=data.get("max_ground_truth_length", 1000),
            ensemble_size=data.get("ensemble_size", 1),
            fallback_behavior=fallback,
            llm_temperature=data.get("llm_temperature", 0.1),
            llm_max_tokens_claims=data.get("llm_max_tokens_claims", 1024),
            llm_max_tokens_verify=data.get("llm_max_tokens_verify", 256),
            llm_max_tokens_relevance=data.get("llm_max_tokens_relevance", 128),
            version=data.get("version", "2.0.0"),
        )


# Global default config instance
_default_config: Optional[MetricsConfig] = None


def get_default_config() -> MetricsConfig:
    """Get the global default metrics configuration."""
    global _default_config
    if _default_config is None:
        _default_config = MetricsConfig()
    return _default_config


def set_default_config(config: MetricsConfig) -> None:
    """Set the global default metrics configuration."""
    global _default_config
    _default_config = config


# Preset configurations for common use cases
STRICT_CONFIG = MetricsConfig(
    claim_similarity_threshold=0.8,
    word_overlap_threshold=0.4,
    embedding_fallback_threshold=0.4,
    fallback_behavior=FallbackBehavior.RAISE_ERROR,
    version="2.0.0-strict",
)

LENIENT_CONFIG = MetricsConfig(
    claim_similarity_threshold=0.6,
    word_overlap_threshold=0.2,
    embedding_fallback_threshold=0.2,
    fallback_behavior=FallbackBehavior.USE_HEURISTIC,
    version="2.0.0-lenient",
)

FAST_CONFIG = MetricsConfig(
    max_text_length=1500,
    max_context_length=2000,
    max_chunk_length=1000,
    llm_max_tokens_claims=512,
    llm_max_tokens_verify=128,
    version="2.0.0-fast",
)
