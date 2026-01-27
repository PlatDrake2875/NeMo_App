"""Schemas for experiment templates (YAML-based configuration management)."""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.datasets.rag_hub.processing import (
    CleaningConfig,
    LightweightMetadataConfig,
    LLMMetadataConfig,
    ChunkingConfigSchema,
    PreprocessingConfig,
)
from schemas.evaluation import EvalConfigSchema


class MetricsConfig(BaseModel):
    """Configuration for evaluation metrics."""

    enabled_metrics: List[str] = Field(
        default_factory=lambda: [
            "context_precision",
            "precision_at_k",
            "recall_at_k",
        ],
        description="List of metrics to compute",
    )
    compute_confidence_intervals: bool = Field(
        default=False,
        description="Whether to compute bootstrap confidence intervals",
    )
    bootstrap_samples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of bootstrap samples for CI computation",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.8,
        le=0.99,
        description="Confidence level for intervals (e.g., 0.95 for 95%)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled_metrics": ["context_precision", "precision_at_k", "recall_at_k"],
                "compute_confidence_intervals": True,
                "bootstrap_samples": 1000,
                "confidence_level": 0.95,
            }
        }
    )


class ExperimentTemplate(BaseModel):
    """
    Complete experiment configuration template.

    This schema allows saving and loading complete experiment configurations
    as YAML files for reproducibility and sharing.
    """

    # Template metadata
    name: str = Field(..., description="Template name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Template description")
    version: str = Field(default="1.0", description="Template version")
    tags: List[str] = Field(default_factory=list, description="Tags for organization")
    author: Optional[str] = Field(None, description="Template author")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")

    # Configurations (all optional to allow partial templates)
    preprocessing: Optional[PreprocessingConfig] = Field(
        None, description="Preprocessing pipeline configuration"
    )
    evaluation: Optional[EvalConfigSchema] = Field(
        None, description="Evaluation run configuration"
    )
    metrics: Optional[MetricsConfig] = Field(
        None, description="Metrics computation configuration"
    )

    # Reproducibility settings
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility"
    )
    notes: Optional[str] = Field(
        None, description="Additional notes or comments"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "production-quality",
                "description": "Balanced settings for production use",
                "version": "1.0",
                "tags": ["production", "balanced"],
                "preprocessing": {
                    "cleaning": {"enabled": True, "normalize_whitespace": True},
                    "chunking": {"method": "recursive", "chunk_size": 1000},
                },
                "evaluation": {"top_k": 5, "temperature": 0.1},
            }
        }
    )

    def to_yaml(self) -> str:
        """Serialize the template to YAML format."""
        # Use model_dump to get dict, excluding None values for cleaner YAML
        data = self.model_dump(exclude_none=True, exclude_unset=False)
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ExperimentTemplate":
        """Parse a template from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls, file_path: str) -> "ExperimentTemplate":
        """Load a template from a YAML file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    def content_hash(self) -> str:
        """
        Generate a SHA-256 hash of the configuration content.

        This is useful for:
        - Detecting configuration changes
        - Deduplication
        - Reproducibility tracking
        """
        # Only hash the configuration parts, not metadata
        config_dict = {
            "preprocessing": self.preprocessing.model_dump() if self.preprocessing else None,
            "evaluation": self.evaluation.model_dump() if self.evaluation else None,
            "metrics": self.metrics.model_dump() if self.metrics else None,
            "seed": self.seed,
        }
        # Sort keys for consistent hashing
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class TemplateMetadata(BaseModel):
    """Metadata about a saved template (used in listings)."""

    name: str
    description: Optional[str] = None
    version: str = "1.0"
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    content_hash: Optional[str] = None
    file_path: Optional[str] = Field(None, description="Path to the template file")
    is_builtin: bool = Field(False, description="Whether this is a built-in preset")


class TemplateListResponse(BaseModel):
    """Response for listing templates."""

    templates: List[TemplateMetadata]
    total: int


class SaveTemplateRequest(BaseModel):
    """Request to save a new template."""

    template: ExperimentTemplate
    overwrite: bool = Field(
        False, description="Whether to overwrite if template with same name exists"
    )


class SaveTemplateResponse(BaseModel):
    """Response after saving a template."""

    name: str
    file_path: str
    content_hash: str
    created: bool = Field(description="True if new, False if updated")


class ImportTemplateRequest(BaseModel):
    """Request to import a template from YAML content."""

    yaml_content: str = Field(..., description="YAML content of the template")
    name: Optional[str] = Field(
        None, description="Override name (uses name from YAML if not provided)"
    )


class ExportTemplateResponse(BaseModel):
    """Response for exporting a template."""

    yaml_content: str
    filename: str
