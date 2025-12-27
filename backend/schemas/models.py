"""Schemas for model management and HuggingFace model operations."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    modified_at: str
    size: int

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> Optional["ModelInfo"]:
        """Create ModelInfo from OpenAI-compatible API response."""
        model_name = raw.get("id")
        if not model_name:
            return None

        created_timestamp = raw.get("created", 0)
        modified_at_str = "N/A"
        if created_timestamp:
            try:
                dt_obj = datetime.fromtimestamp(created_timestamp)
                modified_at_str = dt_obj.isoformat()
            except (ValueError, OSError):
                modified_at_str = "N/A"

        return cls(
            name=model_name,
            modified_at=modified_at_str,
            size=0  # vLLM API doesn't expose model size
        )


# Backward compatibility alias - DEPRECATED: Use ModelInfo instead
OllamaModelInfo = ModelInfo


class ModelValidationRequest(BaseModel):
    """Request to validate a HuggingFace model."""

    model_id: str = Field(
        ..., description="HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B')"
    )
    token: Optional[str] = Field(None, description="HuggingFace API token for gated models")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "token": "hf_..."
            }
        }
    )


class ModelMetadataResponse(BaseModel):
    """Response containing model metadata."""

    model_id: str = Field(..., description="HuggingFace model ID")
    is_gated: bool = Field(..., description="Whether the model requires authentication")
    size_bytes: int = Field(..., description="Model size in bytes")
    size_gb: float = Field(..., description="Model size in GB")
    downloads: int = Field(..., description="Number of downloads on HuggingFace")
    pipeline_tag: Optional[str] = Field(None, description="Model pipeline tag")
    tags: list[str] = Field(default_factory=list, description="Model tags")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "is_gated": True,
                "size_bytes": 16000000000,
                "size_gb": 14.9,
                "downloads": 1000000,
                "pipeline_tag": "text-generation",
                "tags": ["llama", "instruct", "chat"]
            }
        }
    )


class ModelDownloadRequest(BaseModel):
    """Request to download a model from HuggingFace."""

    model_id: str = Field(
        ..., description="HuggingFace model ID to download"
    )
    token: Optional[str] = Field(
        None, description="HuggingFace API token for gated models"
    )
    custom_name: Optional[str] = Field(
        None, description="Custom name for the model (defaults to model_id)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "token": "hf_...",
                "custom_name": "llama-3.1-8b"
            }
        }
    )


class ModelLoadRequest(BaseModel):
    """Request to load a downloaded model into vLLM."""

    model_id: str = Field(..., description="HuggingFace model ID to load")
    served_model_name: Optional[str] = Field(
        None, description="Name to serve the model as (defaults to model_id)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "served_model_name": "llama-3.1-8b"
            }
        }
    )
