"""Schemas for dataset registry - managing datasets with different embedders."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class EmbedderConfig(BaseModel):
    """Configuration for an embedding model."""

    model_name: str = Field(..., description="Name of the embedding model (e.g., 'all-MiniLM-L6-v2')")
    model_type: str = Field(
        default="huggingface",
        description="Type of embedding model ('huggingface', 'openai', 'custom')"
    )
    dimensions: Optional[int] = Field(
        None,
        description="Embedding dimensions (auto-detected if not provided)"
    )
    model_kwargs: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional kwargs for model initialization"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "all-MiniLM-L6-v2",
                "model_type": "huggingface",
                "dimensions": 384,
                "model_kwargs": {}
            }
        }
    )


class DatasetCreateRequest(BaseModel):
    """Request schema for creating a new dataset."""

    name: str = Field(..., description="Unique name for the dataset", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Description of the dataset")
    embedder_config: EmbedderConfig = Field(..., description="Embedder configuration for this dataset")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "aviation_docs_minilm",
                "description": "Aviation documentation with MiniLM embeddings",
                "embedder_config": {
                    "model_name": "all-MiniLM-L6-v2",
                    "model_type": "huggingface",
                    "dimensions": 384
                }
            }
        }
    )


class DatasetMetadata(BaseModel):
    """Metadata about a dataset."""

    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    chunk_count: int = 0

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "created_at": "2025-11-18T10:00:00Z",
                "updated_at": "2025-11-18T15:30:00Z",
                "document_count": 25,
                "chunk_count": 1250
            }
        }
    )


class DatasetInfo(BaseModel):
    """Information about a dataset."""

    id: int
    name: str
    description: Optional[str]
    collection_name: str
    embedder_config: EmbedderConfig
    metadata: DatasetMetadata

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "name": "aviation_docs_minilm",
                "description": "Aviation documentation with MiniLM embeddings",
                "collection_name": "dataset_aviation_docs_minilm",
                "embedder_config": {
                    "model_name": "all-MiniLM-L6-v2",
                    "model_type": "huggingface",
                    "dimensions": 384
                },
                "metadata": {
                    "created_at": "2025-11-18T10:00:00Z",
                    "updated_at": "2025-11-18T15:30:00Z",
                    "document_count": 25,
                    "chunk_count": 1250
                }
            }
        }
    )


class DatasetListResponse(BaseModel):
    """Response schema for listing datasets."""

    count: int
    datasets: list[DatasetInfo]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "count": 2,
                "datasets": [
                    {
                        "id": 1,
                        "name": "aviation_docs_minilm",
                        "description": "Aviation docs with MiniLM",
                        "collection_name": "dataset_aviation_docs_minilm",
                        "embedder_config": {
                            "model_name": "all-MiniLM-L6-v2",
                            "model_type": "huggingface",
                            "dimensions": 384
                        },
                        "metadata": {
                            "created_at": "2025-11-18T10:00:00Z",
                            "updated_at": "2025-11-18T15:30:00Z",
                            "document_count": 25,
                            "chunk_count": 1250
                        }
                    }
                ]
            }
        }
    )
