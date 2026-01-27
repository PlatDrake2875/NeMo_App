"""Schemas for processed datasets in the RAG Benchmark Hub."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from enums import ProcessingStatus
from schemas.datasets.registry import EmbedderConfig
from schemas.datasets.rag_hub.processing import PreprocessingConfig

# Backward compatibility alias
ProcessingStatusEnum = ProcessingStatus


class ProcessedDatasetCreate(BaseModel):
    """Request to create a processed dataset from a raw dataset."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    raw_dataset_id: int
    # embedder_config is optional for new flow where embedding is deferred to evaluation time
    embedder_config: Optional[EmbedderConfig] = Field(
        default_factory=lambda: EmbedderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_type="huggingface"
        )
    )
    preprocessing_config: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    vector_backend: str = Field(default="pgvector", pattern="^(pgvector|qdrant)$")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "research-papers-semantic-1000",
                "description": "Research papers with semantic chunking",
                "raw_dataset_id": 1,
                "embedder_config": {
                    "model_name": "all-MiniLM-L6-v2",
                    "model_type": "huggingface"
                },
                "preprocessing_config": {
                    "chunking": {"method": "semantic", "chunk_size": 1000}
                },
                "vector_backend": "pgvector"
            }
        }
    )


class ProcessedDatasetInfo(BaseModel):
    """Full information about a processed dataset."""

    id: int
    name: str
    description: Optional[str]
    raw_dataset_id: Optional[int]
    raw_dataset_name: Optional[str] = None
    collection_name: str
    vector_backend: str
    embedder_config: EmbedderConfig
    preprocessing_config: PreprocessingConfig
    processing_status: ProcessingStatusEnum
    processing_error: Optional[str]
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    document_count: int
    chunk_count: int

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "name": "research-papers-semantic-1000",
                "description": "Research papers with semantic chunking",
                "raw_dataset_id": 1,
                "raw_dataset_name": "research-papers",
                "collection_name": "processed_research-papers-semantic-1000",
                "vector_backend": "pgvector",
                "embedder_config": {"model_name": "all-MiniLM-L6-v2"},
                "preprocessing_config": {},
                "processing_status": "completed",
                "processing_error": None,
                "processing_started_at": "2025-01-15T10:00:00Z",
                "processing_completed_at": "2025-01-15T10:30:00Z",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:30:00Z",
                "document_count": 15,
                "chunk_count": 450
            }
        }
    )


class ProcessedDatasetListResponse(BaseModel):
    """Response for listing processed datasets."""

    count: int
    datasets: List[ProcessedDatasetInfo]
