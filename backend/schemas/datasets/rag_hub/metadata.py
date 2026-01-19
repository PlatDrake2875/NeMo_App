"""Schemas for metadata and statistics in the RAG Benchmark Hub."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from schemas.datasets.rag_hub.raw import RawFileInfo


class ExtractedMetadataInfo(BaseModel):
    """LLM-extracted metadata for a file."""

    id: int
    raw_file_id: int
    filename: Optional[str] = None
    extraction_model: str
    extraction_timestamp: datetime
    summary: Optional[str]
    keywords: Optional[List[str]]
    entities: Optional[List[Dict[str, str]]]
    categories: Optional[List[str]]
    custom_fields: Optional[Dict[str, Any]]

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "raw_file_id": 5,
                "filename": "paper.pdf",
                "extraction_model": "meta-llama/Llama-3.2-3B-Instruct",
                "extraction_timestamp": "2025-01-15T10:30:00Z",
                "summary": "This paper presents a novel approach to...",
                "keywords": ["machine learning", "neural networks", "optimization"],
                "entities": [{"type": "person", "value": "John Smith"}, {"type": "org", "value": "MIT"}],
                "categories": ["Computer Science", "Artificial Intelligence"]
            }
        }
    )


class BatchUploadFailedFile(BaseModel):
    """Information about a file that failed to upload."""

    filename: str
    error: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "bad_file.xyz",
                "error": "Unsupported file type"
            }
        }
    )


class BatchUploadResponse(BaseModel):
    """Response for batch file upload."""

    successful: List[RawFileInfo]
    failed: List[BatchUploadFailedFile]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "successful": [],
                "failed": [{"filename": "bad.xyz", "error": "Unsupported type"}]
            }
        }
    )


class DatasetStatsResponse(BaseModel):
    """Statistics about datasets."""

    total_raw_datasets: int
    total_processed_datasets: int
    total_raw_files: int
    total_storage_bytes: int
    processing_in_progress: int
    datasets_by_backend: Dict[str, int]
    datasets_by_chunking_method: Dict[str, int]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_raw_datasets": 5,
                "total_processed_datasets": 12,
                "total_raw_files": 150,
                "total_storage_bytes": 500000000,
                "processing_in_progress": 2,
                "datasets_by_backend": {"pgvector": 8, "qdrant": 4},
                "datasets_by_chunking_method": {"recursive": 6, "semantic": 4, "fixed": 2}
            }
        }
    )
