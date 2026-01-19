"""Schemas for raw (unprocessed) datasets in the RAG Benchmark Hub."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from enums import SourceType, FileType


# Backward compatibility aliases
SourceTypeEnum = SourceType
FileTypeEnum = FileType


class RawFileInfo(BaseModel):
    """Information about a raw file."""

    id: int
    filename: str
    file_type: FileTypeEnum
    mime_type: str
    size_bytes: int
    uploaded_at: datetime
    content_hash: str
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "filename": "research_paper.pdf",
                "file_type": "pdf",
                "mime_type": "application/pdf",
                "size_bytes": 1234567,
                "uploaded_at": "2025-01-15T10:30:00Z",
                "content_hash": "abc123def456...",
                "metadata": {"pages": 12}
            }
        }
    )


class RawDatasetCreate(BaseModel):
    """Request to create a new raw dataset."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    source_type: SourceTypeEnum = SourceTypeEnum.UPLOAD

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "research-papers",
                "description": "Collection of ML research papers",
                "source_type": "upload"
            }
        }
    )


class RawDatasetInfo(BaseModel):
    """Full information about a raw dataset."""

    id: int
    name: str
    description: Optional[str]
    source_type: SourceTypeEnum
    source_identifier: Optional[str]
    created_at: datetime
    updated_at: datetime
    total_file_count: int
    total_size_bytes: int
    files: List[RawFileInfo] = []

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "name": "research-papers",
                "description": "Collection of ML research papers",
                "source_type": "upload",
                "source_identifier": None,
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T15:30:00Z",
                "total_file_count": 15,
                "total_size_bytes": 45000000,
                "files": []
            }
        }
    )


class RawDatasetListResponse(BaseModel):
    """Response for listing raw datasets."""

    count: int
    datasets: List[RawDatasetInfo]
