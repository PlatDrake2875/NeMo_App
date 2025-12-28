"""Schemas for chunking configuration and operations."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkingMethodInfo(BaseModel):
    """Information about a chunking method."""

    name: str = Field(..., description="Display name of the chunking method")
    description: str = Field(..., description="Description of how this method works")
    default_params: Dict[str, Any] = Field(..., description="Default parameters for this method")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Recursive Character Splitter",
                "description": "Splits text recursively by trying different separators",
                "default_params": {"chunk_size": 1000, "chunk_overlap": 200}
            }
        }
    )


class ChunkingMethodsResponse(BaseModel):
    """Response with all available chunking methods."""

    methods: Dict[str, ChunkingMethodInfo]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "methods": {
                    "recursive": {
                        "name": "Recursive Character Splitter",
                        "description": "Splits text recursively",
                        "default_params": {"chunk_size": 1000, "chunk_overlap": 200}
                    }
                }
            }
        }
    )


class RechunkRequest(BaseModel):
    """Request to re-chunk a document with a new chunking method."""

    dataset_name: str = Field(..., description="Dataset containing the document")
    original_filename: str = Field(..., description="Original filename of the document to re-chunk")
    chunking_method: str = Field(
        default="recursive",
        description="New chunking method to apply: recursive, fixed, or semantic"
    )
    chunk_size: int = Field(default=1000, description="Chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dataset_name": "test_minilm",
                "original_filename": "document.pdf",
                "chunking_method": "semantic",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        }
    )


class RechunkResponse(BaseModel):
    """Response from re-chunking operation."""

    message: str
    original_filename: str
    chunks_created: int
    chunking_method: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Document re-chunked successfully",
                "original_filename": "document.pdf",
                "chunks_created": 42,
                "chunking_method": "semantic"
            }
        }
    )


class DocumentSource(BaseModel):
    """Information about a document source."""

    original_filename: str
    chunk_count: int
    chunking_method: Optional[str] = None
    chunk_size: Optional[int] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "original_filename": "manual.pdf",
                "chunk_count": 125,
                "chunking_method": "recursive",
                "chunk_size": 1000
            }
        }
    )


class DocumentSourcesResponse(BaseModel):
    """Response with list of document sources."""

    count: int
    documents: List[DocumentSource]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "count": 2,
                "documents": [
                    {
                        "original_filename": "manual.pdf",
                        "chunk_count": 125,
                        "chunking_method": "recursive",
                        "chunk_size": 1000
                    }
                ]
            }
        }
    )
