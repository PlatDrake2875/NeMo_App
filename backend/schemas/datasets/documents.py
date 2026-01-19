"""Schemas for document chunks from vector stores."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """Represents a single document chunk retrieved from the vector store."""

    id: str
    content: str = Field(..., alias="page_content")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class DocumentListResponse(BaseModel):
    """Response model for listing all document chunks."""

    count: int
    documents: list[DocumentChunk]
