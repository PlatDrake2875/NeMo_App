"""Schemas for HuggingFace dataset integration in the RAG Benchmark Hub."""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.datasets.registry import EmbedderConfig
from schemas.datasets.rag_hub.processing import PreprocessingConfig


class HFDatasetConfig(BaseModel):
    """Configuration for HuggingFace dataset import."""

    dataset_id: str = Field(..., description="HuggingFace dataset ID (e.g., 'squad')")
    subset: Optional[str] = Field(None, description="Dataset subset/configuration")
    split: str = Field(default="train", description="Dataset split (train, test, validation)")
    text_column: str = Field(default="text", description="Column containing text content")
    token: Optional[str] = Field(None, description="HuggingFace token for private datasets")
    max_samples: Optional[int] = Field(None, ge=1, description="Maximum number of samples to import")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dataset_id": "microsoft/wiki_qa",
                "subset": None,
                "split": "train",
                "text_column": "question",
                "token": None,
                "max_samples": 1000
            }
        }
    )


class HFImportAsRawRequest(BaseModel):
    """Request to import HuggingFace dataset as raw dataset."""

    hf_config: HFDatasetConfig
    raw_dataset_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hf_config": {
                    "dataset_id": "squad",
                    "split": "train",
                    "text_column": "context"
                },
                "raw_dataset_name": "squad-contexts",
                "description": "SQuAD context passages"
            }
        }
    )


class HFDirectProcessRequest(BaseModel):
    """Request to directly process HuggingFace dataset to vector store."""

    hf_config: HFDatasetConfig
    processed_dataset_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    embedder_config: EmbedderConfig
    preprocessing_config: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    vector_backend: str = Field(default="pgvector")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hf_config": {
                    "dataset_id": "squad",
                    "split": "train",
                    "text_column": "context"
                },
                "processed_dataset_name": "squad-processed",
                "description": "SQuAD processed for RAG",
                "embedder_config": {"model_name": "all-MiniLM-L6-v2"},
                "vector_backend": "pgvector"
            }
        }
    )


class HFColumnInfo(BaseModel):
    """Information about a single column/feature in a HuggingFace dataset."""

    name: str
    dtype: str


class HFDatasetMetadata(BaseModel):
    """Metadata about a HuggingFace dataset."""

    dataset_id: str
    description: Optional[str]
    size_bytes: Optional[int]
    num_rows: Dict[str, int]  # split -> count
    features: Dict[str, str]  # column -> dtype (kept for backward compatibility)
    columns: List[HFColumnInfo]  # column info as list for frontend consumption
    available_splits: List[str]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dataset_id": "squad",
                "description": "Stanford Question Answering Dataset",
                "size_bytes": 35000000,
                "num_rows": {"train": 87599, "validation": 10570},
                "features": {"context": "string", "question": "string", "answers": "dict"},
                "columns": [
                    {"name": "context", "dtype": "string"},
                    {"name": "question", "dtype": "string"},
                    {"name": "answers", "dtype": "dict"},
                ],
                "available_splits": ["train", "validation"]
            }
        }
    )
