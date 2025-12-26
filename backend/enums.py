"""
Centralized enum definitions for the RAG Benchmark Hub.
Single source of truth for all enums used across database models and schemas.
"""

from enum import Enum


class SourceType(str, Enum):
    """Source type for raw datasets."""
    UPLOAD = "upload"
    HUGGINGFACE = "huggingface"


class FileType(str, Enum):
    """Supported file types for raw datasets."""
    PDF = "pdf"
    JSON = "json"
    MD = "md"
    TXT = "txt"
    CSV = "csv"


class ProcessingStatus(str, Enum):
    """Processing status for processed datasets."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
