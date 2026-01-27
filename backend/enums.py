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


class EvaluationTaskStatus(str, Enum):
    """Status for evaluation tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelSwitchStatus(str, Enum):
    """Status for model switch operations."""
    PENDING = "pending"
    CHECKING = "checking"       # Checking if model is cached
    DOWNLOADING = "downloading"  # Downloading model (if not cached)
    STOPPING = "stopping"        # Stopping current vLLM container
    STARTING = "starting"        # Starting vLLM with new model
    LOADING = "loading"          # Model loading into GPU memory
    READY = "ready"              # Model ready for inference
    FAILED = "failed"
    CANCELLED = "cancelled"
