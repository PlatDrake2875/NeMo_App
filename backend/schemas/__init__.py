"""
Schemas package - re-exports all schemas for backward compatibility.

This module maintains backward compatibility with the original monolithic schemas.py.
All schemas can be imported directly from `schemas` or from their specific modules.

Example:
    # Legacy import (still works)
    from schemas import ChatRequest, Message, EmbedderConfig

    # New modular import (preferred)
    from schemas.chat import ChatRequest
    from schemas.common import Message
    from schemas.datasets.registry import EmbedderConfig
"""

# Common schemas
from schemas.common import Message, UploadResponse

# Chat schemas
from schemas.chat import (
    LegacyChatRequest,
    RAGChatRequest,
    RAGChatResponse,
    RAGStreamRequest,
    ChatResponseToken,
    HistoryMessage,
    ChatRequest,
    ChatStreamChunk,
)

# Health schemas
from schemas.health import HealthStatusDetail, HealthResponse

# Model schemas
from schemas.models import (
    ModelInfo,
    OllamaModelInfo,
    ModelValidationRequest,
    ModelMetadataResponse,
    ModelDownloadRequest,
    ModelLoadRequest,
)

# Automation schemas
from schemas.automation import AutomateRequest, AutomateResponse

# Dataset schemas
from schemas.datasets import (
    # Registry
    EmbedderConfig,
    DatasetCreateRequest,
    DatasetMetadata,
    DatasetInfo,
    DatasetListResponse,
    # Documents
    DocumentChunk,
    DocumentListResponse,
    # Chunking
    ChunkingMethodInfo,
    ChunkingMethodsResponse,
    RechunkRequest,
    RechunkResponse,
    DocumentSource,
    DocumentSourcesResponse,
    # RAG Hub - Raw
    RawFileInfo,
    RawDatasetCreate,
    RawDatasetInfo,
    RawDatasetListResponse,
    SourceTypeEnum,
    FileTypeEnum,
    # RAG Hub - Processing
    CleaningConfig,
    LLMMetadataConfig,
    ChunkingConfigSchema,
    PreprocessingConfig,
    ProcessingStatusResponse,
    ProcessingStatusEnum,
    # RAG Hub - Processed
    ProcessedDatasetCreate,
    ProcessedDatasetInfo,
    ProcessedDatasetListResponse,
    # RAG Hub - HuggingFace
    HFDatasetConfig,
    HFImportAsRawRequest,
    HFDirectProcessRequest,
    HFColumnInfo,
    HFDatasetMetadata,
    # RAG Hub - Metadata
    ExtractedMetadataInfo,
    BatchUploadFailedFile,
    BatchUploadResponse,
    DatasetStatsResponse,
)

__all__ = [
    # Common
    "Message",
    "UploadResponse",
    # Chat
    "LegacyChatRequest",
    "RAGChatRequest",
    "RAGChatResponse",
    "RAGStreamRequest",
    "ChatResponseToken",
    "HistoryMessage",
    "ChatRequest",
    "ChatStreamChunk",
    # Health
    "HealthStatusDetail",
    "HealthResponse",
    # Models
    "ModelInfo",
    "OllamaModelInfo",
    "ModelValidationRequest",
    "ModelMetadataResponse",
    "ModelDownloadRequest",
    "ModelLoadRequest",
    # Automation
    "AutomateRequest",
    "AutomateResponse",
    # Dataset Registry
    "EmbedderConfig",
    "DatasetCreateRequest",
    "DatasetMetadata",
    "DatasetInfo",
    "DatasetListResponse",
    # Documents
    "DocumentChunk",
    "DocumentListResponse",
    # Chunking
    "ChunkingMethodInfo",
    "ChunkingMethodsResponse",
    "RechunkRequest",
    "RechunkResponse",
    "DocumentSource",
    "DocumentSourcesResponse",
    # RAG Hub - Raw
    "RawFileInfo",
    "RawDatasetCreate",
    "RawDatasetInfo",
    "RawDatasetListResponse",
    "SourceTypeEnum",
    "FileTypeEnum",
    # RAG Hub - Processing
    "CleaningConfig",
    "LLMMetadataConfig",
    "ChunkingConfigSchema",
    "PreprocessingConfig",
    "ProcessingStatusResponse",
    "ProcessingStatusEnum",
    # RAG Hub - Processed
    "ProcessedDatasetCreate",
    "ProcessedDatasetInfo",
    "ProcessedDatasetListResponse",
    # RAG Hub - HuggingFace
    "HFDatasetConfig",
    "HFImportAsRawRequest",
    "HFDirectProcessRequest",
    "HFColumnInfo",
    "HFDatasetMetadata",
    # RAG Hub - Metadata
    "ExtractedMetadataInfo",
    "BatchUploadFailedFile",
    "BatchUploadResponse",
    "DatasetStatsResponse",
]
