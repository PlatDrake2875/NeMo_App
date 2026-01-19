"""Dataset-related schemas."""

from schemas.datasets.registry import (
    EmbedderConfig,
    DatasetCreateRequest,
    DatasetMetadata,
    DatasetInfo,
    DatasetListResponse,
)
from schemas.datasets.documents import (
    DocumentChunk,
    DocumentListResponse,
)
from schemas.datasets.chunking import (
    ChunkingMethodInfo,
    ChunkingMethodsResponse,
    RechunkRequest,
    RechunkResponse,
    DocumentSource,
    DocumentSourcesResponse,
)
from schemas.datasets.rag_hub import (
    # Raw
    RawFileInfo,
    RawDatasetCreate,
    RawDatasetInfo,
    RawDatasetListResponse,
    SourceTypeEnum,
    FileTypeEnum,
    # Processing
    CleaningConfig,
    LightweightMetadataConfig,
    LLMMetadataConfig,
    ChunkingConfigSchema,
    PreprocessingConfig,
    ProcessingStatusResponse,
    ProcessingStatusEnum,
    # Processed
    ProcessedDatasetCreate,
    ProcessedDatasetInfo,
    ProcessedDatasetListResponse,
    # HuggingFace
    HFDatasetConfig,
    HFImportAsRawRequest,
    HFDirectProcessRequest,
    HFColumnInfo,
    HFDatasetMetadata,
    # Metadata
    ExtractedMetadataInfo,
    BatchUploadFailedFile,
    BatchUploadResponse,
    DatasetStatsResponse,
)

__all__ = [
    # Registry
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
    "LightweightMetadataConfig",
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
