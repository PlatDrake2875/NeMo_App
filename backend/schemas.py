# backend/schemas.py
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Generic Message Model (useful for conversation histories) ---
class Message(BaseModel):
    role: str = Field(
        ...,
        description="Role of the message sender (e.g., 'user', 'assistant', 'system')",
    )
    content: str = Field(..., description="Content of the message")

    class Config:
        json_schema_extra = {
            "example": {"role": "user", "content": "Hello, how are you?"}
        }


# --- Pydantic Models (Existing) ---
class ModelInfo(BaseModel):
    name: str
    modified_at: str  # Storing as string
    size: int

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> Optional["ModelInfo"]:
        # vLLM OpenAI-compatible API returns: {"id": "model-name", "object": "model", ...}
        model_name = raw.get("id")
        if not model_name:
            return None

        # vLLM doesn't provide modified_at or size in the API response
        # We'll use placeholder values
        created_timestamp = raw.get("created", 0)
        modified_at_str = "N/A"
        if created_timestamp:
            try:
                dt_obj = datetime.fromtimestamp(created_timestamp)
                modified_at_str = dt_obj.isoformat()
            except (ValueError, OSError):
                modified_at_str = "N/A"

        return cls(
            name=model_name,
            modified_at=modified_at_str,
            size=0  # vLLM API doesn't expose model size
        )


# Backward compatibility alias - DEPRECATED: Use ModelInfo instead
# This will be removed in a future version
OllamaModelInfo = ModelInfo


# --- Model Download and Validation Schemas ---
class ModelValidationRequest(BaseModel):
    """Request to validate a HuggingFace model."""

    model_id: str = Field(
        ..., description="HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B')"
    )
    token: Optional[str] = Field(None, description="HuggingFace API token for gated models")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "token": "hf_..."
            }
        }


class ModelMetadataResponse(BaseModel):
    """Response containing model metadata."""

    model_id: str = Field(..., description="HuggingFace model ID")
    is_gated: bool = Field(..., description="Whether the model requires authentication")
    size_bytes: int = Field(..., description="Model size in bytes")
    size_gb: float = Field(..., description="Model size in GB")
    downloads: int = Field(..., description="Number of downloads on HuggingFace")
    pipeline_tag: Optional[str] = Field(None, description="Model pipeline tag")
    tags: list[str] = Field(default_factory=list, description="Model tags")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "is_gated": True,
                "size_bytes": 16000000000,
                "size_gb": 14.9,
                "downloads": 1000000,
                "pipeline_tag": "text-generation",
                "tags": ["llama", "instruct", "chat"]
            }
        }


class ModelDownloadRequest(BaseModel):
    """Request to download a model from HuggingFace."""

    model_id: str = Field(
        ..., description="HuggingFace model ID to download"
    )
    token: Optional[str] = Field(
        None, description="HuggingFace API token for gated models"
    )
    custom_name: Optional[str] = Field(
        None, description="Custom name for the model (defaults to model_id)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "token": "hf_...",
                "custom_name": "llama-3.1-8b"
            }
        }


class ModelLoadRequest(BaseModel):
    """Request to load a downloaded model into vLLM."""

    model_id: str = Field(..., description="HuggingFace model ID to load")
    served_model_name: Optional[str] = Field(
        None, description="Name to serve the model as (defaults to model_id)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "served_model_name": "llama-3.1-8b"
            }
        }


class LegacyChatRequest(BaseModel):
    query: str
    model: Optional[str] = None  # Made model optional to align with usage


class RAGChatRequest(BaseModel):
    session_id: str
    prompt: str
    # Consider using List[Message] here if it fits your history structure
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Conversation history as a list of role/content dicts.",
    )
    use_rag: bool = True


class RAGChatResponse(BaseModel):
    session_id: str
    response: str


class RAGStreamRequest(BaseModel):
    session_id: str
    prompt: str
    # Consider using List[Message] here
    history: list[dict[str, str]] = Field(
        default_factory=list, description="Conversation history for RAG stream."
    )
    use_rag: bool = True


# Model for the response from /api/chat (streaming uses dicts, but this can be for non-streaming if needed)
class ChatResponseToken(BaseModel):
    token: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    message: str
    filename: str  # Assuming one file per response, adjust if multiple
    chunks_added: Optional[int] = None


class HealthStatusDetail(BaseModel):
    status: str
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    vllm: HealthStatusDetail
    postgres: HealthStatusDetail


# --- Schemas for Document Chunks (Existing) ---
class DocumentChunk(BaseModel):
    """Represents a single document chunk retrieved from the vector store."""

    id: str
    content: str = Field(..., alias="page_content")  # Use alias if field name differs
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )  # Ensure default is a factory

    class Config:
        # Correct Pydantic V2 config key
        populate_by_name = True


class DocumentListResponse(BaseModel):
    """Response model for listing all document chunks."""

    count: int
    documents: list[DocumentChunk]


# --- Schemas for Automation Endpoint ---
class AutomateRequest(BaseModel):
    conversation_history: list[Message] = Field(
        ..., description="The current conversation history to be automated."
    )
    model: str = Field(..., description="The model to use for the automation task.")
    automation_task: Optional[str] = Field(
        None,
        description="Specific automation task to perform (e.g., 'summarize', 'generate_next_steps').",
    )
    config_params: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional configuration parameters for automation.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "We discussed the project timeline and deliverables.",
                    },
                    {
                        "role": "assistant",
                        "content": "Okay, I've noted that. The key deliverables are X, Y, and Z due by next Friday.",
                    },
                    {
                        "role": "user",
                        "content": "Correct. Also, remember to schedule the follow-up meeting.",
                    },
                ],
                "model": "llama3:latest",  # Example model
                "automation_task": "generate_meeting_summary_and_actions",
                "config_params": {"max_summary_length": 200},
            }
        }


class AutomateResponse(BaseModel):
    status: str = Field(
        ..., description="Status of the automation request (e.g., 'success', 'error')."
    )
    message: Optional[str] = Field(
        None, description="A message providing details about the outcome."
    )
    data: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Output data from the automation process."
    )
    error_details: Optional[str] = Field(
        None, description="Details if an error occurred."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Conversation automated successfully.",
                "data": {
                    "summary": "The project timeline and deliverables were discussed. Key items are X, Y, Z due next Friday. A follow-up meeting needs to be scheduled.",
                    "action_items": ["Schedule follow-up meeting."],
                },
            }
        }


# --- Chat Models ---
class HistoryMessage(BaseModel):
    """Message in chat history with sender and text fields."""

    sender: str = Field(..., description="Sender of the message ('user' or 'bot')")
    text: str = Field(..., description="Content of the message")

    class Config:
        json_schema_extra = {
            "example": {"sender": "user", "text": "Hello, how are you?"}
        }


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    query: str = Field(..., description="User's question or message")
    model: Optional[str] = Field(None, description="Model to use for the response")
    agent_name: Optional[str] = Field(
        None, description="Name of the NeMo Guardrails agent to use"
    )
    history: Optional[list[HistoryMessage]] = Field(
        default=[], description="Previous conversation history"
    )
    use_rag: Optional[bool] = Field(
        None, description="Whether to use RAG for this request"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the weather like today?",
                "model": "gemma3:4b-it-q4_K_M",
                "agent_name": "math_assistant",
                "history": [
                    {"sender": "user", "text": "Hello"},
                    {"sender": "bot", "text": "Hi there! How can I help you?"},
                ],
                "use_rag": True,
            }
        }


class ChatStreamChunk(BaseModel):
    """Individual chunk in a streaming chat response."""

    token: Optional[str] = Field(None, description="Text token being streamed")
    error: Optional[str] = Field(
        None, description="Error message if something went wrong"
    )
    status: Optional[str] = Field(None, description="Status information")

    class Config:
        json_schema_extra = {"example": {"token": "Hello, I'm doing well!"}}


# --- Dataset Registry Schemas ---
class EmbedderConfig(BaseModel):
    """Configuration for an embedding model."""

    model_name: str = Field(..., description="Name of the embedding model (e.g., 'all-MiniLM-L6-v2')")
    model_type: str = Field(
        default="huggingface",
        description="Type of embedding model ('huggingface', 'openai', 'custom')"
    )
    dimensions: Optional[int] = Field(
        None,
        description="Embedding dimensions (auto-detected if not provided)"
    )
    model_kwargs: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional kwargs for model initialization"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "all-MiniLM-L6-v2",
                "model_type": "huggingface",
                "dimensions": 384,
                "model_kwargs": {}
            }
        }


class DatasetCreateRequest(BaseModel):
    """Request schema for creating a new dataset."""

    name: str = Field(..., description="Unique name for the dataset", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Description of the dataset")
    embedder_config: EmbedderConfig = Field(..., description="Embedder configuration for this dataset")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "aviation_docs_minilm",
                "description": "Aviation documentation with MiniLM embeddings",
                "embedder_config": {
                    "model_name": "all-MiniLM-L6-v2",
                    "model_type": "huggingface",
                    "dimensions": 384
                }
            }
        }


class DatasetMetadata(BaseModel):
    """Metadata about a dataset."""

    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    chunk_count: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "created_at": "2025-11-18T10:00:00Z",
                "updated_at": "2025-11-18T15:30:00Z",
                "document_count": 25,
                "chunk_count": 1250
            }
        }


class DatasetInfo(BaseModel):
    """Information about a dataset."""

    id: int
    name: str
    description: Optional[str]
    collection_name: str
    embedder_config: EmbedderConfig
    metadata: DatasetMetadata

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "aviation_docs_minilm",
                "description": "Aviation documentation with MiniLM embeddings",
                "collection_name": "dataset_aviation_docs_minilm",
                "embedder_config": {
                    "model_name": "all-MiniLM-L6-v2",
                    "model_type": "huggingface",
                    "dimensions": 384
                },
                "metadata": {
                    "created_at": "2025-11-18T10:00:00Z",
                    "updated_at": "2025-11-18T15:30:00Z",
                    "document_count": 25,
                    "chunk_count": 1250
                }
            }
        }


class DatasetListResponse(BaseModel):
    """Response schema for listing datasets."""

    count: int
    datasets: list[DatasetInfo]

    class Config:
        json_schema_extra = {
            "example": {
                "count": 2,
                "datasets": [
                    {
                        "id": 1,
                        "name": "aviation_docs_minilm",
                        "description": "Aviation docs with MiniLM",
                        "collection_name": "dataset_aviation_docs_minilm",
                        "embedder_config": {
                            "model_name": "all-MiniLM-L6-v2",
                            "model_type": "huggingface",
                            "dimensions": 384
                        },
                        "metadata": {
                            "created_at": "2025-11-18T10:00:00Z",
                            "updated_at": "2025-11-18T15:30:00Z",
                            "document_count": 25,
                            "chunk_count": 1250
                        }
                    }
                ]
            }
        }


# --- Chunking Configuration Schemas ---
class ChunkingMethodInfo(BaseModel):
    """Information about a chunking method."""

    name: str = Field(..., description="Display name of the chunking method")
    description: str = Field(..., description="Description of how this method works")
    default_params: Dict[str, Any] = Field(..., description="Default parameters for this method")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Recursive Character Splitter",
                "description": "Splits text recursively by trying different separators",
                "default_params": {"chunk_size": 1000, "chunk_overlap": 200}
            }
        }


class ChunkingMethodsResponse(BaseModel):
    """Response with all available chunking methods."""

    methods: Dict[str, ChunkingMethodInfo]

    class Config:
        json_schema_extra = {
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

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "test_minilm",
                "original_filename": "document.pdf",
                "chunking_method": "semantic",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        }


class RechunkResponse(BaseModel):
    """Response from re-chunking operation."""

    message: str
    original_filename: str
    chunks_created: int
    chunking_method: str

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Document re-chunked successfully",
                "original_filename": "document.pdf",
                "chunks_created": 42,
                "chunking_method": "semantic"
            }
        }


class DocumentSource(BaseModel):
    """Information about a document source."""

    original_filename: str
    chunk_count: int
    chunking_method: Optional[str] = None
    chunk_size: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "original_filename": "manual.pdf",
                "chunk_count": 125,
                "chunking_method": "recursive",
                "chunk_size": 1000
            }
        }


class DocumentSourcesResponse(BaseModel):
    """Response with list of document sources."""

    count: int
    documents: List[DocumentSource]

    class Config:
        json_schema_extra = {
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


# =============================================================================
# RAG BENCHMARK HUB SCHEMAS
# =============================================================================

from enum import Enum as PyEnum


# --- Enums for RAG Hub ---
class SourceTypeEnum(str, PyEnum):
    """Source type for raw datasets."""
    UPLOAD = "upload"
    HUGGINGFACE = "huggingface"


class FileTypeEnum(str, PyEnum):
    """Supported file types for raw datasets."""
    PDF = "pdf"
    JSON = "json"
    MD = "md"
    TXT = "txt"
    CSV = "csv"


class ProcessingStatusEnum(str, PyEnum):
    """Processing status for processed datasets."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Raw Dataset Schemas ---
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

    class Config:
        from_attributes = True
        json_schema_extra = {
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


class RawDatasetCreate(BaseModel):
    """Request to create a new raw dataset."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    source_type: SourceTypeEnum = SourceTypeEnum.UPLOAD

    class Config:
        json_schema_extra = {
            "example": {
                "name": "research-papers",
                "description": "Collection of ML research papers",
                "source_type": "upload"
            }
        }


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

    class Config:
        from_attributes = True
        json_schema_extra = {
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


class RawDatasetListResponse(BaseModel):
    """Response for listing raw datasets."""
    count: int
    datasets: List[RawDatasetInfo]


# --- Preprocessing Configuration Schemas ---
class CleaningConfig(BaseModel):
    """Configuration for document cleaning."""
    enabled: bool = False
    remove_headers_footers: bool = False
    remove_page_numbers: bool = False
    normalize_whitespace: bool = True
    custom_patterns: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "remove_headers_footers": True,
                "remove_page_numbers": True,
                "normalize_whitespace": True,
                "custom_patterns": []
            }
        }


class LLMMetadataConfig(BaseModel):
    """Configuration for LLM metadata extraction."""
    enabled: bool = False
    model: str = Field(default="meta-llama/Llama-3.2-3B-Instruct")
    extract_summary: bool = True
    extract_keywords: bool = True
    extract_entities: bool = True
    extract_categories: bool = True
    max_summary_length: int = Field(default=200, ge=50, le=1000)
    max_keywords: int = Field(default=10, ge=1, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "extract_summary": True,
                "extract_keywords": True,
                "extract_entities": True,
                "extract_categories": True,
                "max_summary_length": 200,
                "max_keywords": 10
            }
        }


class ChunkingConfigSchema(BaseModel):
    """Configuration for document chunking."""
    method: str = Field(default="recursive", description="recursive, fixed, or semantic")
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    # Semantic chunking specific
    embedder_model: Optional[str] = None
    breakpoint_threshold_type: str = Field(default="percentile")

    class Config:
        json_schema_extra = {
            "example": {
                "method": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "embedder_model": None,
                "breakpoint_threshold_type": "percentile"
            }
        }


class PreprocessingConfig(BaseModel):
    """Complete preprocessing configuration."""
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    llm_metadata: LLMMetadataConfig = Field(default_factory=LLMMetadataConfig)
    chunking: ChunkingConfigSchema = Field(default_factory=ChunkingConfigSchema)

    class Config:
        json_schema_extra = {
            "example": {
                "cleaning": {"enabled": False},
                "llm_metadata": {"enabled": True, "extract_summary": True},
                "chunking": {"method": "recursive", "chunk_size": 1000}
            }
        }


# --- Processed Dataset Schemas ---
class ProcessedDatasetCreate(BaseModel):
    """Request to create a processed dataset from a raw dataset."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    raw_dataset_id: int
    embedder_config: EmbedderConfig
    preprocessing_config: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    vector_backend: str = Field(default="pgvector", pattern="^(pgvector|qdrant)$")

    class Config:
        json_schema_extra = {
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

    class Config:
        from_attributes = True
        json_schema_extra = {
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


class ProcessedDatasetListResponse(BaseModel):
    """Response for listing processed datasets."""
    count: int
    datasets: List[ProcessedDatasetInfo]


class ProcessingStatusResponse(BaseModel):
    """Response for processing status check."""
    status: ProcessingStatusEnum
    progress_percent: Optional[float] = None
    current_step: Optional[str] = None
    error: Optional[str] = None


# --- HuggingFace Dataset Schemas ---
class HFDatasetConfig(BaseModel):
    """Configuration for HuggingFace dataset import."""
    dataset_id: str = Field(..., description="HuggingFace dataset ID (e.g., 'squad')")
    subset: Optional[str] = Field(None, description="Dataset subset/configuration")
    split: str = Field(default="train", description="Dataset split (train, test, validation)")
    text_column: str = Field(default="text", description="Column containing text content")
    token: Optional[str] = Field(None, description="HuggingFace token for private datasets")
    max_samples: Optional[int] = Field(None, ge=1, description="Maximum number of samples to import")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "microsoft/wiki_qa",
                "subset": None,
                "split": "train",
                "text_column": "question",
                "token": None,
                "max_samples": 1000
            }
        }


class HFImportAsRawRequest(BaseModel):
    """Request to import HuggingFace dataset as raw dataset."""
    hf_config: HFDatasetConfig
    raw_dataset_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None

    class Config:
        json_schema_extra = {
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


class HFDirectProcessRequest(BaseModel):
    """Request to directly process HuggingFace dataset to vector store."""
    hf_config: HFDatasetConfig
    processed_dataset_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    embedder_config: EmbedderConfig
    preprocessing_config: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    vector_backend: str = Field(default="pgvector")

    class Config:
        json_schema_extra = {
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


class HFDatasetMetadata(BaseModel):
    """Metadata about a HuggingFace dataset."""
    dataset_id: str
    description: Optional[str]
    size_bytes: Optional[int]
    num_rows: Dict[str, int]  # split -> count
    features: Dict[str, str]  # column -> dtype
    available_splits: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "squad",
                "description": "Stanford Question Answering Dataset",
                "size_bytes": 35000000,
                "num_rows": {"train": 87599, "validation": 10570},
                "features": {"context": "string", "question": "string", "answers": "dict"},
                "available_splits": ["train", "validation"]
            }
        }


# --- LLM Extracted Metadata Schemas ---
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

    class Config:
        from_attributes = True
        json_schema_extra = {
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


# --- Dataset Statistics Schemas ---
class DatasetStatsResponse(BaseModel):
    """Statistics about datasets."""
    total_raw_datasets: int
    total_processed_datasets: int
    total_raw_files: int
    total_storage_bytes: int
    processing_in_progress: int
    datasets_by_backend: Dict[str, int]
    datasets_by_chunking_method: Dict[str, int]

    class Config:
        json_schema_extra = {
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
