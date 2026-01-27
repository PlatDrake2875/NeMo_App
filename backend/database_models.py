"""
Database models for the dataset registry using SQLAlchemy.
This module defines the database schema for storing dataset configurations and metadata.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Enum,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    DateTime,
    JSON,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from config import POSTGRES_CONNECTION_STRING
from enums import SourceType, FileType, ProcessingStatus, EvaluationTaskStatus

Base = declarative_base()


# Re-export enums for backward compatibility
__all__ = ["SourceType", "FileType", "ProcessingStatus"]


# --- Raw Dataset Models ---
class RawDataset(Base):
    """Container for raw, unprocessed datasets."""
    __tablename__ = "raw_datasets"
    __table_args__ = (
        CheckConstraint("total_file_count >= 0", name="check_file_count_positive"),
        CheckConstraint("total_size_bytes >= 0", name="check_size_positive"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    source_type = Column(String(50), nullable=False, default=SourceType.UPLOAD.value)
    source_identifier = Column(String(500), nullable=True)  # HuggingFace dataset ID or null for uploads

    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    total_file_count = Column(Integer, nullable=False, default=0)
    total_size_bytes = Column(BigInteger, nullable=False, default=0)

    # Relationships
    files = relationship("RawFile", back_populates="dataset", cascade="all, delete-orphan")
    processed_datasets = relationship("ProcessedDataset", back_populates="raw_dataset")

    def __repr__(self):
        return f"<RawDataset(id={self.id}, name='{self.name}', files={self.total_file_count})>"


class RawFile(Base):
    """Individual files within a raw dataset (stored as BLOBs)."""
    __tablename__ = "raw_files"
    __table_args__ = (
        UniqueConstraint("raw_dataset_id", "content_hash", name="uq_file_per_dataset_hash"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_dataset_id = Column(Integer, ForeignKey("raw_datasets.id", ondelete="CASCADE"), nullable=False, index=True)

    # File info
    filename = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    mime_type = Column(String(100), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    file_content = Column(LargeBinary, nullable=False)  # BLOB storage

    # Content hash for deduplication
    content_hash = Column(String(64), nullable=False, index=True)  # SHA-256

    # Metadata
    uploaded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    metadata_json = Column(JSON, nullable=True, default=dict)  # Extensible metadata

    # Relationships
    dataset = relationship("RawDataset", back_populates="files")

    def __repr__(self):
        return f"<RawFile(id={self.id}, filename='{self.filename}', type='{self.file_type}')>"


# --- Processed Dataset Models ---
class ProcessedDataset(Base):
    """Processed/indexed datasets in vector stores."""
    __tablename__ = "processed_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Link to raw dataset (nullable for backward compatibility with legacy datasets)
    raw_dataset_id = Column(Integer, ForeignKey("raw_datasets.id", ondelete="SET NULL"), nullable=True, index=True)

    # Vector store configuration
    collection_name = Column(String(200), unique=True, nullable=False)
    vector_backend = Column(String(50), nullable=False, default="pgvector")  # 'pgvector', 'qdrant'

    # Embedder configuration
    embedder_model_name = Column(String(200), nullable=False)
    embedder_model_type = Column(String(50), nullable=False, default="huggingface")
    embedder_dimensions = Column(Integer, nullable=True)
    embedder_model_kwargs = Column(JSON, nullable=True, default=dict)

    # Preprocessing configuration
    preprocessing_config = Column(JSON, nullable=False, default=dict)
    # Example structure:
    # {
    #     "cleaning": {"enabled": False, "options": {}},
    #     "llm_metadata": {
    #         "enabled": True,
    #         "model": "<model-name>",
    #         "extract_summary": True,
    #         "extract_keywords": True,
    #         "extract_entities": False,
    #         "extract_categories": True
    #     },
    #     "chunking": {
    #         "method": "recursive",
    #         "chunk_size": 1000,
    #         "chunk_overlap": 200
    #     }
    # }

    # Processing status
    processing_status = Column(String(50), nullable=False, default=ProcessingStatus.PENDING.value)
    processing_error = Column(Text, nullable=True)
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    document_count = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)

    # Relationships
    raw_dataset = relationship("RawDataset", back_populates="processed_datasets")

    def __repr__(self):
        return f"<ProcessedDataset(id={self.id}, name='{self.name}', status='{self.processing_status}')>"


class LLMExtractedMetadata(Base):
    """LLM-extracted metadata for raw files."""
    __tablename__ = "llm_extracted_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_file_id = Column(Integer, ForeignKey("raw_files.id", ondelete="CASCADE"), nullable=False, index=True)
    processed_dataset_id = Column(Integer, ForeignKey("processed_datasets.id", ondelete="CASCADE"), nullable=False, index=True)

    # Extraction configuration
    extraction_model = Column(String(200), nullable=False)
    extraction_timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # Extracted fields (all nullable as they depend on config)
    summary = Column(Text, nullable=True)
    keywords = Column(JSON, nullable=True)  # List[str]
    entities = Column(JSON, nullable=True)  # List[Dict] with type and value
    categories = Column(JSON, nullable=True)  # List[str]
    custom_fields = Column(JSON, nullable=True)  # Extensible for custom extractions

    def __repr__(self):
        return f"<LLMExtractedMetadata(id={self.id}, file_id={self.raw_file_id}, model='{self.extraction_model}')>"


class EvaluationTask(Base):
    """Background evaluation tasks with progress tracking."""
    __tablename__ = "evaluation_tasks"

    id = Column(String(36), primary_key=True)  # UUID

    # Task configuration (stored as JSON for flexibility)
    config = Column(JSON, nullable=False)
    # Example: {
    #     "eval_dataset_id": "abc123",
    #     "collection_name": "my_collection",
    #     "use_rag": true,
    #     "use_colbert": true,
    #     "top_k": 5,
    #     "temperature": 0.1,
    #     "embedder": "sentence-transformers/all-MiniLM-L6-v2"
    # }

    # Status tracking
    status = Column(String(20), nullable=False, default=EvaluationTaskStatus.PENDING.value)

    # Progress tracking
    current_pair = Column(Integer, nullable=False, default=0)
    total_pairs = Column(Integer, nullable=False, default=0)
    current_step = Column(String(200), nullable=True)  # Human-readable current action

    # Timing
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Results
    result_run_id = Column(String(36), nullable=True)  # Links to saved evaluation run
    error_message = Column(Text, nullable=True)

    # Metadata
    eval_dataset_name = Column(String(200), nullable=True)
    collection_display_name = Column(String(200), nullable=True)

    def __repr__(self):
        return f"<EvaluationTask(id={self.id}, status='{self.status}', progress={self.current_pair}/{self.total_pairs})>"

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_pairs == 0:
            return 0.0
        return round((self.current_pair / self.total_pairs) * 100, 1)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "status": self.status,
            "current_pair": self.current_pair,
            "total_pairs": self.total_pairs,
            "pair_count": self.total_pairs,  # Alias for frontend compatibility
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_run_id": self.result_run_id,
            "error_message": self.error_message,
            "eval_dataset_name": self.eval_dataset_name,
            "collection_display_name": self.collection_display_name,
            "config": self.config,
        }


class Dataset(Base):
    """SQLAlchemy model for the datasets table."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    collection_name = Column(String(200), unique=True, nullable=False)

    # Embedder configuration stored as JSON
    embedder_model_name = Column(String(200), nullable=False)
    embedder_model_type = Column(String(50), nullable=False, default="huggingface")
    embedder_dimensions = Column(Integer, nullable=True)
    embedder_model_kwargs = Column(JSON, nullable=True, default={})

    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    document_count = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)

    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', collection='{self.collection_name}')>"


# Database connection and session management
def get_engine():
    """Create and return a SQLAlchemy engine."""
    return create_engine(POSTGRES_CONNECTION_STRING, echo=False)


def get_session_maker():
    """Create and return a session maker."""
    engine = get_engine()
    return sessionmaker(bind=engine)


def init_database():
    """Initialize the database by creating all tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully")


def get_db_session():
    """Get a new database session (context manager compatible)."""
    SessionLocal = get_session_maker()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
