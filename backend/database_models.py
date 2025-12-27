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
from enums import SourceType, FileType, ProcessingStatus

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
