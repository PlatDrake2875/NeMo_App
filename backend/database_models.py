"""
Database models for the dataset registry using SQLAlchemy.
This module defines the database schema for storing dataset configurations and metadata.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import POSTGRES_CONNECTION_STRING

Base = declarative_base()


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
    print("Database tables created successfully!")


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
