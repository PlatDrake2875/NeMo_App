"""
Dependency injection setup for the FastAPI application.
Provides service instances and database sessions to endpoints.
"""

from functools import lru_cache
from typing import Generator

from sqlalchemy.orm import Session

from database_models import get_db_session
from services.automate import AutomateService
from services.chat import ChatService
from services.document import DocumentService
from services.health import HealthService
from services.model import ModelService
from services.upload import UploadService


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI routes.

    Yields a SQLAlchemy session that is automatically committed on success
    or rolled back on error.

    Example:
        @router.get("/items")
        async def list_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    yield from get_db_session()


@lru_cache
def get_chat_service() -> ChatService:
    """
    Get a ChatService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return ChatService()


@lru_cache
def get_model_service() -> ModelService:
    """
    Get a ModelService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return ModelService()


@lru_cache
def get_upload_service() -> UploadService:
    """
    Get an UploadService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return UploadService()


@lru_cache
def get_document_service() -> DocumentService:
    """
    Get a DocumentService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return DocumentService()


@lru_cache
def get_health_service() -> HealthService:
    """
    Get a HealthService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return HealthService()


@lru_cache
def get_automate_service() -> AutomateService:
    """
    Get an AutomateService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return AutomateService()
