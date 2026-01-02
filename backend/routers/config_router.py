# backend/routers/config_router.py
"""
Configuration router - exposes application configuration to the frontend.
"""

from fastapi import APIRouter

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    RAG_ENABLED,
    VECTOR_STORE_BACKEND,
)

router = APIRouter(
    prefix="/config",
    tags=["config"],
)


@router.get("")
async def get_config():
    """
    Get current application configuration.

    Returns configuration that's useful for the frontend to display
    and adapt its behavior.
    """
    return {
        "vector_store": {
            "backend": VECTOR_STORE_BACKEND,
            "collection_name": COLLECTION_NAME,
            # Include connection info for display (not sensitive)
            "qdrant_host": QDRANT_HOST if VECTOR_STORE_BACKEND == "qdrant" else None,
            "qdrant_port": QDRANT_PORT if VECTOR_STORE_BACKEND == "qdrant" else None,
        },
        "embedding": {
            "model_name": EMBEDDING_MODEL_NAME,
        },
        "rag_enabled": RAG_ENABLED,
    }


@router.get("/vector-store")
async def get_vector_store_config():
    """
    Get current vector store configuration.

    Returns:
        Current vector store backend and related settings.
    """
    return {
        "backend": VECTOR_STORE_BACKEND,
        "collection_name": COLLECTION_NAME,
        "is_qdrant": VECTOR_STORE_BACKEND == "qdrant",
        "is_pgvector": VECTOR_STORE_BACKEND == "pgvector",
    }
