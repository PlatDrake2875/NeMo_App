# backend/routers/config_router.py
"""
Configuration router - exposes application configuration to the frontend.
"""
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    RAG_ENABLED,
    VECTOR_STORE_BACKEND,
    logger,
)
from services.runtime_config import get_runtime_config

router = APIRouter(
    prefix="/config",
    tags=["config"],
)


class QdrantUrlRequest(BaseModel):
    """Request model for updating Qdrant URL."""

    qdrant_url: str


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


@router.get("/qdrant")
async def get_qdrant_config():
    """
    Get current Qdrant configuration.
    """
    runtime_config = get_runtime_config()

    if runtime_config.has_qdrant_override():
        config = runtime_config.get_qdrant_config()
        source = "runtime"
    else:
        config = {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "https": False,
            "url": f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        }
        source = "environment"

    return {"config": config, "source": source}


@router.post("/qdrant")
async def update_qdrant_config(request: QdrantUrlRequest):
    """
    Update Qdrant connection URL at runtime.

    This will reconfigure the Qdrant client to connect to the specified URL.
    """
    from services.qdrant_vectorstore import reset_qdrant_service
    from vectorstore_factory import VectorStoreFactory

    try:
        runtime_config = get_runtime_config()
        config = runtime_config.set_qdrant_url(request.qdrant_url)

        # Reset existing Qdrant clients to use new config
        VectorStoreFactory.reset_qdrant_client()
        reset_qdrant_service()

        return {"status": "success", "message": "Qdrant configuration updated", "config": config}
    except Exception as e:
        logger.error(f"Failed to update Qdrant config: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/test-qdrant")
async def test_qdrant_connection(request: QdrantUrlRequest):
    """
    Test connection to a Qdrant instance without updating config.
    """
    try:
        parsed = urlparse(request.qdrant_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 6333)
        https = parsed.scheme == "https"

        if https:
            url = f"https://{host}:{port}" if port != 443 else f"https://{host}"
            test_client = QdrantClient(
                url=url,
                api_key=QDRANT_API_KEY,
                prefer_grpc=False,
                timeout=5,
            )
        else:
            test_client = QdrantClient(
                host=host,
                port=port,
                api_key=QDRANT_API_KEY,
                timeout=5,
            )

        # Try to list collections as a connectivity test
        collections = test_client.get_collections()
        test_client.close()

        return {
            "connected": True,
            "message": f"Successfully connected to Qdrant at {request.qdrant_url}",
            "collections_count": len(collections.collections),
        }
    except Exception as e:
        logger.error(f"Qdrant connection test failed: {e}")
        return {"connected": False, "message": f"Connection failed: {str(e)}"}
