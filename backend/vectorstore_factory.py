# backend/vectorstore_factory.py
"""
Vector store factory for creating different vector store backends.
Supports PGVector (PostgreSQL) and Qdrant.
"""

from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector as LangchainPGVector
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import (
    COLLECTION_NAME,
    POSTGRES_CONNECTION_STRING,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_PREFER_GRPC,
    VECTOR_STORE_BACKEND,
    logger,
)


class VectorStoreFactory:
    """Factory class for creating vector store instances."""

    _qdrant_client: Optional[QdrantClient] = None

    @classmethod
    def reset_qdrant_client(cls):
        """Reset the Qdrant client to force recreation with new config."""
        if cls._qdrant_client is not None:
            try:
                cls._qdrant_client.close()
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {e}")
            cls._qdrant_client = None
            logger.info("Qdrant client reset - will reconnect with new config on next use")

    @classmethod
    def get_qdrant_client(cls) -> QdrantClient:
        """Get or create a shared Qdrant client instance."""
        from services.runtime_config import get_runtime_config

        if cls._qdrant_client is None:
            runtime_config = get_runtime_config()

            # Use runtime config if available, otherwise fall back to env vars
            if runtime_config.has_qdrant_override():
                host = runtime_config.qdrant_host
                port = runtime_config.qdrant_port
                https = runtime_config.qdrant_https

                if https:
                    # For HTTPS URLs, use url parameter instead of host/port
                    url = f"https://{host}:{port}" if port != 443 else f"https://{host}"
                    cls._qdrant_client = QdrantClient(
                        url=url,
                        api_key=QDRANT_API_KEY,
                        prefer_grpc=False,  # Use HTTP for HTTPS connections
                    )
                else:
                    cls._qdrant_client = QdrantClient(
                        host=host,
                        port=port,
                        api_key=QDRANT_API_KEY,
                        prefer_grpc=QDRANT_PREFER_GRPC,
                    )
                logger.info(
                    f"Created Qdrant client with runtime config: {host}:{port} (HTTPS: {https})"
                )
            else:
                # Use default env var config
                cls._qdrant_client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    api_key=QDRANT_API_KEY,
                    prefer_grpc=QDRANT_PREFER_GRPC,
                )
                logger.info(f"Created Qdrant client connecting to {QDRANT_HOST}:{QDRANT_PORT}")
        return cls._qdrant_client

    @classmethod
    def create_vectorstore(
        cls,
        embedding_function: Embeddings,
        collection_name: Optional[str] = None,
        backend: Optional[str] = None,
        async_mode: bool = False,
    ) -> Any:
        """
        Create a vector store instance based on the configured backend.

        Args:
            embedding_function: The embedding function to use
            collection_name: Name of the collection (defaults to COLLECTION_NAME)
            backend: Override the default backend ("pgvector" or "qdrant")
            async_mode: For PGVector - True for retrieval (ainvoke), False for indexing (add_documents)

        Returns:
            A LangChain-compatible vector store instance
        """
        backend = backend or VECTOR_STORE_BACKEND
        collection = collection_name or COLLECTION_NAME

        if backend == "qdrant":
            return cls._create_qdrant_vectorstore(embedding_function, collection)
        elif backend == "pgvector":
            return cls._create_pgvector_vectorstore(embedding_function, collection, async_mode)
        else:
            raise ValueError(f"Unknown vector store backend: {backend}")

    @classmethod
    def _create_pgvector_vectorstore(
        cls,
        embedding_function: Embeddings,
        collection_name: str,
        async_mode: bool = False,
    ) -> LangchainPGVector:
        """Create a PGVector (PostgreSQL) vector store instance."""
        logger.info(f"Creating PGVector vectorstore with collection: {collection_name} (async_mode={async_mode})")

        vectorstore = LangchainPGVector(
            embeddings=embedding_function,
            collection_name=collection_name,
            connection=POSTGRES_CONNECTION_STRING,
            use_jsonb=True,
            async_mode=async_mode,
        )

        return vectorstore

    @classmethod
    def _create_qdrant_vectorstore(
        cls,
        embedding_function: Embeddings,
        collection_name: str,
    ) -> QdrantVectorStore:
        """Create a Qdrant vector store instance."""
        logger.info(f"Creating Qdrant vectorstore with collection: {collection_name}")

        client = cls.get_qdrant_client()

        # Check if collection exists, if not create it
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            # Get embedding dimension by embedding a test string
            test_embedding = embedding_function.embed_query("test")
            embedding_dim = len(test_embedding)

            logger.info(f"Creating Qdrant collection '{collection_name}' with dimension {embedding_dim}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

        # Create the LangChain Qdrant vectorstore
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_function,
        )

        return vectorstore

    @classmethod
    def get_backend_type(cls) -> str:
        """Get the currently configured backend type."""
        return VECTOR_STORE_BACKEND


def create_vectorstore(
    embedding_function: Embeddings,
    collection_name: Optional[str] = None,
    backend: Optional[str] = None,
    async_mode: bool = False,
) -> Any:
    """
    Convenience function to create a vector store.

    Args:
        embedding_function: The embedding function to use
        collection_name: Name of the collection
        backend: Override the default backend
        async_mode: For PGVector - True for retrieval (ainvoke), False for indexing (add_documents)

    Returns:
        A LangChain-compatible vector store instance
    """
    return VectorStoreFactory.create_vectorstore(
        embedding_function=embedding_function,
        collection_name=collection_name,
        backend=backend,
        async_mode=async_mode,
    )
