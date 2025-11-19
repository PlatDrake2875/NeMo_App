"""
Multi-embedder manager for handling multiple embedding models and vectorstores.
Provides factory methods and caching for dataset-specific embedders and vectorstores.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from config import VECTOR_STORE_BACKEND, logger
from schemas import EmbedderConfig
from vectorstore_factory import create_vectorstore


class MultiEmbedderManager:
    """Manages multiple embedders and vectorstores for different datasets."""

    def __init__(self):
        # Cache for embedders: {(model_name, model_type): embedder_instance}
        self._embedder_cache: Dict[tuple[str, str], Any] = {}
        # Cache for vectorstores: {collection_name: vectorstore_instance}
        self._vectorstore_cache: Dict[str, Any] = {}  # Can be PGVector or QdrantVectorStore

    def get_embedder(self, embedder_config: EmbedderConfig) -> Any:
        """
        Get or create an embedder based on configuration.
        Uses caching to avoid recreating embedders for the same model.
        """
        cache_key = (embedder_config.model_name, embedder_config.model_type)

        # Return cached embedder if available
        if cache_key in self._embedder_cache:
            logger.info(f"Using cached embedder: {embedder_config.model_name}")
            return self._embedder_cache[cache_key]

        # Create new embedder based on type
        logger.info(f"Creating new embedder: {embedder_config.model_name} ({embedder_config.model_type})")

        try:
            if embedder_config.model_type == "huggingface":
                embedder = self._create_huggingface_embedder(embedder_config)
            elif embedder_config.model_type == "openai":
                embedder = self._create_openai_embedder(embedder_config)
            else:
                raise ValueError(f"Unsupported embedder type: {embedder_config.model_type}")

            # Cache the embedder
            self._embedder_cache[cache_key] = embedder
            return embedder

        except Exception as e:
            logger.error(f"Failed to create embedder {embedder_config.model_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create embedder: {str(e)}"
            ) from e

    def get_vectorstore(
        self,
        collection_name: str,
        embedder_config: EmbedderConfig,
        use_cache: bool = True,
        backend: Optional[str] = None
    ) -> Any:
        """
        Get or create a vectorstore for a specific dataset collection.
        Supports both PGVector and Qdrant backends.
        """
        # Return cached vectorstore if available and caching is enabled
        if use_cache and collection_name in self._vectorstore_cache:
            logger.info(f"Using cached vectorstore: {collection_name}")
            return self._vectorstore_cache[collection_name]

        # Get the embedder for this dataset
        embedder = self.get_embedder(embedder_config)

        # Use configured backend or override
        vector_backend = backend or VECTOR_STORE_BACKEND

        # Create vectorstore using factory
        logger.info(f"Creating new {vector_backend} vectorstore for collection: {collection_name}")

        try:
            vectorstore = create_vectorstore(
                embedding_function=embedder,
                collection_name=collection_name,
                backend=vector_backend,
            )

            # Cache the vectorstore
            if use_cache:
                self._vectorstore_cache[collection_name] = vectorstore

            return vectorstore

        except Exception as e:
            logger.error(f"Failed to create vectorstore for {collection_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create vectorstore: {str(e)}"
            ) from e

    def clear_cache(self, collection_name: Optional[str] = None) -> None:
        """
        Clear cached vectorstores. If collection_name is provided, only clear that one.
        Otherwise, clear all caches.
        """
        if collection_name:
            self._vectorstore_cache.pop(collection_name, None)
            logger.info(f"Cleared vectorstore cache for: {collection_name}")
        else:
            self._vectorstore_cache.clear()
            self._embedder_cache.clear()
            logger.info("Cleared all vectorstore and embedder caches")

    def _create_huggingface_embedder(self, config: EmbedderConfig) -> HuggingFaceEmbeddings:
        """Create a HuggingFace embedder."""
        model_kwargs = config.model_kwargs or {}

        return HuggingFaceEmbeddings(
            model_name=config.model_name,
            model_kwargs=model_kwargs
        )

    def _create_openai_embedder(self, config: EmbedderConfig) -> OpenAIEmbeddings:
        """Create an OpenAI embedder."""
        model_kwargs = config.model_kwargs or {}

        # Extract OpenAI-specific kwargs
        api_key = model_kwargs.pop("api_key", None)
        organization = model_kwargs.pop("organization", None)

        return OpenAIEmbeddings(
            model=config.model_name,
            openai_api_key=api_key,
            openai_organization=organization,
            **model_kwargs
        )

    def get_or_detect_dimensions(self, embedder_config: EmbedderConfig) -> int:
        """
        Get or auto-detect the dimensions for an embedder.
        """
        if embedder_config.dimensions:
            return embedder_config.dimensions

        # Auto-detect by creating a test embedding
        try:
            embedder = self.get_embedder(embedder_config)
            test_embedding = embedder.embed_query("test")
            dimensions = len(test_embedding)
            logger.info(f"Auto-detected dimensions for {embedder_config.model_name}: {dimensions}")
            return dimensions
        except Exception as e:
            logger.error(f"Failed to auto-detect dimensions: {e}")
            # Return a default based on model type
            if "mini" in embedder_config.model_name.lower():
                return 384
            elif "base" in embedder_config.model_name.lower():
                return 768
            else:
                return 768  # Default fallback


# Global singleton instance
_multi_embedder_manager: Optional[MultiEmbedderManager] = None


def get_multi_embedder_manager() -> MultiEmbedderManager:
    """Get the global MultiEmbedderManager instance."""
    global _multi_embedder_manager
    if _multi_embedder_manager is None:
        _multi_embedder_manager = MultiEmbedderManager()
    return _multi_embedder_manager
