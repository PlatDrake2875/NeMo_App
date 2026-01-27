"""
Embedding Service - On-demand chunking and embedding for evaluation.

This service handles the deferred chunking/embedding workflow where:
1. Preprocessing stops at cleaning (saves to PreprocessedDocument table)
2. At evaluation time, user chooses chunking + embedder config
3. This service chunks, embeds, and indexes on-demand (with caching)

Caching strategy:
- Cache key = hash(preprocessed_dataset_id + chunking_config + embedder_model)
- If cache exists, reuse the vector store collection
- If not, create new collection and cache it
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy.orm import Session

from database_models import (
    EmbeddingCache,
    PreprocessedDocument,
    ProcessedDataset,
)
from schemas.evaluation import ChunkingConfig
from services.chunking import ChunkingService
from vectorstore_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Handles on-demand chunking and embedding for evaluation.

    This service is the bridge between preprocessed documents and vector store
    collections. It allows experimenting with different chunking/embedding
    configurations on the same preprocessed data.
    """

    def __init__(self):
        self.chunking_service = ChunkingService()

    def compute_config_hash(
        self,
        preprocessed_dataset_id: int,
        chunking_config: ChunkingConfig,
        embedder_model: str,
    ) -> str:
        """
        Compute a hash for the configuration to use as cache key.

        Args:
            preprocessed_dataset_id: ID of the preprocessed dataset
            chunking_config: Chunking configuration
            embedder_model: Embedding model name

        Returns:
            SHA-256 hash of the configuration (first 16 chars)
        """
        config_dict = {
            "dataset_id": preprocessed_dataset_id,
            "chunking": {
                "method": chunking_config.method,
                "chunk_size": chunking_config.chunk_size,
                "chunk_overlap": chunking_config.chunk_overlap,
            },
            "embedder": embedder_model,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_cached_collection(
        self,
        db: Session,
        preprocessed_dataset_id: int,
        chunking_config: ChunkingConfig,
        embedder_model: str,
    ) -> Optional[str]:
        """
        Check if a cached embedding collection exists for this configuration.

        Args:
            db: Database session
            preprocessed_dataset_id: ID of the preprocessed dataset
            chunking_config: Chunking configuration
            embedder_model: Embedding model name

        Returns:
            Collection name if cached, None otherwise
        """
        config_hash = self.compute_config_hash(
            preprocessed_dataset_id, chunking_config, embedder_model
        )

        cache_entry = (
            db.query(EmbeddingCache)
            .filter(
                EmbeddingCache.preprocessed_dataset_id == preprocessed_dataset_id,
                EmbeddingCache.config_hash == config_hash,
            )
            .first()
        )

        if cache_entry:
            # Update last_used_at
            cache_entry.last_used_at = datetime.now(timezone.utc)
            db.commit()

            logger.info(
                f"Cache hit for dataset {preprocessed_dataset_id} with config hash {config_hash}"
            )
            return cache_entry.collection_name

        logger.info(
            f"Cache miss for dataset {preprocessed_dataset_id} with config hash {config_hash}"
        )
        return None

    async def get_or_create_collection(
        self,
        db: Session,
        preprocessed_dataset_id: int,
        chunking_config: ChunkingConfig,
        embedder_model: str,
        vector_backend: str = "pgvector",
        progress_callback: Optional[callable] = None,
    ) -> Tuple[str, bool]:
        """
        Get existing cached collection or create a new one.

        This is the main entry point for the evaluation flow.

        Args:
            db: Database session
            preprocessed_dataset_id: ID of the preprocessed dataset
            chunking_config: Chunking configuration
            embedder_model: Embedding model name
            vector_backend: Vector store backend ('pgvector' or 'qdrant')
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (collection_name, was_cached)
        """
        # Check cache first
        cached_collection = self.get_cached_collection(
            db, preprocessed_dataset_id, chunking_config, embedder_model
        )

        if cached_collection:
            return cached_collection, True

        # No cache, create new collection
        collection_name = await self.create_collection(
            db=db,
            preprocessed_dataset_id=preprocessed_dataset_id,
            chunking_config=chunking_config,
            embedder_model=embedder_model,
            vector_backend=vector_backend,
            progress_callback=progress_callback,
        )

        return collection_name, False

    async def create_collection(
        self,
        db: Session,
        preprocessed_dataset_id: int,
        chunking_config: ChunkingConfig,
        embedder_model: str,
        vector_backend: str = "pgvector",
        progress_callback: Optional[callable] = None,
    ) -> str:
        """
        Create a new vector store collection from preprocessed documents.

        Steps:
        1. Load preprocessed documents from database
        2. Chunk documents using provided config
        3. Create embeddings and index to vector store
        4. Save to cache for future reuse

        Args:
            db: Database session
            preprocessed_dataset_id: ID of the preprocessed dataset
            chunking_config: Chunking configuration
            embedder_model: Embedding model name
            vector_backend: Vector store backend
            progress_callback: Optional callback for progress updates

        Returns:
            The created collection name
        """
        config_hash = self.compute_config_hash(
            preprocessed_dataset_id, chunking_config, embedder_model
        )

        # Generate unique collection name
        collection_name = f"eval_{preprocessed_dataset_id}_{config_hash}"

        logger.info(
            f"Creating collection '{collection_name}' for dataset {preprocessed_dataset_id}"
        )

        if progress_callback:
            progress_callback("loading_documents", {"status": "Loading preprocessed documents..."})

        # Step 1: Load preprocessed documents
        preprocessed_docs = (
            db.query(PreprocessedDocument)
            .filter(PreprocessedDocument.processed_dataset_id == preprocessed_dataset_id)
            .all()
        )

        if not preprocessed_docs:
            raise ValueError(
                f"No preprocessed documents found for dataset {preprocessed_dataset_id}"
            )

        logger.info(f"Loaded {len(preprocessed_docs)} preprocessed documents")

        # Convert to LangChain documents
        documents = [
            Document(
                page_content=doc.content,
                metadata={
                    "preprocessed_doc_id": doc.id,
                    "raw_file_id": doc.raw_file_id,
                    "original_filename": doc.original_filename,
                    **(doc.metadata_json or {}),
                },
            )
            for doc in preprocessed_docs
        ]

        if progress_callback:
            progress_callback("chunking", {"status": f"Chunking {len(documents)} documents..."})

        # Step 2: Chunk documents
        chunks = self.chunking_service.chunk_documents(
            documents=documents,
            method=chunking_config.method,
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap,
        )

        logger.info(f"Created {len(chunks)} chunks")

        if progress_callback:
            progress_callback("embedding", {"status": f"Embedding {len(chunks)} chunks..."})

        # Step 3: Create embeddings and index
        # Handle nomic models which require trust_remote_code=True
        model_kwargs = {"trust_remote_code": True} if "nomic" in embedder_model else {}

        embeddings = HuggingFaceEmbeddings(
            model_name=embedder_model,
            model_kwargs=model_kwargs,
        )

        vectorstore = VectorStoreFactory.create_vectorstore(
            backend=vector_backend,
            embedding_function=embeddings,
            collection_name=collection_name,
        )

        # Add documents to vector store
        vectorstore.add_documents(chunks)

        logger.info(f"Indexed {len(chunks)} chunks to collection '{collection_name}'")

        if progress_callback:
            progress_callback("caching", {"status": "Saving to cache..."})

        # Step 4: Save to cache
        cache_entry = EmbeddingCache(
            preprocessed_dataset_id=preprocessed_dataset_id,
            config_hash=config_hash,
            chunking_config={
                "method": chunking_config.method,
                "chunk_size": chunking_config.chunk_size,
                "chunk_overlap": chunking_config.chunk_overlap,
            },
            embedder_model=embedder_model,
            collection_name=collection_name,
            vector_backend=vector_backend,
            chunk_count=len(chunks),
        )

        db.add(cache_entry)
        db.commit()

        logger.info(f"Cached collection '{collection_name}' with config hash {config_hash}")

        if progress_callback:
            progress_callback("completed", {"collection_name": collection_name, "chunk_count": len(chunks)})

        return collection_name

    def list_cached_collections(
        self,
        db: Session,
        preprocessed_dataset_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all cached embedding collections.

        Args:
            db: Database session
            preprocessed_dataset_id: Optional filter by dataset ID

        Returns:
            List of cache entries as dictionaries
        """
        query = db.query(EmbeddingCache)

        if preprocessed_dataset_id:
            query = query.filter(
                EmbeddingCache.preprocessed_dataset_id == preprocessed_dataset_id
            )

        cache_entries = query.order_by(EmbeddingCache.created_at.desc()).all()

        return [
            {
                "id": entry.id,
                "preprocessed_dataset_id": entry.preprocessed_dataset_id,
                "config_hash": entry.config_hash,
                "chunking_config": entry.chunking_config,
                "embedder_model": entry.embedder_model,
                "collection_name": entry.collection_name,
                "vector_backend": entry.vector_backend,
                "chunk_count": entry.chunk_count,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
                "last_used_at": entry.last_used_at.isoformat() if entry.last_used_at else None,
            }
            for entry in cache_entries
        ]

    def delete_cached_collection(
        self,
        db: Session,
        cache_id: int,
    ) -> bool:
        """
        Delete a cached embedding collection.

        Note: This only removes the cache entry, not the actual vector store collection.
        The vector store collection may need to be cleaned up separately.

        Args:
            db: Database session
            cache_id: ID of the cache entry to delete

        Returns:
            True if deleted, False if not found
        """
        cache_entry = db.query(EmbeddingCache).filter(EmbeddingCache.id == cache_id).first()

        if not cache_entry:
            return False

        collection_name = cache_entry.collection_name
        db.delete(cache_entry)
        db.commit()

        logger.info(f"Deleted cache entry for collection '{collection_name}'")

        # TODO: Also delete the actual vector store collection
        # This would require calling vectorstore.delete_collection() or similar

        return True


# Singleton instance
embedding_service = EmbeddingService()
