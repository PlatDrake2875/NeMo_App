# backend/services/qdrant_vectorstore.py
"""
Qdrant-specific vector store operations.
Provides methods for document retrieval and management specific to Qdrant.
"""

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest

from config import (
    COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_PREFER_GRPC,
    logger,
)
from schemas import DocumentChunk


class QdrantVectorStoreService:
    """Service class for Qdrant-specific operations."""

    def __init__(self):
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            prefer_grpc=QDRANT_PREFER_GRPC,
        )

    def get_all_documents(
        self,
        collection_name: Optional[str] = None,
        limit: int = 1000,
    ) -> list[DocumentChunk]:
        """
        Retrieve all documents from a Qdrant collection.

        Args:
            collection_name: Name of the collection (defaults to COLLECTION_NAME)
            limit: Maximum number of documents to retrieve

        Returns:
            List of DocumentChunk objects
        """
        collection = collection_name or COLLECTION_NAME

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection not in collection_names:
                logger.warning(f"Collection '{collection}' does not exist in Qdrant")
                return []

            # Scroll through all points in the collection
            documents = []
            offset = None

            while True:
                results = self.client.scroll(
                    collection_name=collection,
                    limit=min(100, limit - len(documents)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, next_offset = results

                for point in points:
                    payload = point.payload or {}
                    doc = DocumentChunk(
                        id=str(point.id),
                        page_content=payload.get("page_content", ""),
                        metadata=payload.get("metadata", {}),
                    )
                    documents.append(doc)

                if next_offset is None or len(documents) >= limit:
                    break

                offset = next_offset

            logger.info(f"Retrieved {len(documents)} documents from Qdrant collection '{collection}'")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents from Qdrant: {e}")
            return []

    def get_document_sources(
        self,
        collection_name: Optional[str] = None,
    ) -> list[str]:
        """
        Get unique document sources (filenames) from a Qdrant collection.

        Args:
            collection_name: Name of the collection

        Returns:
            List of unique source filenames
        """
        documents = self.get_all_documents(collection_name)

        sources = set()
        for doc in documents:
            metadata = doc.metadata or {}
            source = metadata.get("original_filename") or metadata.get("source")
            if source:
                sources.add(source)

        return sorted(list(sources))

    def delete_by_source(
        self,
        source: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Delete all documents with a specific source from the collection.

        Args:
            source: The source filename to delete
            collection_name: Name of the collection

        Returns:
            Number of documents deleted
        """
        collection = collection_name or COLLECTION_NAME

        try:
            # Get count before deletion
            documents_before = self.get_all_documents(collection)
            count_before = len([
                d for d in documents_before
                if (d.metadata or {}).get("original_filename") == source
                or (d.metadata or {}).get("source") == source
            ])

            if count_before == 0:
                return 0

            # Delete points matching the source
            self.client.delete(
                collection_name=collection,
                points_selector=Filter(
                    should=[
                        FieldCondition(
                            key="metadata.original_filename",
                            match=MatchValue(value=source),
                        ),
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source),
                        ),
                    ]
                ),
            )

            logger.info(f"Deleted {count_before} documents with source '{source}' from Qdrant")
            return count_before

        except Exception as e:
            logger.error(f"Error deleting documents from Qdrant: {e}")
            return 0

    def get_collection_info(
        self,
        collection_name: Optional[str] = None,
    ) -> dict:
        """
        Get information about a Qdrant collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection information
        """
        collection = collection_name or COLLECTION_NAME

        try:
            info = self.client.get_collection(collection_name=collection)
            return {
                "name": collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "vector_size": info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else None,
            }
        except Exception as e:
            logger.error(f"Error getting collection info from Qdrant: {e}")
            return {"name": collection, "error": str(e)}

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Check if a collection exists in Qdrant."""
        collection = collection_name or COLLECTION_NAME

        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection for c in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False


# Singleton instance
_qdrant_service: Optional[QdrantVectorStoreService] = None


def get_qdrant_service() -> QdrantVectorStoreService:
    """Get or create the Qdrant vector store service singleton."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantVectorStoreService()
    return _qdrant_service
