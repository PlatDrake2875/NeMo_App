"""
Document service layer containing all document retrieval business logic.
Separated from the web layer for better testability and maintainability.
"""

from typing import Any, Optional

from fastapi import HTTPException
import psycopg

from config import COLLECTION_NAME, POSTGRES_LIBPQ_CONNECTION, VECTOR_STORE_BACKEND
from multi_embedder_manager import get_multi_embedder_manager
from schemas import DocumentChunk, DocumentListResponse
from services.dataset_registry import DatasetRegistryService
from services.qdrant_vectorstore import get_qdrant_service


class DocumentService:
    """Service class handling all document retrieval business logic."""

    def __init__(self):
        self.collection_name = COLLECTION_NAME
        self.backend = VECTOR_STORE_BACKEND

    async def get_all_documents(
        self,
        default_vectorstore: Any,
        dataset_name: Optional[str] = None
    ) -> DocumentListResponse:
        try:
            # Determine which collection to query
            if dataset_name:
                # Get dataset info
                dataset_registry = DatasetRegistryService()
                dataset_info = dataset_registry.get_dataset(dataset_name)
                collection_name = dataset_info.collection_name
            else:
                # Use default collection
                collection_name = self.collection_name

            # Route to appropriate backend
            if self.backend == "qdrant":
                return await self._get_documents_from_qdrant(collection_name)
            else:
                return await self._get_documents_from_pgvector(collection_name)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred while retrieving documents: {str(e)}",
            ) from e

    async def _get_documents_from_pgvector(self, collection_name: str) -> DocumentListResponse:
        """Retrieve documents from PostgreSQL/PGVector."""
        # Query PostgreSQL directly to get all documents from the collection
        with psycopg.connect(POSTGRES_LIBPQ_CONNECTION) as conn:
            with conn.cursor() as cur:
                # Query the langchain_pg_embedding table (default table used by PGVector)
                # The table structure includes: id, collection_id, embedding, document, cmetadata
                cur.execute(
                    """
                    SELECT e.id, e.document, e.cmetadata
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                    ORDER BY e.id
                    """,
                    (collection_name,)
                )

                results = cur.fetchall()

                if not results:
                    return DocumentListResponse(count=0, documents=[])

                # Process results into DocumentChunk objects
                doc_chunks = []
                for row in results:
                    doc_id, document, metadata = row

                    # Ensure metadata is a dict
                    if metadata is None:
                        metadata = {}

                    # Ensure document is a string
                    if document is None:
                        document = ""

                    doc_chunks.append(
                        DocumentChunk(
                            id=str(doc_id),
                            page_content=document,
                            metadata=metadata,
                        )
                    )

                return DocumentListResponse(count=len(doc_chunks), documents=doc_chunks)

    async def _get_documents_from_qdrant(self, collection_name: str) -> DocumentListResponse:
        """Retrieve documents from Qdrant."""
        qdrant_service = get_qdrant_service()
        doc_chunks = qdrant_service.get_all_documents(collection_name)
        return DocumentListResponse(count=len(doc_chunks), documents=doc_chunks)
