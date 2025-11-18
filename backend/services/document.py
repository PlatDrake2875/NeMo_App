"""
Document service layer containing all document retrieval business logic.
Separated from the web layer for better testability and maintainability.
"""

from typing import Any, Optional

from fastapi import HTTPException
from langchain_postgres.vectorstores import PGVector as LangchainPGVector
import psycopg

from config import COLLECTION_NAME, POSTGRES_LIBPQ_CONNECTION
from multi_embedder_manager import get_multi_embedder_manager
from schemas import DocumentChunk, DocumentListResponse
from services.dataset_registry import DatasetRegistryService


class DocumentService:
    """Service class handling all document retrieval business logic."""

    def __init__(self):
        self.collection_name = COLLECTION_NAME

    async def get_all_documents(
        self,
        default_vectorstore: LangchainPGVector,
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

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred while retrieving documents: {str(e)}",
            ) from e
