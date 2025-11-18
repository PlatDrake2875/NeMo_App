# backend/routers/document_router.py
"""
Document router - thin web layer that delegates to DocumentService.
Handles HTTP concerns only, business logic is in services.document.DocumentService.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from langchain_postgres import PGVector as LangchainPGVector
import psycopg

from config import POSTGRES_LIBPQ_CONNECTION
from deps import get_document_service
from rag_components import get_vectorstore
from schemas import DocumentListResponse, DocumentSourcesResponse, DocumentSource
from services.document import DocumentService
from services.dataset_registry import DatasetRegistryService

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.get("", response_model=DocumentListResponse)
async def get_all_documents(
    dataset: Optional[str] = Query(None, description="Filter by dataset name"),
    document_service: DocumentService = Depends(get_document_service),
    vectorstore: LangchainPGVector = Depends(get_vectorstore),
):
    """
    Retrieves all document chunks currently stored in the vector database.

    Args:
        dataset: Optional dataset name to filter documents.
                 If provided, returns documents from that dataset only.
                 If not provided, returns documents from the default collection.
        document_service: Injected DocumentService instance
        vectorstore: Injected vectorstore instance

    Returns:
        DocumentListResponse with all documents and metadata
    """
    return await document_service.get_all_documents(vectorstore, dataset)


@router.get("/sources", response_model=DocumentSourcesResponse)
async def get_document_sources(
    dataset: Optional[str] = Query(None, description="Filter by dataset name"),
):
    """
    Get list of unique document sources (original filenames) in a dataset.

    Args:
        dataset: Optional dataset name to filter documents.
                 If not provided, uses the default collection.

    Returns:
        DocumentSourcesResponse with list of document sources and their metadata
    """
    try:
        # Determine which collection to query
        if dataset:
            # Get dataset info
            dataset_registry = DatasetRegistryService()
            dataset_info = dataset_registry.get_dataset(dataset)
            collection_name = dataset_info.collection_name
        else:
            from config import COLLECTION_NAME
            collection_name = COLLECTION_NAME

        # Query for unique document sources with metadata
        with psycopg.connect(POSTGRES_LIBPQ_CONNECTION) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        e.cmetadata->>'original_filename' as filename,
                        COUNT(*) as chunk_count,
                        (array_agg(e.cmetadata->>'chunking_method'))[1] as chunking_method,
                        (array_agg((e.cmetadata->>'chunk_size')::int))[1] as chunk_size
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                    AND e.cmetadata ? 'original_filename'
                    GROUP BY e.cmetadata->>'original_filename'
                    ORDER BY filename
                    """,
                    (collection_name,)
                )

                results = cur.fetchall()

                if not results:
                    return DocumentSourcesResponse(count=0, documents=[])

                # Process results into DocumentSource objects
                sources = []
                for row in results:
                    filename, chunk_count, chunking_method, chunk_size = row

                    sources.append(
                        DocumentSource(
                            original_filename=filename,
                            chunk_count=chunk_count,
                            chunking_method=chunking_method,
                            chunk_size=chunk_size
                        )
                    )

                return DocumentSourcesResponse(count=len(sources), documents=sources)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document sources: {str(e)}"
        ) from e
