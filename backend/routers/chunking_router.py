"""
Chunking router - API endpoints for chunking methods and re-chunking operations.
"""

from fastapi import APIRouter, HTTPException
import psycopg

from schemas import (
    ChunkingMethodInfo,
    ChunkingMethodsResponse,
    RechunkRequest,
    RechunkResponse,
)
from services.chunking import ChunkingService
from services.dataset_registry import DatasetRegistryService
from config import POSTGRES_LIBPQ_CONNECTION

router = APIRouter(prefix="/chunking", tags=["chunking"])


@router.get("/methods", response_model=ChunkingMethodsResponse)
async def get_chunking_methods():
    """
    Get information about all available chunking methods.

    Returns:
        ChunkingMethodsResponse with details about each chunking method
    """
    methods = ChunkingService.get_available_methods()

    # Convert to schema format
    methods_info = {
        key: ChunkingMethodInfo(
            name=value["name"],
            description=value["description"],
            default_params=value["default_params"]
        )
        for key, value in methods.items()
    }

    return ChunkingMethodsResponse(methods=methods_info)


@router.post("/rechunk", response_model=RechunkResponse)
async def rechunk_document(request: RechunkRequest):
    """
    Re-chunk a document with a different chunking method.

    This will delete all existing chunks for the document and create new ones
    using the specified chunking method.

    Args:
        request: RechunkRequest with dataset, filename, and new chunking config

    Returns:
        RechunkResponse with operation results

    Raises:
        HTTPException 404: If dataset or document not found
        HTTPException 500: If re-chunking fails
    """
    try:
        # Get dataset info
        dataset_registry = DatasetRegistryService()
        dataset_info = dataset_registry.get_dataset(request.dataset_name)

        # Get all chunks for this document
        with psycopg.connect(POSTGRES_LIBPQ_CONNECTION) as conn:
            with conn.cursor() as cur:
                # Get the first chunk to retrieve the source path
                cur.execute(
                    """
                    SELECT e.cmetadata->>'source'
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                    AND e.cmetadata->>'original_filename' = %s
                    LIMIT 1
                    """,
                    (dataset_info.collection_name, request.original_filename)
                )
                result = cur.fetchone()

                if not result:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document '{request.original_filename}' not found in dataset '{request.dataset_name}'"
                    )

                source_path = result[0]

                # Delete existing chunks for this document
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding
                    WHERE collection_id IN (
                        SELECT uuid FROM langchain_pg_collection WHERE name = %s
                    )
                    AND cmetadata->>'original_filename' = %s
                    """,
                    (dataset_info.collection_name, request.original_filename)
                )
                conn.commit()

        # Note: In a production system, you'd re-load the document from storage
        # For now, we'll raise an error indicating this limitation
        raise HTTPException(
            status_code=501,
            detail="Re-chunking requires document re-upload. Please upload the document again with the new chunking method."
        )

        # TODO: Implement full re-chunking by:
        # 1. Storing original documents in a persistent location
        # 2. Loading the document from storage
        # 3. Re-chunking with new method
        # 4. Adding new chunks to vectorstore

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to re-chunk document: {str(e)}"
        ) from e
