# backend/routers/upload_router.py
"""
Upload router - thin web layer that delegates to UploadService.
Handles HTTP concerns only, business logic is in services.upload.UploadService.
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from langchain_postgres import PGVector as LangchainPGVector

from deps import get_upload_service
from rag_components import get_vectorstore
from schemas import UploadResponse
from services.upload import UploadService

router = APIRouter(
    prefix="/upload",
    tags=["upload"],
)


@router.post("", response_model=UploadResponse)
async def upload_document_endpoint(
    file: UploadFile = File(...),
    dataset: Optional[str] = Query(None, description="Dataset name to upload to"),
    chunking_method: str = Query("recursive", description="Chunking method: recursive, fixed, or semantic"),
    chunk_size: int = Query(1000, description="Chunk size in characters"),
    chunk_overlap: int = Query(200, description="Overlap between chunks in characters"),
    upload_service: UploadService = Depends(get_upload_service),
    vectorstore: LangchainPGVector = Depends(get_vectorstore),
):
    """
    Upload and process a document for vector storage.

    Args:
        file: The uploaded file
        dataset: Optional dataset name. If provided, uploads to that dataset.
                 If not provided, uses the default vectorstore.
        chunking_method: Method to use for chunking (recursive, fixed, semantic)
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
        upload_service: Injected UploadService instance
        vectorstore: Injected default vectorstore instance (used when dataset is None)

    Returns:
        UploadResponse with processing results
    """
    # Delegate all business logic to the service
    return await upload_service.process_document_upload(
        file,
        vectorstore,
        dataset_name=dataset,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
