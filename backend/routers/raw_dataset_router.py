"""
Raw Dataset Router - API endpoints for managing raw (unprocessed) datasets.
"""

import logging
from typing import List
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from deps import get_db
from schemas import (
    BatchUploadResponse,
    RawDatasetCreate,
    RawDatasetInfo,
    RawDatasetListResponse,
    RawFileInfo,
)
from services.raw_dataset import raw_dataset_service
from utils.error_handlers import handle_service_errors, require_found

logger = logging.getLogger(__name__)

# Rate limiter for upload endpoints
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/raw-datasets", tags=["raw-datasets"])


def _safe_content_disposition(filename: str, disposition: str = "attachment") -> str:
    """
    Generate safe Content-Disposition header value using RFC 5987 encoding.

    This prevents header injection attacks from malicious filenames and
    properly handles Unicode filenames.

    Args:
        filename: The filename to encode
        disposition: Either "attachment" (download) or "inline" (preview)

    Returns:
        Safe Content-Disposition header value
    """
    # ASCII-safe fallback filename (replace non-ASCII chars with _)
    ascii_filename = filename.encode("ascii", "replace").decode("ascii").replace("?", "_")
    # Remove any quotes or newlines that could break the header
    ascii_filename = ascii_filename.replace('"', "_").replace("\n", "_").replace("\r", "_")
    # RFC 5987 encoded filename for Unicode support
    encoded_filename = quote(filename, safe="")
    return f'{disposition}; filename="{ascii_filename}"; filename*=UTF-8\'\'{encoded_filename}'


@router.post("", response_model=RawDatasetInfo)
@handle_service_errors("create raw dataset")
async def create_raw_dataset(
    request: RawDatasetCreate,
    db: Session = Depends(get_db),
):
    """
    Create a new empty raw dataset container.

    Raw datasets hold unprocessed documents that can later be
    processed through the preprocessing pipeline.
    """
    return raw_dataset_service.create_dataset(db, request)


@router.get("", response_model=RawDatasetListResponse)
async def list_raw_datasets(
    include_files: bool = Query(False, description="Include file list in response"),
    db: Session = Depends(get_db),
):
    """
    List all raw datasets.

    Returns a list of all raw datasets with optional file details.
    """
    return raw_dataset_service.list_datasets(db, include_files=include_files)


@router.get("/{dataset_id}", response_model=RawDatasetInfo)
async def get_raw_dataset(
    dataset_id: int,
    include_files: bool = Query(True, description="Include file list in response"),
    db: Session = Depends(get_db),
):
    """
    Get a raw dataset by ID.

    Returns detailed information about the dataset including its files.
    """
    dataset = raw_dataset_service.get_dataset(db, dataset_id, include_files=include_files)
    return require_found(dataset, "Dataset", dataset_id)


@router.delete("/{dataset_id}")
@handle_service_errors("delete raw dataset")
async def delete_raw_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a raw dataset and all its files.

    This is a permanent operation that cannot be undone.
    """
    return raw_dataset_service.delete_dataset(db, dataset_id)


@router.post("/{dataset_id}/files", response_model=RawFileInfo)
@limiter.limit("10/minute")
@handle_service_errors("upload file")
async def upload_file(
    request: Request,  # Required for rate limiter
    dataset_id: int,
    file: UploadFile = File(..., description="File to upload (PDF, JSON, MD, TXT, CSV)"),
    db: Session = Depends(get_db),
):
    """
    Upload a file to a raw dataset.

    Supported file types: PDF, JSON, Markdown, Text, CSV.
    Files are stored as BLOBs in PostgreSQL with deduplication.
    Rate limited to 10 uploads per minute per IP.
    """
    return await raw_dataset_service.add_file(db, dataset_id, file)


@router.post("/{dataset_id}/files/batch", response_model=BatchUploadResponse)
@limiter.limit("5/minute")
@handle_service_errors("upload files batch")
async def upload_files_batch(
    request: Request,  # Required for rate limiter
    dataset_id: int,
    files: List[UploadFile] = File(..., description="Files to upload"),
    db: Session = Depends(get_db),
):
    """
    Upload multiple files to a raw dataset.

    Returns structured response with both successful uploads and failures.
    Rate limited to 5 batch uploads per minute per IP.
    """
    result = await raw_dataset_service.add_files_batch(db, dataset_id, files)

    # Transform service response to match BatchUploadResponse schema
    return BatchUploadResponse(
        successful_count=len(result["successful"]),
        failed_count=len(result["failed"]),
        successful_files=[f.filename for f in result["successful"]],
        failed_files=result["failed"],
    )


@router.delete("/{dataset_id}/files/{file_id}")
@handle_service_errors("delete file")
async def delete_file(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a specific file from a raw dataset.

    This is a permanent operation that cannot be undone.
    """
    return raw_dataset_service.delete_file(db, dataset_id, file_id)


@router.get("/{dataset_id}/files/{file_id}")
async def get_file_info(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
):
    """
    Get information about a specific file (without content).
    """
    file_info = raw_dataset_service.get_file_info(db, dataset_id, file_id)
    return require_found(file_info, "File", file_id)


@router.get("/{dataset_id}/files/{file_id}/download")
@handle_service_errors("download file")
async def download_file(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
):
    """
    Download the content of a file.

    Returns the raw file content with appropriate content-type header.
    """
    content, filename, mime_type = raw_dataset_service.get_file_content(
        db, dataset_id, file_id
    )
    return Response(
        content=content,
        media_type=mime_type,
        headers={
            "Content-Disposition": _safe_content_disposition(filename, "attachment"),
            "Content-Length": str(len(content)),
        },
    )


@router.get("/{dataset_id}/files/{file_id}/preview")
@handle_service_errors("preview file")
async def preview_file(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
):
    """
    Get file content for preview (inline display).

    Returns the raw file content with inline content-disposition.
    """
    content, filename, mime_type = raw_dataset_service.get_file_content(
        db, dataset_id, file_id
    )
    return Response(
        content=content,
        media_type=mime_type,
        headers={
            "Content-Disposition": _safe_content_disposition(filename, "inline"),
            "Content-Length": str(len(content)),
        },
    )


@router.post("/{dataset_id}/refresh-stats")
async def refresh_dataset_stats(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Recalculate and update dataset statistics.

    Useful if stats become out of sync.
    """
    raw_dataset_service.update_dataset_stats(db, dataset_id)
    dataset = raw_dataset_service.get_dataset(db, dataset_id, include_files=False)
    require_found(dataset, "Dataset", dataset_id)
    return {
        "message": "Stats refreshed",
        "total_file_count": dataset.total_file_count,
        "total_size_bytes": dataset.total_size_bytes,
    }
