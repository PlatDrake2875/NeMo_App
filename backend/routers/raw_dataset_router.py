"""
Raw Dataset Router - API endpoints for managing raw (unprocessed) datasets.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from database_models import get_db_session
from schemas import (
    RawDatasetCreate,
    RawDatasetInfo,
    RawDatasetListResponse,
    RawFileInfo,
)
from services.raw_dataset import raw_dataset_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/raw-datasets", tags=["raw-datasets"])


def get_db():
    """Dependency to get database session."""
    yield from get_db_session()


@router.post("", response_model=RawDatasetInfo)
async def create_raw_dataset(
    request: RawDatasetCreate,
    db: Session = Depends(get_db),
):
    """
    Create a new empty raw dataset container.

    Raw datasets hold unprocessed documents that can later be
    processed through the preprocessing pipeline.
    """
    try:
        return raw_dataset_service.create_dataset(db, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found")
    return dataset


@router.delete("/{dataset_id}")
async def delete_raw_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a raw dataset and all its files.

    This is a permanent operation that cannot be undone.
    """
    try:
        return raw_dataset_service.delete_dataset(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{dataset_id}/files", response_model=RawFileInfo)
async def upload_file(
    dataset_id: int,
    file: UploadFile = File(..., description="File to upload (PDF, JSON, MD, TXT, CSV)"),
    db: Session = Depends(get_db),
):
    """
    Upload a file to a raw dataset.

    Supported file types: PDF, JSON, Markdown, Text, CSV.
    Files are stored as BLOBs in PostgreSQL with deduplication.
    """
    try:
        return await raw_dataset_service.add_file(db, dataset_id, file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/files/batch", response_model=List[RawFileInfo])
async def upload_files_batch(
    dataset_id: int,
    files: List[UploadFile] = File(..., description="Files to upload"),
    db: Session = Depends(get_db),
):
    """
    Upload multiple files to a raw dataset.

    Returns a list of successfully uploaded files.
    Files that fail validation are skipped.
    """
    try:
        return await raw_dataset_service.add_files_batch(db, dataset_id, files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{dataset_id}/files/{file_id}")
async def delete_file(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a specific file from a raw dataset.

    This is a permanent operation that cannot be undone.
    """
    try:
        return raw_dataset_service.delete_file(db, dataset_id, file_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


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
    if not file_info:
        raise HTTPException(status_code=404, detail=f"File with id {file_id} not found")
    return file_info


@router.get("/{dataset_id}/files/{file_id}/download")
async def download_file(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
):
    """
    Download the content of a file.

    Returns the raw file content with appropriate content-type header.
    """
    try:
        content, filename, mime_type = raw_dataset_service.get_file_content(
            db, dataset_id, file_id
        )
        return Response(
            content=content,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(content)),
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_id}/files/{file_id}/preview")
async def preview_file(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
):
    """
    Get file content for preview (inline display).

    Returns the raw file content with inline content-disposition.
    """
    try:
        content, filename, mime_type = raw_dataset_service.get_file_content(
            db, dataset_id, file_id
        )
        return Response(
            content=content,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Length": str(len(content)),
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


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
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found")
    return {
        "message": "Stats refreshed",
        "total_file_count": dataset.total_file_count,
        "total_size_bytes": dataset.total_size_bytes,
    }
