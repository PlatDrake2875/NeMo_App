"""
Processed Dataset Router - API endpoints for managing processed/indexed datasets.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from database_models import get_db_session, ProcessedDataset, RawDataset, ProcessingStatus
from schemas import (
    DatasetStatsResponse,
    ProcessedDatasetCreate,
    ProcessedDatasetInfo,
    ProcessedDatasetListResponse,
    ProcessingStatusResponse,
)
from services.processed_dataset import processed_dataset_service
from services.preprocessing_pipeline import preprocessing_pipeline_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/processed-datasets", tags=["processed-datasets"])


def get_db():
    """Dependency to get database session."""
    yield from get_db_session()


@router.post("", response_model=ProcessedDatasetInfo)
async def create_processed_dataset(
    request: ProcessedDatasetCreate,
    start_processing: bool = Query(False, description="Start processing immediately"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
):
    """
    Create a new processed dataset from a raw dataset.

    If start_processing is True, processing will begin in the background.
    Otherwise, call POST /processed-datasets/{id}/process to start.
    """
    try:
        dataset = processed_dataset_service.create_dataset(db, request)

        if start_processing and background_tasks:
            background_tasks.add_task(
                _run_processing,
                dataset.id,
            )

        return dataset
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=ProcessedDatasetListResponse)
async def list_processed_datasets(
    db: Session = Depends(get_db),
):
    """
    List all processed datasets.

    Returns a list of all processed datasets with their configurations and status.
    """
    return processed_dataset_service.list_datasets(db)


@router.get("/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(
    db: Session = Depends(get_db),
):
    """
    Get aggregate statistics about all datasets.

    Returns counts grouped by vector backend and chunking method.
    """
    from sqlalchemy import func

    # Count raw datasets
    raw_count = db.query(func.count(RawDataset.id)).scalar() or 0

    # Count raw files
    from database_models import RawFile
    raw_files_count = db.query(func.count(RawFile.id)).scalar() or 0

    # Total storage
    total_storage = db.query(func.sum(RawDataset.total_size_bytes)).scalar() or 0

    # Count processed datasets
    processed_count = db.query(func.count(ProcessedDataset.id)).scalar() or 0

    # Count in-progress
    in_progress = (
        db.query(func.count(ProcessedDataset.id))
        .filter(ProcessedDataset.processing_status == ProcessingStatus.PROCESSING.value)
        .scalar()
        or 0
    )

    # Group by backend
    datasets_by_backend = processed_dataset_service.get_datasets_by_backend(db)

    # Group by chunking method
    datasets_by_chunking = processed_dataset_service.get_datasets_by_chunking_method(db)

    return DatasetStatsResponse(
        total_raw_datasets=raw_count,
        total_processed_datasets=processed_count,
        total_raw_files=raw_files_count,
        total_storage_bytes=total_storage,
        processing_in_progress=in_progress,
        datasets_by_backend=datasets_by_backend,
        datasets_by_chunking_method=datasets_by_chunking,
    )


@router.get("/{dataset_id}", response_model=ProcessedDatasetInfo)
async def get_processed_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Get a processed dataset by ID.

    Returns detailed information about the dataset including preprocessing config.
    """
    dataset = processed_dataset_service.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found")
    return dataset


@router.delete("/{dataset_id}")
async def delete_processed_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a processed dataset and its vector store collection.

    This is a permanent operation that cannot be undone.
    """
    try:
        return processed_dataset_service.delete_dataset(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Get the processing status of a dataset.

    Returns current status, progress, and any errors.
    """
    try:
        return processed_dataset_service.get_processing_status(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{dataset_id}/process")
async def start_processing(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Start processing a dataset.

    Processing runs in the background. Use GET /status to check progress.
    """
    dataset = processed_dataset_service.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found")

    if dataset.processing_status == "processing":
        raise HTTPException(status_code=400, detail="Dataset is already being processed")

    if dataset.processing_status == "completed":
        raise HTTPException(
            status_code=400,
            detail="Dataset already processed. Use /reprocess to reprocess."
        )

    background_tasks.add_task(_run_processing, dataset_id)

    return {"message": "Processing started", "dataset_id": dataset_id}


@router.post("/{dataset_id}/process/stream")
async def start_processing_stream(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Start processing with Server-Sent Events for progress updates.

    Returns a stream of progress events.
    """
    dataset = processed_dataset_service.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found")

    if dataset.processing_status == "processing":
        raise HTTPException(status_code=400, detail="Dataset is already being processed")

    async def event_generator():
        try:
            async for update in preprocessing_pipeline_service.process_dataset_stream(
                db, dataset_id
            ):
                yield {
                    "event": update.get("type", "message"),
                    "data": json.dumps(update.get("data", {})),
                }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


@router.post("/{dataset_id}/reprocess")
async def reprocess_dataset(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Reprocess a dataset (clear and rebuild).

    This will delete existing chunks and reprocess from the raw dataset.
    """
    dataset = processed_dataset_service.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found")

    if dataset.processing_status == "processing":
        raise HTTPException(status_code=400, detail="Dataset is currently being processed")

    # Reset status
    processed_dataset_service.update_processing_status(
        db, dataset_id, ProcessingStatus.PENDING
    )

    # TODO: Clear existing vectors before reprocessing

    background_tasks.add_task(_run_processing, dataset_id)

    return {"message": "Reprocessing started", "dataset_id": dataset_id}


async def _run_processing(dataset_id: int):
    """Background task to run preprocessing pipeline."""
    from database_models import get_session_maker

    SessionLocal = get_session_maker()
    db = SessionLocal()
    try:
        await preprocessing_pipeline_service.process_dataset(db, dataset_id)
    except Exception as e:
        logger.error(f"Processing failed for dataset {dataset_id}: {e}")
        processed_dataset_service.update_processing_status(
            db, dataset_id, ProcessingStatus.FAILED, str(e)
        )
    finally:
        db.close()
