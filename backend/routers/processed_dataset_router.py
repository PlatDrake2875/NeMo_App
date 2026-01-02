"""
Processed Dataset Router - API endpoints for managing processed/indexed datasets.
"""

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from database_models import ProcessedDataset, RawDataset, ProcessingStatus
from deps import get_db
from schemas import (
    DatasetStatsResponse,
    ProcessedDatasetCreate,
    ProcessedDatasetInfo,
    ProcessedDatasetListResponse,
    ProcessingStatusResponse,
)
from services.processed_dataset import processed_dataset_service
from services.preprocessing_pipeline import preprocessing_pipeline_service
from utils.error_handlers import handle_service_errors, require_found

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/processed-datasets", tags=["processed-datasets"])


@router.post("", response_model=ProcessedDatasetInfo)
@handle_service_errors("create processed dataset")
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
    dataset = processed_dataset_service.create_dataset(db, request)

    if start_processing and background_tasks:
        background_tasks.add_task(
            _run_processing,
            dataset.id,
        )

    return dataset


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
    from database_models import RawFile

    raw_count = db.query(func.count(RawDataset.id)).scalar() or 0
    raw_files_count = db.query(func.count(RawFile.id)).scalar() or 0
    total_storage = db.query(func.sum(RawDataset.total_size_bytes)).scalar() or 0
    processed_count = db.query(func.count(ProcessedDataset.id)).scalar() or 0
    in_progress = (
        db.query(func.count(ProcessedDataset.id))
        .filter(ProcessedDataset.processing_status == ProcessingStatus.PROCESSING.value)
        .scalar()
        or 0
    )
    datasets_by_backend = processed_dataset_service.get_datasets_by_backend(db)
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
    return require_found(dataset, "Dataset", dataset_id)


@router.get("/{dataset_id}/chunks")
@handle_service_errors("get chunks")
async def get_chunks(
    dataset_id: int,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(20, ge=1, le=100, description="Number of chunks per page"),
    search: str = Query(None, description="Optional search term to filter chunks"),
    db: Session = Depends(get_db),
):
    """
    Get chunks for a processed dataset with pagination and optional search.

    Returns a list of chunks with their content and metadata.
    """
    return processed_dataset_service.get_chunks(db, dataset_id, page, limit, search)


@router.delete("/{dataset_id}")
@handle_service_errors("delete processed dataset")
async def delete_processed_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a processed dataset and its vector store collection.

    This is a permanent operation that cannot be undone.
    """
    return processed_dataset_service.delete_dataset(db, dataset_id)


@router.get("/{dataset_id}/status", response_model=ProcessingStatusResponse)
@handle_service_errors("get processing status")
async def get_processing_status(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Get the processing status of a dataset.

    Returns current status, progress, and any errors.
    """
    return processed_dataset_service.get_processing_status(db, dataset_id)


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
    require_found(dataset, "Dataset", dataset_id)

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

    Note: Unlike the non-streaming endpoint, this allows reprocessing of
    completed datasets without requiring a call to /reprocess first.
    """
    dataset = processed_dataset_service.get_dataset(db, dataset_id)
    require_found(dataset, "Dataset", dataset_id)

    if dataset.processing_status == "processing":
        raise HTTPException(status_code=400, detail="Dataset is already being processed")

    async def event_generator():
        # Create dedicated database session for the generator
        from database_models import get_session_maker
        SessionLocal = get_session_maker()
        stream_db = SessionLocal()
        try:
            async for update in preprocessing_pipeline_service.process_dataset_stream(
                stream_db, dataset_id
            ):
                yield {
                    "event": update.get("type", "message"),
                    "data": json.dumps(update.get("data", {})),
                }
        except Exception as e:
            logger.error(f"SSE processing stream error for dataset {dataset_id}: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }
        finally:
            stream_db.close()

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
    require_found(dataset, "Dataset", dataset_id)

    if dataset.processing_status == "processing":
        raise HTTPException(status_code=400, detail="Dataset is currently being processed")

    # Clear existing vectors before reprocessing
    vectors_cleared = True
    clear_warning = None
    try:
        processed_dataset_service._delete_vector_collection(
            dataset.collection_name, dataset.vector_backend
        )
        logger.info(f"Cleared vector collection for reprocessing: {dataset.collection_name}")
    except Exception as e:
        # Log as ERROR since this could cause data integrity issues
        logger.error(f"Failed to clear vectors for reprocessing {dataset.collection_name}: {e}", exc_info=True)
        vectors_cleared = False
        clear_warning = f"CAUTION: Could not clear existing vectors. Reprocessing may result in duplicate or stale data. Error: {e}"

    # Reset status
    processed_dataset_service.update_processing_status(
        db, dataset_id, ProcessingStatus.PENDING
    )

    # Reset counts
    processed_dataset_service.update_counts(db, dataset_id, 0, 0)

    background_tasks.add_task(_run_processing, dataset_id)

    return {
        "message": "Reprocessing started",
        "dataset_id": dataset_id,
        "vectors_cleared": vectors_cleared,
        "warning": clear_warning,
    }


async def _run_processing(dataset_id: int):
    """Background task to run preprocessing pipeline."""
    from database_models import get_session_maker

    SessionLocal = get_session_maker()
    db = SessionLocal()
    try:
        await preprocessing_pipeline_service.process_dataset(db, dataset_id)
    except Exception as e:
        logger.error(f"Processing failed for dataset {dataset_id}: {e}", exc_info=True)
        try:
            processed_dataset_service.update_processing_status(
                db, dataset_id, ProcessingStatus.FAILED, str(e)
            )
        except Exception as status_error:
            logger.error(f"Failed to update status for dataset {dataset_id}: {status_error}", exc_info=True)
    finally:
        db.close()
