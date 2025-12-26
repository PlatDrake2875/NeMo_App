"""
HuggingFace Integration Router - Import datasets from HuggingFace Hub.
"""

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from database_models import get_db_session
from schemas import (
    HFDatasetConfig,
    HFDatasetMetadata,
    HFDirectProcessRequest,
    HFImportAsRawRequest,
)
from services.huggingface_dataset import huggingface_dataset_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/huggingface", tags=["huggingface"])


def get_db():
    """Dependency to get database session."""
    yield from get_db_session()


@router.get("/datasets/search")
async def search_datasets(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
):
    """
    Search for datasets on HuggingFace Hub.

    Returns a list of matching datasets with basic metadata.
    """
    results = await huggingface_dataset_service.search_datasets(query, limit)
    return {"count": len(results), "datasets": results}


@router.get("/datasets/{dataset_id:path}/metadata", response_model=HFDatasetMetadata)
async def get_dataset_metadata(
    dataset_id: str,
):
    """
    Get metadata about a HuggingFace dataset.

    Returns information about the dataset including available splits,
    features (columns), and row counts.
    """
    try:
        return await huggingface_dataset_service.get_dataset_metadata(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/datasets/{dataset_id:path}/validate")
async def validate_dataset(
    dataset_id: str,
):
    """
    Validate that a HuggingFace dataset exists and is accessible.

    Returns validation status and any error messages.
    """
    is_valid, error = await huggingface_dataset_service.validate_dataset(dataset_id)
    return {
        "dataset_id": dataset_id,
        "is_valid": is_valid,
        "error": error,
    }


@router.post("/import-raw")
async def import_as_raw_dataset(
    request: HFImportAsRawRequest,
    db: Session = Depends(get_db),
):
    """
    Import a HuggingFace dataset as a raw dataset.

    This streams progress updates via Server-Sent Events.
    The dataset will be downloaded and stored as JSON files
    in the raw dataset for later processing.
    """

    async def event_generator():
        try:
            async for update in huggingface_dataset_service.import_as_raw(db, request):
                yield {
                    "event": update.get("type", "message"),
                    "data": json.dumps(update),
                }
        except Exception as e:
            logger.error(f"Import error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": str(e)}),
            }

    return EventSourceResponse(event_generator())


@router.post("/process-direct")
async def process_dataset_directly(
    request: HFDirectProcessRequest,
    db: Session = Depends(get_db),
):
    """
    Download and process a HuggingFace dataset directly to vector store.

    This combines import and processing into a single operation.
    Streams progress updates via Server-Sent Events.
    """

    async def event_generator():
        try:
            async for update in huggingface_dataset_service.process_direct(db, request):
                yield {
                    "event": update.get("type", "message"),
                    "data": json.dumps(update),
                }
        except Exception as e:
            logger.error(f"Direct processing error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": str(e)}),
            }

    return EventSourceResponse(event_generator())


@router.get("/datasets/{dataset_id:path}/preview")
async def preview_dataset(
    dataset_id: str,
    split: str = Query("train", description="Dataset split to preview"),
    limit: int = Query(10, ge=1, le=100, description="Number of rows to preview"),
    text_column: Optional[str] = Query(None, description="Column to use as text"),
):
    """
    Preview rows from a HuggingFace dataset.

    Returns sample rows from the specified split.
    """
    try:
        from datasets import load_dataset
        import asyncio

        loop = asyncio.get_event_loop()

        # Load a small sample
        dataset = await loop.run_in_executor(
            None,
            lambda: load_dataset(
                dataset_id,
                split=f"{split}[:{limit}]",
                trust_remote_code=True,
            ),
        )

        # Convert to list of dicts
        rows = []
        for i in range(min(limit, len(dataset))):
            row = dataset[i]
            # Convert non-JSON-serializable types to strings
            cleaned_row = {}
            for k, v in row.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    cleaned_row[k] = v
                elif isinstance(v, (list, dict)):
                    cleaned_row[k] = v
                else:
                    cleaned_row[k] = str(v)
            rows.append(cleaned_row)

        # Get columns
        columns = list(dataset.features.keys()) if hasattr(dataset, "features") else []

        return {
            "dataset_id": dataset_id,
            "split": split,
            "columns": columns,
            "row_count": len(rows),
            "rows": rows,
            "suggested_text_column": text_column or _suggest_text_column(columns, rows),
        }

    except Exception as e:
        logger.error(f"Preview error for {dataset_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to preview dataset: {e}")


def _suggest_text_column(columns: List[str], rows: List[dict]) -> Optional[str]:
    """Suggest the best column to use as text content."""
    # Common text column names
    preferred = ["text", "content", "body", "context", "passage", "document", "question"]

    for col in preferred:
        if col in columns:
            return col

    # Find the column with the longest average text
    if rows:
        avg_lengths = {}
        for col in columns:
            lengths = [
                len(str(row.get(col, "")))
                for row in rows
                if isinstance(row.get(col), str)
            ]
            if lengths:
                avg_lengths[col] = sum(lengths) / len(lengths)

        if avg_lengths:
            return max(avg_lengths, key=avg_lengths.get)

    return columns[0] if columns else None
