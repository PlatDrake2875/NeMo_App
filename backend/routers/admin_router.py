"""
Admin Router - Administrative endpoints for system management.

Provides endpoints for data reset and cleanup operations.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from database_models import (
    EvaluationTask,
    ProcessedDataset,
    RawDataset,
    get_db_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin"])

# Data directories
EVAL_DATASETS_DIR = Path("data/evaluation_datasets")
EVAL_RUNS_DIR = Path("data/evaluation_runs")


class ResetRequest(BaseModel):
    """Request body for data reset."""
    evaluation_runs: bool = True
    evaluation_datasets: bool = True
    evaluation_tasks: bool = True
    processed_datasets: bool = False
    raw_datasets: bool = False
    vector_collections: bool = False


class ResetResponse(BaseModel):
    """Response from data reset operation."""
    success: bool
    deleted_evaluation_runs: int = 0
    deleted_evaluation_datasets: int = 0
    deleted_evaluation_tasks: int = 0
    deleted_processed_datasets: int = 0
    deleted_raw_datasets: int = 0
    deleted_vector_collections: int = 0
    errors: list[str] = []


@router.post("/reset", response_model=ResetResponse)
async def reset_data(
    request: ResetRequest,
    db: Session = Depends(get_db_session),
):
    """
    Reset selected data from the system.

    This is a destructive operation that permanently deletes data.
    Use with caution and ensure proper backups are in place.
    """
    response = ResetResponse(success=True)
    errors = []

    try:
        # 1. Delete evaluation runs (JSON files)
        if request.evaluation_runs:
            count = _delete_evaluation_runs()
            response.deleted_evaluation_runs = count
            logger.info(f"Deleted {count} evaluation runs")

        # 2. Delete evaluation datasets (JSON files)
        if request.evaluation_datasets:
            count = _delete_evaluation_datasets()
            response.deleted_evaluation_datasets = count
            logger.info(f"Deleted {count} evaluation datasets")

        # 3. Delete evaluation tasks from database
        if request.evaluation_tasks:
            count = db.query(EvaluationTask).delete()
            response.deleted_evaluation_tasks = count
            db.commit()
            logger.info(f"Deleted {count} evaluation tasks from database")

        # 4. Delete processed datasets
        if request.processed_datasets:
            # Get processed datasets to find their collection names
            processed = db.query(ProcessedDataset).all()
            collection_names = [p.collection_name for p in processed]

            # Delete from database
            count = db.query(ProcessedDataset).delete()
            response.deleted_processed_datasets = count
            db.commit()

            # If vector_collections is also enabled, they'll be deleted there
            # Otherwise, we need to clean up pgvector collections
            if not request.vector_collections:
                _delete_pgvector_collections(db, collection_names)

            logger.info(f"Deleted {count} processed datasets")

        # 5. Delete raw datasets (cascades to raw_files)
        if request.raw_datasets:
            count = db.query(RawDataset).delete()
            response.deleted_raw_datasets = count
            db.commit()
            logger.info(f"Deleted {count} raw datasets")

        # 6. Delete vector collections (Qdrant + pgvector)
        if request.vector_collections:
            qdrant_count = _delete_qdrant_collections()
            pgvector_count = _delete_all_pgvector_collections(db)
            response.deleted_vector_collections = qdrant_count + pgvector_count
            logger.info(f"Deleted {response.deleted_vector_collections} vector collections")

        response.errors = errors
        return response

    except Exception as e:
        logger.error(f"Error during data reset: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error during data reset: {str(e)}"
        )


def _delete_evaluation_runs() -> int:
    """Delete all evaluation run JSON files."""
    if not EVAL_RUNS_DIR.exists():
        return 0

    count = 0
    for file in EVAL_RUNS_DIR.glob("*.json"):
        try:
            file.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete evaluation run {file}: {e}")

    return count


def _delete_evaluation_datasets() -> int:
    """Delete all evaluation dataset JSON files."""
    if not EVAL_DATASETS_DIR.exists():
        return 0

    count = 0
    for file in EVAL_DATASETS_DIR.glob("*.json"):
        try:
            file.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete evaluation dataset {file}: {e}")

    return count


def _delete_pgvector_collections(db: Session, collection_names: list[str]) -> int:
    """Delete specific pgvector collections by name."""
    if not collection_names:
        return 0

    count = 0
    for name in collection_names:
        try:
            # Delete embeddings first
            db.execute(
                text("""
                    DELETE FROM langchain_pg_embedding
                    WHERE collection_id = (
                        SELECT uuid FROM langchain_pg_collection WHERE name = :name
                    )
                """),
                {"name": name}
            )
            # Delete collection
            result = db.execute(
                text("DELETE FROM langchain_pg_collection WHERE name = :name"),
                {"name": name}
            )
            if result.rowcount > 0:
                count += 1
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to delete pgvector collection {name}: {e}")
            db.rollback()

    return count


def _delete_all_pgvector_collections(db: Session) -> int:
    """Delete all pgvector collections."""
    try:
        # Check if tables exist
        result = db.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'langchain_pg_collection'
            )
        """))
        if not result.scalar():
            return 0

        # Delete all embeddings
        db.execute(text("DELETE FROM langchain_pg_embedding"))
        # Delete all collections
        result = db.execute(text("DELETE FROM langchain_pg_collection"))
        count = result.rowcount
        db.commit()
        return count
    except Exception as e:
        logger.warning(f"Failed to delete pgvector collections: {e}")
        db.rollback()
        return 0


def _delete_qdrant_collections() -> int:
    """Delete all Qdrant collections."""
    try:
        from services.qdrant_vectorstore import get_qdrant_service

        qdrant_service = get_qdrant_service()
        if not qdrant_service:
            return 0

        # Get all collections
        collections = qdrant_service.list_collections()
        count = 0

        for collection_name in collections:
            try:
                qdrant_service.delete_collection(collection_name)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete Qdrant collection {collection_name}: {e}")

        return count
    except Exception as e:
        logger.warning(f"Failed to access Qdrant: {e}")
        return 0
