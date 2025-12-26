"""
Processed Dataset Service - Business logic for processed dataset management.
Coordinates with preprocessing pipeline and vector stores.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from database_models import (
    ProcessedDataset,
    ProcessingStatus,
    RawDataset,
    get_db_session,
)
from schemas import (
    EmbedderConfig,
    PreprocessingConfig,
    ProcessedDatasetCreate,
    ProcessedDatasetInfo,
    ProcessedDatasetListResponse,
    ProcessingStatusEnum,
    ProcessingStatusResponse,
)

logger = logging.getLogger(__name__)


class ProcessedDatasetService:
    """Service for managing processed datasets."""

    def create_dataset(
        self, db: Session, request: ProcessedDatasetCreate
    ) -> ProcessedDatasetInfo:
        """
        Create a new processed dataset record (does not start processing).

        Args:
            db: Database session
            request: Dataset creation request

        Returns:
            ProcessedDatasetInfo with the created dataset

        Raises:
            ValueError: If name already exists or raw dataset not found
        """
        # Check if name already exists
        existing = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.name == request.name)
            .first()
        )
        if existing:
            raise ValueError(f"Processed dataset with name '{request.name}' already exists")

        # Verify raw dataset exists
        raw_dataset = (
            db.query(RawDataset)
            .filter(RawDataset.id == request.raw_dataset_id)
            .first()
        )
        if not raw_dataset:
            raise ValueError(f"Raw dataset with id {request.raw_dataset_id} not found")

        # Generate collection name
        collection_name = f"processed_{request.name}".replace("-", "_").replace(" ", "_")

        # Create the processed dataset record
        dataset = ProcessedDataset(
            name=request.name,
            description=request.description,
            raw_dataset_id=request.raw_dataset_id,
            collection_name=collection_name,
            vector_backend=request.vector_backend,
            embedder_model_name=request.embedder_config.model_name,
            embedder_model_type=request.embedder_config.model_type,
            embedder_dimensions=request.embedder_config.dimensions,
            embedder_model_kwargs=request.embedder_config.model_kwargs or {},
            preprocessing_config=request.preprocessing_config.model_dump(),
            processing_status=ProcessingStatus.PENDING.value,
            document_count=0,
            chunk_count=0,
        )

        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        logger.info(
            f"Created processed dataset: {dataset.name} "
            f"(id={dataset.id}, raw_dataset={raw_dataset.name})"
        )

        return self._to_dataset_info(dataset, raw_dataset.name)

    def list_datasets(self, db: Session) -> ProcessedDatasetListResponse:
        """List all processed datasets."""
        datasets = (
            db.query(ProcessedDataset)
            .order_by(ProcessedDataset.created_at.desc())
            .all()
        )

        dataset_infos = []
        for ds in datasets:
            raw_name = None
            if ds.raw_dataset_id:
                raw_ds = (
                    db.query(RawDataset)
                    .filter(RawDataset.id == ds.raw_dataset_id)
                    .first()
                )
                raw_name = raw_ds.name if raw_ds else None
            dataset_infos.append(self._to_dataset_info(ds, raw_name))

        return ProcessedDatasetListResponse(count=len(dataset_infos), datasets=dataset_infos)

    def get_dataset(
        self, db: Session, dataset_id: int
    ) -> Optional[ProcessedDatasetInfo]:
        """Get a processed dataset by ID."""
        dataset = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.id == dataset_id)
            .first()
        )
        if not dataset:
            return None

        raw_name = None
        if dataset.raw_dataset_id:
            raw_ds = (
                db.query(RawDataset)
                .filter(RawDataset.id == dataset.raw_dataset_id)
                .first()
            )
            raw_name = raw_ds.name if raw_ds else None

        return self._to_dataset_info(dataset, raw_name)

    def get_dataset_by_name(
        self, db: Session, name: str
    ) -> Optional[ProcessedDatasetInfo]:
        """Get a processed dataset by name."""
        dataset = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.name == name)
            .first()
        )
        if not dataset:
            return None

        raw_name = None
        if dataset.raw_dataset_id:
            raw_ds = (
                db.query(RawDataset)
                .filter(RawDataset.id == dataset.raw_dataset_id)
                .first()
            )
            raw_name = raw_ds.name if raw_ds else None

        return self._to_dataset_info(dataset, raw_name)

    def delete_dataset(self, db: Session, dataset_id: int) -> dict:
        """
        Delete a processed dataset and its vector store collection.

        Args:
            db: Database session
            dataset_id: ID of the dataset to delete

        Returns:
            Dict with deletion details
        """
        dataset = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.id == dataset_id)
            .first()
        )
        if not dataset:
            raise ValueError(f"Processed dataset with id {dataset_id} not found")

        name = dataset.name
        collection_name = dataset.collection_name
        vector_backend = dataset.vector_backend

        # Delete the vector store collection
        try:
            self._delete_vector_collection(collection_name, vector_backend)
        except Exception as e:
            logger.warning(f"Error deleting vector collection {collection_name}: {e}")

        # Delete the database record
        db.delete(dataset)
        db.commit()

        logger.info(f"Deleted processed dataset: {name} (id={dataset_id})")
        return {
            "message": f"Processed dataset '{name}' deleted successfully",
            "collection_deleted": collection_name,
        }

    def get_processing_status(
        self, db: Session, dataset_id: int
    ) -> ProcessingStatusResponse:
        """Get the processing status of a dataset."""
        dataset = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.id == dataset_id)
            .first()
        )
        if not dataset:
            raise ValueError(f"Processed dataset with id {dataset_id} not found")

        return ProcessingStatusResponse(
            status=ProcessingStatusEnum(dataset.processing_status),
            progress_percent=None,  # Could be tracked in a separate table
            current_step=None,
            error=dataset.processing_error,
        )

    def update_processing_status(
        self,
        db: Session,
        dataset_id: int,
        status: ProcessingStatus,
        error: Optional[str] = None,
    ) -> None:
        """Update the processing status of a dataset."""
        dataset = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.id == dataset_id)
            .first()
        )
        if not dataset:
            return

        dataset.processing_status = status.value
        dataset.processing_error = error
        dataset.updated_at = datetime.now(timezone.utc)

        if status == ProcessingStatus.PROCESSING:
            dataset.processing_started_at = datetime.now(timezone.utc)
        elif status in (ProcessingStatus.COMPLETED, ProcessingStatus.FAILED):
            dataset.processing_completed_at = datetime.now(timezone.utc)

        db.commit()

    def update_counts(
        self, db: Session, dataset_id: int, document_count: int, chunk_count: int
    ) -> None:
        """Update the document and chunk counts."""
        dataset = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.id == dataset_id)
            .first()
        )
        if not dataset:
            return

        dataset.document_count = document_count
        dataset.chunk_count = chunk_count
        dataset.updated_at = datetime.now(timezone.utc)
        db.commit()

    def get_datasets_by_backend(self, db: Session) -> dict:
        """Get count of datasets grouped by vector backend."""
        result = (
            db.query(
                ProcessedDataset.vector_backend,
                func.count(ProcessedDataset.id).label("count"),
            )
            .group_by(ProcessedDataset.vector_backend)
            .all()
        )
        return {backend: count for backend, count in result}

    def get_datasets_by_chunking_method(self, db: Session) -> dict:
        """Get count of datasets grouped by chunking method."""
        datasets = db.query(ProcessedDataset).all()
        method_counts = {}
        for ds in datasets:
            config = ds.preprocessing_config or {}
            chunking = config.get("chunking", {})
            method = chunking.get("method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
        return method_counts

    def _delete_vector_collection(
        self, collection_name: str, vector_backend: str
    ) -> None:
        """Delete a vector store collection."""
        if vector_backend == "pgvector":
            # For pgvector, we need to delete from langchain_pg_embedding
            # This would require raw SQL or using the vectorstore directly
            logger.info(f"Would delete pgvector collection: {collection_name}")
            # Implementation depends on how collections are stored
        elif vector_backend == "qdrant":
            try:
                from qdrant_client import QdrantClient
                from config import QDRANT_HOST, QDRANT_PORT

                client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                client.delete_collection(collection_name)
                logger.info(f"Deleted Qdrant collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Error deleting Qdrant collection: {e}")

    def _to_dataset_info(
        self, dataset: ProcessedDataset, raw_dataset_name: Optional[str] = None
    ) -> ProcessedDatasetInfo:
        """Convert SQLAlchemy model to Pydantic schema."""
        preprocessing_config = PreprocessingConfig.model_validate(
            dataset.preprocessing_config or {}
        )

        return ProcessedDatasetInfo(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            raw_dataset_id=dataset.raw_dataset_id,
            raw_dataset_name=raw_dataset_name,
            collection_name=dataset.collection_name,
            vector_backend=dataset.vector_backend,
            embedder_config=EmbedderConfig(
                model_name=dataset.embedder_model_name,
                model_type=dataset.embedder_model_type,
                dimensions=dataset.embedder_dimensions,
                model_kwargs=dataset.embedder_model_kwargs,
            ),
            preprocessing_config=preprocessing_config,
            processing_status=ProcessingStatusEnum(dataset.processing_status),
            processing_error=dataset.processing_error,
            processing_started_at=dataset.processing_started_at,
            processing_completed_at=dataset.processing_completed_at,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            document_count=dataset.document_count,
            chunk_count=dataset.chunk_count,
        )


# Singleton instance
processed_dataset_service = ProcessedDatasetService()
