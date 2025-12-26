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
            Dict with deletion details including any warnings
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
        collection_deleted = True
        collection_warning = None
        try:
            self._delete_vector_collection(collection_name, vector_backend)
        except Exception as e:
            logger.warning(f"Error deleting vector collection {collection_name}: {e}")
            collection_deleted = False
            collection_warning = f"Vector collection could not be deleted: {e}"

        # Delete the database record
        db.delete(dataset)
        db.commit()

        logger.info(f"Deleted processed dataset: {name} (id={dataset_id})")
        return {
            "message": f"Processed dataset '{name}' deleted successfully",
            "collection_name": collection_name,
            "collection_deleted": collection_deleted,
            "warning": collection_warning,
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
    ) -> bool:
        """
        Update the processing status of a dataset.

        Returns:
            True if update successful, False if dataset not found
        """
        dataset = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.id == dataset_id)
            .first()
        )
        if not dataset:
            logger.error(f"Cannot update status: dataset {dataset_id} not found")
            return False

        dataset.processing_status = status.value
        dataset.processing_error = error
        dataset.updated_at = datetime.now(timezone.utc)

        if status == ProcessingStatus.PROCESSING:
            dataset.processing_started_at = datetime.now(timezone.utc)
        elif status in (ProcessingStatus.COMPLETED, ProcessingStatus.FAILED):
            dataset.processing_completed_at = datetime.now(timezone.utc)

        db.commit()
        logger.debug(f"Updated dataset {dataset_id} status to {status.value}")
        return True

    def update_counts(
        self, db: Session, dataset_id: int, document_count: int, chunk_count: int
    ) -> bool:
        """
        Update the document and chunk counts.

        Returns:
            True if update successful, False if dataset not found
        """
        dataset = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.id == dataset_id)
            .first()
        )
        if not dataset:
            logger.error(f"Cannot update counts: dataset {dataset_id} not found")
            return False

        dataset.document_count = document_count
        dataset.chunk_count = chunk_count
        dataset.updated_at = datetime.now(timezone.utc)
        db.commit()
        return True

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

    def get_chunks(
        self,
        db: Session,
        dataset_id: int,
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
    ) -> dict:
        """
        Fetch chunks from vector store with pagination and optional search.

        Args:
            db: Database session
            dataset_id: ID of the processed dataset
            page: Page number (1-indexed)
            limit: Number of chunks per page
            search: Optional search term to filter chunks

        Returns:
            Dict with 'chunks' list and 'total' count

        Raises:
            ValueError: If dataset not found
        """
        dataset = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.id == dataset_id)
            .first()
        )
        if not dataset:
            raise ValueError(f"Processed dataset with id {dataset_id} not found")

        collection_name = dataset.collection_name
        vector_backend = dataset.vector_backend

        if vector_backend == "pgvector":
            return self._get_chunks_pgvector(collection_name, page, limit, search)
        elif vector_backend == "qdrant":
            return self._get_chunks_qdrant(collection_name, page, limit, search)
        else:
            logger.warning(f"Unknown vector backend: {vector_backend}")
            return {"chunks": [], "total": 0}

    def _get_chunks_pgvector(
        self,
        collection_name: str,
        page: int,
        limit: int,
        search: Optional[str] = None,
    ) -> dict:
        """Fetch chunks from pgvector."""
        try:
            from config import DATABASE_URL
            from sqlalchemy import create_engine, text

            engine = create_engine(DATABASE_URL)
            offset = (page - 1) * limit

            with engine.connect() as conn:
                # Get collection UUID
                result = conn.execute(
                    text("SELECT uuid FROM langchain_pg_collection WHERE name = :name"),
                    {"name": collection_name},
                )
                row = result.fetchone()
                if not row:
                    logger.warning(f"pgvector collection not found: {collection_name}")
                    return {"chunks": [], "total": 0}

                collection_uuid = row[0]

                # Build query based on search
                if search:
                    # Search in document content
                    count_query = text("""
                        SELECT COUNT(*) FROM langchain_pg_embedding
                        WHERE collection_id = :uuid
                        AND document ILIKE :search
                    """)
                    data_query = text("""
                        SELECT id, document, cmetadata
                        FROM langchain_pg_embedding
                        WHERE collection_id = :uuid
                        AND document ILIKE :search
                        ORDER BY id
                        LIMIT :limit OFFSET :offset
                    """)
                    params = {
                        "uuid": collection_uuid,
                        "search": f"%{search}%",
                        "limit": limit,
                        "offset": offset,
                    }
                else:
                    count_query = text("""
                        SELECT COUNT(*) FROM langchain_pg_embedding
                        WHERE collection_id = :uuid
                    """)
                    data_query = text("""
                        SELECT id, document, cmetadata
                        FROM langchain_pg_embedding
                        WHERE collection_id = :uuid
                        ORDER BY id
                        LIMIT :limit OFFSET :offset
                    """)
                    params = {"uuid": collection_uuid, "limit": limit, "offset": offset}

                # Get total count
                total_result = conn.execute(
                    count_query,
                    {"uuid": collection_uuid, "search": f"%{search}%" if search else None},
                )
                total = total_result.scalar() or 0

                # Get chunks
                result = conn.execute(data_query, params)
                rows = result.fetchall()

                chunks = []
                for row in rows:
                    chunk_id, content, metadata = row
                    # Parse metadata if it's a string
                    if isinstance(metadata, str):
                        import json
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}

                    chunks.append({
                        "id": str(chunk_id),
                        "content": content,
                        "metadata": metadata or {},
                    })

                return {"chunks": chunks, "total": total}

        except Exception as e:
            logger.error(f"Error fetching chunks from pgvector: {e}", exc_info=True)
            return {"chunks": [], "total": 0, "error": str(e)}

    def _get_chunks_qdrant(
        self,
        collection_name: str,
        page: int,
        limit: int,
        search: Optional[str] = None,
    ) -> dict:
        """Fetch chunks from Qdrant."""
        try:
            from qdrant_client import QdrantClient
            from config import QDRANT_HOST, QDRANT_PORT

            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            offset = (page - 1) * limit

            # Get collection info for total count
            collection_info = client.get_collection(collection_name)
            total = collection_info.points_count

            # Scroll through points
            points, _ = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            chunks = []
            for point in points:
                payload = point.payload or {}
                # Qdrant stores content in 'page_content' or 'text' field
                content = payload.get("page_content") or payload.get("text") or ""
                metadata = payload.get("metadata", {})

                # Filter by search if provided
                if search and search.lower() not in content.lower():
                    continue

                chunks.append({
                    "id": str(point.id),
                    "content": content,
                    "metadata": metadata,
                })

            return {"chunks": chunks, "total": total}

        except Exception as e:
            logger.error(f"Error fetching chunks from Qdrant: {e}", exc_info=True)
            return {"chunks": [], "total": 0, "error": str(e)}

    def _delete_vector_collection(
        self, collection_name: str, vector_backend: str
    ) -> None:
        """Delete a vector store collection."""
        if vector_backend == "pgvector":
            try:
                from config import DATABASE_URL
                from sqlalchemy import create_engine, text

                engine = create_engine(DATABASE_URL)
                with engine.connect() as conn:
                    # Delete from langchain_pg_embedding where collection matches
                    # First get the collection UUID
                    result = conn.execute(
                        text(
                            "SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                        ),
                        {"name": collection_name}
                    )
                    row = result.fetchone()
                    if row:
                        collection_uuid = row[0]
                        # Delete embeddings for this collection
                        conn.execute(
                            text(
                                "DELETE FROM langchain_pg_embedding WHERE collection_id = :uuid"
                            ),
                            {"uuid": collection_uuid}
                        )
                        # Delete the collection record
                        conn.execute(
                            text(
                                "DELETE FROM langchain_pg_collection WHERE uuid = :uuid"
                            ),
                            {"uuid": collection_uuid}
                        )
                        conn.commit()
                        logger.info(f"Deleted pgvector collection: {collection_name}")
                    else:
                        logger.warning(f"pgvector collection not found: {collection_name}")
            except Exception as e:
                logger.error(f"Error deleting pgvector collection {collection_name}: {e}")
                raise
        elif vector_backend == "qdrant":
            try:
                from qdrant_client import QdrantClient
                from config import QDRANT_HOST, QDRANT_PORT

                client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                client.delete_collection(collection_name)
                logger.info(f"Deleted Qdrant collection: {collection_name}")
            except Exception as e:
                logger.error(f"Error deleting Qdrant collection: {e}")
                raise

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
