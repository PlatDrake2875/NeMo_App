"""
Raw Dataset Service - Business logic for raw dataset management.
Handles BLOB storage, file deduplication, and metadata tracking.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from fastapi import UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from database_models import RawDataset, RawFile, SourceType, get_db_session
from schemas import (
    FileTypeEnum,
    RawDatasetCreate,
    RawDatasetInfo,
    RawDatasetListResponse,
    RawFileInfo,
    SourceTypeEnum,
)
from services.file_loader import FileLoaderService

logger = logging.getLogger(__name__)


class RawDatasetService:
    """Service for managing raw datasets and their files."""

    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB default

    def __init__(self):
        self.file_loader = FileLoaderService()

    def create_dataset(self, db: Session, request: RawDatasetCreate) -> RawDatasetInfo:
        """Create a new empty raw dataset container."""
        # Check if name already exists
        existing = db.query(RawDataset).filter(RawDataset.name == request.name).first()
        if existing:
            raise ValueError(f"Dataset with name '{request.name}' already exists")

        dataset = RawDataset(
            name=request.name,
            description=request.description,
            source_type=request.source_type.value,
            total_file_count=0,
            total_size_bytes=0,
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        logger.info(f"Created raw dataset: {dataset.name} (id={dataset.id})")
        return self._to_dataset_info(dataset, include_files=False)

    def list_datasets(
        self, db: Session, include_files: bool = False
    ) -> RawDatasetListResponse:
        """List all raw datasets."""
        datasets = db.query(RawDataset).order_by(RawDataset.created_at.desc()).all()

        dataset_infos = [
            self._to_dataset_info(ds, include_files=include_files) for ds in datasets
        ]

        return RawDatasetListResponse(count=len(dataset_infos), datasets=dataset_infos)

    def get_dataset(
        self, db: Session, dataset_id: int, include_files: bool = True
    ) -> Optional[RawDatasetInfo]:
        """Get a raw dataset by ID."""
        dataset = db.query(RawDataset).filter(RawDataset.id == dataset_id).first()
        if not dataset:
            return None

        return self._to_dataset_info(dataset, include_files=include_files)

    def get_dataset_by_name(
        self, db: Session, name: str, include_files: bool = True
    ) -> Optional[RawDatasetInfo]:
        """Get a raw dataset by name."""
        dataset = db.query(RawDataset).filter(RawDataset.name == name).first()
        if not dataset:
            return None

        return self._to_dataset_info(dataset, include_files=include_files)

    def delete_dataset(self, db: Session, dataset_id: int) -> dict:
        """
        Delete a raw dataset and all its files.

        Args:
            db: Database session
            dataset_id: ID of the dataset to delete

        Returns:
            Dict with deletion confirmation and file count

        Raises:
            ValueError: If dataset with given ID does not exist
        """
        dataset = db.query(RawDataset).filter(RawDataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset with id {dataset_id} not found")

        name = dataset.name
        file_count = dataset.total_file_count

        # Cascade delete will handle files
        db.delete(dataset)
        db.commit()

        logger.info(f"Deleted raw dataset: {name} (id={dataset_id}, files={file_count})")
        return {
            "message": f"Dataset '{name}' deleted successfully",
            "files_deleted": file_count,
        }

    async def add_file(
        self, db: Session, dataset_id: int, file: UploadFile
    ) -> RawFileInfo:
        """
        Add a file to a raw dataset.

        Args:
            db: Database session
            dataset_id: ID of the raw dataset
            file: Uploaded file

        Returns:
            RawFileInfo with file details

        Raises:
            ValueError: If dataset not found, file type not supported, or file too large
        """
        dataset = db.query(RawDataset).filter(RawDataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset with id {dataset_id} not found")

        # Validate and detect file type
        filename = file.filename or "unknown"
        try:
            file_type = self.file_loader.detect_file_type(filename, file.content_type)
        except ValueError as e:
            raise ValueError(f"Unsupported file type: {e}")

        # Check file size before reading into memory (if Content-Length available)
        if hasattr(file, 'size') and file.size:
            if file.size > self.MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"File too large: {file.size} bytes. "
                    f"Maximum: {self.MAX_FILE_SIZE_BYTES} bytes"
                )

        # Read file content with streaming size check
        # Use bytearray for efficient memory usage (avoids O(n^2) byte concatenation)
        content_buffer = bytearray()
        chunk_size = 8192  # 8KB chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            # Check size before extending to fail fast
            if len(content_buffer) + len(chunk) > self.MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"File too large (>{self.MAX_FILE_SIZE_BYTES} bytes). "
                    f"Maximum: {self.MAX_FILE_SIZE_BYTES} bytes"
                )
            content_buffer.extend(chunk)
        content = bytes(content_buffer)

        # Compute hash for deduplication
        content_hash = self._compute_hash(content)

        # Check for duplicate in this dataset
        existing = (
            db.query(RawFile)
            .filter(
                RawFile.raw_dataset_id == dataset_id,
                RawFile.content_hash == content_hash,
            )
            .first()
        )
        if existing:
            logger.warning(
                f"Duplicate file detected: {filename} (hash={content_hash[:16]}...)"
            )
            # Return existing file info instead of creating duplicate
            return self._to_file_info(existing)

        # Get MIME type
        mime_type = file.content_type or self.file_loader.get_mime_type(file_type)

        # Create file record
        raw_file = RawFile(
            raw_dataset_id=dataset_id,
            filename=filename,
            file_type=file_type,
            mime_type=mime_type,
            size_bytes=len(content),
            file_content=content,
            content_hash=content_hash,
            metadata_json={},
        )
        db.add(raw_file)

        # Update dataset stats
        dataset.total_file_count += 1
        dataset.total_size_bytes += len(content)
        dataset.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(raw_file)

        logger.info(
            f"Added file to dataset {dataset.name}: {filename} "
            f"({file_type}, {len(content)} bytes)"
        )
        return self._to_file_info(raw_file)

    async def add_files_batch(
        self, db: Session, dataset_id: int, files: List[UploadFile]
    ) -> dict:
        """
        Add multiple files to a raw dataset.

        Returns:
            Dict with 'successful' (list of RawFileInfo) and 'failed' (list of error dicts)
        """
        results = {"successful": [], "failed": []}
        for file in files:
            try:
                file_info = await self.add_file(db, dataset_id, file)
                results["successful"].append(file_info)
            except ValueError as e:
                logger.warning(f"Validation failed for {file.filename}: {e}")
                results["failed"].append({
                    "filename": file.filename,
                    "error": str(e)
                })
            except Exception as e:
                logger.error(f"Error adding file {file.filename}: {e}", exc_info=True)
                results["failed"].append({
                    "filename": file.filename,
                    "error": f"Internal error: {str(e)}"
                })
        return results

    def delete_file(self, db: Session, dataset_id: int, file_id: int) -> dict:
        """Delete a specific file from a raw dataset."""
        raw_file = (
            db.query(RawFile)
            .filter(RawFile.id == file_id, RawFile.raw_dataset_id == dataset_id)
            .first()
        )
        if not raw_file:
            raise ValueError(f"File with id {file_id} not found in dataset {dataset_id}")

        dataset = raw_file.dataset
        filename = raw_file.filename
        size = raw_file.size_bytes

        # Update dataset stats
        dataset.total_file_count -= 1
        dataset.total_size_bytes -= size
        dataset.updated_at = datetime.now(timezone.utc)

        db.delete(raw_file)
        db.commit()

        logger.info(f"Deleted file from dataset {dataset.name}: {filename}")
        return {"message": f"File '{filename}' deleted successfully"}

    def get_file_content(
        self, db: Session, dataset_id: int, file_id: int
    ) -> Tuple[bytes, str, str]:
        """
        Get the content of a file.

        Returns:
            Tuple of (content_bytes, filename, mime_type)
        """
        raw_file = (
            db.query(RawFile)
            .filter(RawFile.id == file_id, RawFile.raw_dataset_id == dataset_id)
            .first()
        )
        if not raw_file:
            raise ValueError(f"File with id {file_id} not found in dataset {dataset_id}")

        return raw_file.file_content, raw_file.filename, raw_file.mime_type

    def get_file_info(
        self, db: Session, dataset_id: int, file_id: int
    ) -> Optional[RawFileInfo]:
        """Get file info without content."""
        raw_file = (
            db.query(RawFile)
            .filter(RawFile.id == file_id, RawFile.raw_dataset_id == dataset_id)
            .first()
        )
        if not raw_file:
            return None
        return self._to_file_info(raw_file)

    def update_dataset_stats(self, db: Session, dataset_id: int) -> None:
        """Recalculate and update dataset statistics."""
        dataset = db.query(RawDataset).filter(RawDataset.id == dataset_id).first()
        if not dataset:
            return

        # Recalculate stats
        stats = (
            db.query(
                func.count(RawFile.id).label("count"),
                func.coalesce(func.sum(RawFile.size_bytes), 0).label("total_size"),
            )
            .filter(RawFile.raw_dataset_id == dataset_id)
            .first()
        )

        dataset.total_file_count = stats.count
        dataset.total_size_bytes = stats.total_size
        dataset.updated_at = datetime.now(timezone.utc)

        db.commit()

    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash for deduplication."""
        return hashlib.sha256(content).hexdigest()

    def _to_dataset_info(
        self, dataset: RawDataset, include_files: bool = False
    ) -> RawDatasetInfo:
        """Convert SQLAlchemy model to Pydantic schema."""
        files = []
        if include_files:
            files = [self._to_file_info(f) for f in dataset.files]

        return RawDatasetInfo(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            source_type=SourceTypeEnum(dataset.source_type),
            source_identifier=dataset.source_identifier,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            total_file_count=dataset.total_file_count,
            total_size_bytes=dataset.total_size_bytes,
            files=files,
        )

    def _to_file_info(self, raw_file: RawFile) -> RawFileInfo:
        """Convert SQLAlchemy model to Pydantic schema."""
        return RawFileInfo(
            id=raw_file.id,
            filename=raw_file.filename,
            file_type=FileTypeEnum(raw_file.file_type),
            mime_type=raw_file.mime_type,
            size_bytes=raw_file.size_bytes,
            uploaded_at=raw_file.uploaded_at,
            content_hash=raw_file.content_hash,
            metadata=raw_file.metadata_json,
        )


# Singleton instance
raw_dataset_service = RawDatasetService()
