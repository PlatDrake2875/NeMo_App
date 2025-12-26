"""
HuggingFace Dataset Service - Integration with HuggingFace Datasets Hub.
Download and process datasets from HuggingFace.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from config import HUGGING_FACE_HUB_TOKEN
from database_models import RawDataset, RawFile, SourceType
from schemas import (
    HFDatasetConfig,
    HFDatasetMetadata,
    HFDirectProcessRequest,
    HFImportAsRawRequest,
    ProcessedDatasetCreate,
    RawDatasetInfo,
    SourceTypeEnum,
)
from services.raw_dataset import raw_dataset_service
from services.processed_dataset import processed_dataset_service
from services.preprocessing_pipeline import preprocessing_pipeline_service

logger = logging.getLogger(__name__)


class HuggingFaceDatasetService:
    """Service for HuggingFace dataset operations."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or HUGGING_FACE_HUB_TOKEN

    async def get_dataset_metadata(self, dataset_id: str) -> HFDatasetMetadata:
        """
        Fetch dataset metadata from HuggingFace.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., 'squad', 'microsoft/wiki_qa')

        Returns:
            HFDatasetMetadata with dataset information
        """
        try:
            from datasets import load_dataset_builder

            # Run in executor to not block
            loop = asyncio.get_event_loop()
            builder = await loop.run_in_executor(
                None,
                lambda: load_dataset_builder(dataset_id, token=self.token),
            )

            info = builder.info

            # Get features as string descriptions
            features = {}
            if info.features:
                for name, feature in info.features.items():
                    features[name] = str(type(feature).__name__)

            # Get row counts per split
            num_rows = {}
            if info.splits:
                for split_name, split_info in info.splits.items():
                    num_rows[split_name] = split_info.num_examples

            return HFDatasetMetadata(
                dataset_id=dataset_id,
                description=info.description,
                size_bytes=info.size_in_bytes,
                num_rows=num_rows,
                features=features,
                available_splits=list(num_rows.keys()),
            )

        except Exception as e:
            logger.error(f"Error fetching HF dataset metadata for {dataset_id}: {e}")
            raise ValueError(f"Failed to fetch dataset metadata: {e}")

    async def validate_dataset(
        self, dataset_id: str
    ) -> tuple[bool, Optional[str]]:
        """
        Validate if dataset exists and is accessible.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            await self.get_dataset_metadata(dataset_id)
            return True, None
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {e}"

    async def import_as_raw(
        self, db: Session, request: HFImportAsRawRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Import HuggingFace dataset as raw dataset with progress streaming.

        Yields:
            Progress updates as dicts
        """
        config = request.hf_config

        yield {"type": "status", "message": "Validating dataset..."}

        # Validate dataset
        is_valid, error = await self.validate_dataset(config.dataset_id)
        if not is_valid:
            yield {"type": "error", "message": error}
            return

        yield {"type": "status", "message": "Creating raw dataset..."}

        # Create raw dataset
        try:
            from schemas import RawDatasetCreate

            raw_ds_create = RawDatasetCreate(
                name=request.raw_dataset_name,
                description=request.description or f"Imported from HuggingFace: {config.dataset_id}",
                source_type=SourceTypeEnum.HUGGINGFACE,
            )

            # Create the raw dataset record
            raw_dataset = RawDataset(
                name=raw_ds_create.name,
                description=raw_ds_create.description,
                source_type=SourceType.HUGGINGFACE.value,
                source_identifier=config.dataset_id,
                total_file_count=0,
                total_size_bytes=0,
            )
            db.add(raw_dataset)
            db.commit()
            db.refresh(raw_dataset)

        except Exception as e:
            yield {"type": "error", "message": f"Failed to create raw dataset: {e}"}
            return

        yield {"type": "status", "message": "Loading dataset from HuggingFace..."}

        try:
            from datasets import load_dataset
            import hashlib

            # Load dataset
            loop = asyncio.get_event_loop()

            load_kwargs = {
                "path": config.dataset_id,
                "split": config.split,
                "token": config.token or self.token,
            }
            if config.subset:
                load_kwargs["name"] = config.subset

            dataset = await loop.run_in_executor(
                None,
                lambda: load_dataset(**load_kwargs),
            )

            total_rows = len(dataset)
            if config.max_samples:
                total_rows = min(total_rows, config.max_samples)

            yield {
                "type": "progress",
                "message": f"Processing {total_rows} samples...",
                "total": total_rows,
                "current": 0,
            }

            # Process rows and create files
            batch_size = 100
            processed = 0

            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                batch_data = []

                for j in range(i, batch_end):
                    row = dataset[j]
                    # Extract text from the configured column
                    text = row.get(config.text_column, "")
                    if not text and isinstance(row, dict):
                        # Try to find any text content
                        for value in row.values():
                            if isinstance(value, str) and len(value) > 10:
                                text = value
                                break

                    if text:
                        batch_data.append({
                            "index": j,
                            "text": str(text),
                            "metadata": {
                                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                                for k, v in row.items()
                                if k != config.text_column
                            },
                        })

                # Create a JSON file for this batch
                if batch_data:
                    content = json.dumps(batch_data, indent=2).encode("utf-8")
                    content_hash = hashlib.sha256(content).hexdigest()

                    raw_file = RawFile(
                        raw_dataset_id=raw_dataset.id,
                        filename=f"batch_{i:06d}_{batch_end:06d}.json",
                        file_type="json",
                        mime_type="application/json",
                        size_bytes=len(content),
                        file_content=content,
                        content_hash=content_hash,
                        metadata_json={"batch_start": i, "batch_end": batch_end},
                    )
                    db.add(raw_file)

                    raw_dataset.total_file_count += 1
                    raw_dataset.total_size_bytes += len(content)

                processed = batch_end
                yield {
                    "type": "progress",
                    "message": f"Processed {processed}/{total_rows} samples",
                    "total": total_rows,
                    "current": processed,
                }

            db.commit()

            yield {
                "type": "completed",
                "message": f"Successfully imported {processed} samples",
                "raw_dataset_id": raw_dataset.id,
                "raw_dataset_name": raw_dataset.name,
            }

        except Exception as e:
            logger.error(f"Error importing HF dataset: {e}")
            # Clean up on error
            db.delete(raw_dataset)
            db.commit()
            yield {"type": "error", "message": f"Import failed: {e}"}

    async def process_direct(
        self, db: Session, request: HFDirectProcessRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Download and process HuggingFace dataset directly to vector store.

        Yields:
            Progress updates as dicts
        """
        yield {"type": "status", "message": "Starting direct processing..."}

        # First, import as raw
        import_request = HFImportAsRawRequest(
            hf_config=request.hf_config,
            raw_dataset_name=f"_temp_{request.processed_dataset_name}",
            description=f"Temporary raw dataset for {request.processed_dataset_name}",
        )

        raw_dataset_id = None
        async for update in self.import_as_raw(db, import_request):
            if update["type"] == "error":
                yield update
                return
            elif update["type"] == "completed":
                raw_dataset_id = update["raw_dataset_id"]
            else:
                yield update

        if not raw_dataset_id:
            yield {"type": "error", "message": "Failed to import raw dataset"}
            return

        yield {"type": "status", "message": "Creating processed dataset..."}

        try:
            # Create processed dataset
            create_request = ProcessedDatasetCreate(
                name=request.processed_dataset_name,
                description=request.description,
                raw_dataset_id=raw_dataset_id,
                embedder_config=request.embedder_config,
                preprocessing_config=request.preprocessing_config,
                vector_backend=request.vector_backend,
            )

            processed_ds = processed_dataset_service.create_dataset(db, create_request)

            yield {"type": "status", "message": "Processing and indexing..."}

            # Run preprocessing pipeline
            def progress_callback(event_type: str, data: Dict[str, Any]):
                pass  # We'll handle progress differently for streaming

            stats = await preprocessing_pipeline_service.process_dataset(
                db, processed_ds.id, progress_callback
            )

            yield {
                "type": "completed",
                "message": f"Successfully processed and indexed",
                "processed_dataset_id": processed_ds.id,
                "processed_dataset_name": processed_ds.name,
                "stats": stats,
            }

        except Exception as e:
            logger.error(f"Error in direct processing: {e}")
            yield {"type": "error", "message": f"Processing failed: {e}"}

    async def search_datasets(
        self, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets on HuggingFace Hub.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of dataset info dicts
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)

            loop = asyncio.get_event_loop()
            datasets = await loop.run_in_executor(
                None,
                lambda: list(api.list_datasets(search=query, limit=limit)),
            )

            results = []
            for ds in datasets:
                results.append({
                    "id": ds.id,
                    "author": ds.author,
                    "downloads": ds.downloads,
                    "likes": ds.likes,
                    "tags": ds.tags[:10] if ds.tags else [],
                    "description": getattr(ds, "description", None),
                })

            return results

        except Exception as e:
            logger.error(f"Error searching HF datasets: {e}")
            return []


# Factory function
def create_huggingface_service(
    token: Optional[str] = None,
) -> HuggingFaceDatasetService:
    """Create a HuggingFace dataset service with optional custom token."""
    return HuggingFaceDatasetService(token=token)


# Default singleton instance
huggingface_dataset_service = HuggingFaceDatasetService()
