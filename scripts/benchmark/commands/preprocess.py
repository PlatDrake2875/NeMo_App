"""Preprocess command - create processed dataset from files or raw dataset."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.benchmark.client import BenchmarkAPIClient
from scripts.benchmark.config import (
    EmbedderConfig,
    PreprocessingConfig,
    PreprocessResult,
)
from scripts.benchmark.utils.polling import poll_until_complete

logger = logging.getLogger(__name__)


async def preprocess(
    client: BenchmarkAPIClient,
    name: str,
    files: list[Path] | None = None,
    raw_dataset_id: int | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    embedder_config: EmbedderConfig | None = None,
    vector_backend: str = "pgvector",
    timeout: float = 3600.0,
    progress_callback: Callable[[dict], None] | None = None,
) -> PreprocessResult:
    """
    Create a processed dataset from files or an existing raw dataset.

    This is Stage 1 of the benchmark pipeline. It:
    1. Creates or uses an existing raw dataset
    2. Creates a processed dataset with specified config
    3. Starts and waits for processing to complete

    Args:
        client: API client instance
        name: Name for the datasets
        files: List of file paths to upload (mutually exclusive with raw_dataset_id)
        raw_dataset_id: Existing raw dataset ID (mutually exclusive with files)
        preprocessing_config: Preprocessing configuration
        embedder_config: Embedder configuration
        vector_backend: Vector store backend (pgvector|qdrant)
        timeout: Maximum time to wait for processing
        progress_callback: Optional callback for progress updates

    Returns:
        PreprocessResult with raw_dataset_id and processed_dataset_id

    Raises:
        ValueError: If neither files nor raw_dataset_id is provided
    """
    if files is None and raw_dataset_id is None:
        raise ValueError("Either 'files' or 'raw_dataset_id' must be provided")
    if files is not None and raw_dataset_id is not None:
        raise ValueError("Cannot provide both 'files' and 'raw_dataset_id'")

    # Step 1: Create or use raw dataset
    if files:
        logger.info(f"Creating raw dataset '{name}' with {len(files)} files")
        raw_dataset = await client.create_raw_dataset(name=name)
        raw_id = raw_dataset["id"]
        logger.info(f"Created raw dataset with ID: {raw_id}")

        # Upload files
        for file_path in files:
            logger.info(f"Uploading file: {file_path.name}")
            await client.upload_file(raw_id, file_path)
        logger.info(f"Uploaded {len(files)} files")
    else:
        raw_id = raw_dataset_id
        logger.info(f"Using existing raw dataset ID: {raw_id}")

    # Step 2: Create processed dataset
    logger.info(f"Creating processed dataset '{name}'")
    processed_dataset = await client.create_processed_dataset(
        raw_dataset_id=raw_id,
        name=name,
        preprocessing_config=preprocessing_config,
        embedder_config=embedder_config,
        vector_backend=vector_backend,
    )
    processed_id = processed_dataset["id"]
    logger.info(f"Created processed dataset with ID: {processed_id}")

    # Step 3: Start processing
    logger.info("Starting processing...")
    await client.start_processing(processed_id)

    # Step 4: Poll until complete
    def status_callback(status: dict) -> None:
        processing_status = status.get("status", "unknown")
        if progress_callback:
            progress_callback(status)
        logger.debug(f"Processing status: {processing_status}")

    final_status = await poll_until_complete(
        client.get_processing_status,
        processed_id,
        timeout=timeout,
        interval=2.0,
        status_field="status",
        complete_statuses={"completed"},
        failed_statuses={"failed"},
        error_field="error",
        progress_callback=status_callback,
    )

    # Get final dataset info for counts
    dataset_info = await client.get_processed_dataset(processed_id)

    logger.info(
        f"Processing completed. Documents: {dataset_info.get('document_count', 0)}, "
        f"Chunks: {dataset_info.get('chunk_count', 0)}"
    )

    return PreprocessResult(
        raw_dataset_id=raw_id,
        processed_dataset_id=processed_id,
        document_count=dataset_info.get("document_count", 0),
        chunk_count=dataset_info.get("chunk_count", 0),
    )
