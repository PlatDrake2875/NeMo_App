"""
Preprocessing Pipeline Service - Orchestrates the document preprocessing workflow.
Implements the pipeline: Load -> Clean -> Extract Metadata -> Chunk -> Index
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from database_models import (
    LLMExtractedMetadata,
    ProcessedDataset,
    ProcessingStatus,
    RawDataset,
    RawFile,
)
from schemas import CleaningConfig, LLMMetadataConfig, PreprocessingConfig
from services.chunking import ChunkingService
from services.file_loader import FileLoaderService
from services.llm_metadata_extractor import LLMMetadataExtractor
from services.processed_dataset import processed_dataset_service
from vectorstore_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class PreprocessingPipelineService:
    """Orchestrates document preprocessing pipeline."""

    def __init__(self):
        self.file_loader = FileLoaderService()
        self.chunking_service = ChunkingService()

    async def process_dataset(
        self,
        db: Session,
        processed_dataset_id: int,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process all files from a raw dataset into a processed dataset.

        Args:
            db: Database session
            processed_dataset_id: ID of the processed dataset to populate
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with processing statistics
        """
        # Get the processed dataset
        processed_ds = (
            db.query(ProcessedDataset)
            .filter(ProcessedDataset.id == processed_dataset_id)
            .first()
        )
        if not processed_ds:
            raise ValueError(f"Processed dataset {processed_dataset_id} not found")

        # Get the raw dataset
        raw_ds = (
            db.query(RawDataset)
            .filter(RawDataset.id == processed_ds.raw_dataset_id)
            .first()
        )
        if not raw_ds:
            raise ValueError(f"Raw dataset {processed_ds.raw_dataset_id} not found")

        # Update status to processing
        processed_dataset_service.update_processing_status(
            db, processed_dataset_id, ProcessingStatus.PROCESSING
        )

        # Parse preprocessing config
        config = PreprocessingConfig.model_validate(processed_ds.preprocessing_config)

        # Initialize metadata extractor if needed
        metadata_extractor = None
        if config.llm_metadata.enabled:
            metadata_extractor = LLMMetadataExtractor(config.llm_metadata.model)

        # Get vector store
        vectorstore = self._get_vectorstore(processed_ds)

        # Process statistics
        stats = {
            "total_files": len(raw_ds.files),
            "processed_files": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "errors": [],
        }

        try:
            # Process each file
            for idx, raw_file in enumerate(raw_ds.files):
                if progress_callback:
                    progress_callback(
                        "processing_file",
                        {
                            "file_index": idx,
                            "total_files": stats["total_files"],
                            "filename": raw_file.filename,
                        },
                    )

                try:
                    file_stats = await self._process_file(
                        db=db,
                        raw_file=raw_file,
                        processed_ds=processed_ds,
                        config=config,
                        metadata_extractor=metadata_extractor,
                        vectorstore=vectorstore,
                    )

                    stats["processed_files"] += 1
                    stats["total_documents"] += file_stats["documents"]
                    stats["total_chunks"] += file_stats["chunks"]

                except Exception as e:
                    logger.error(f"Error processing file {raw_file.filename}: {e}")
                    stats["errors"].append(
                        {"filename": raw_file.filename, "error": str(e)}
                    )

            # Update counts
            processed_dataset_service.update_counts(
                db, processed_dataset_id, stats["total_documents"], stats["total_chunks"]
            )

            # Update status based on whether there were errors
            if stats["errors"]:
                error_msg = f"{len(stats['errors'])} file(s) failed to process"
                processed_dataset_service.update_processing_status(
                    db, processed_dataset_id, ProcessingStatus.COMPLETED, error=error_msg
                )
                logger.warning(f"Dataset {processed_dataset_id} completed with errors: {error_msg}")
            else:
                processed_dataset_service.update_processing_status(
                    db, processed_dataset_id, ProcessingStatus.COMPLETED
                )

            if progress_callback:
                progress_callback("completed", stats)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            processed_dataset_service.update_processing_status(
                db, processed_dataset_id, ProcessingStatus.FAILED, str(e)
            )
            raise

        return stats

    async def process_dataset_stream(
        self, db: Session, processed_dataset_id: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process dataset with streaming progress updates via SSE.

        Yields:
            Dict with progress updates
        """
        progress_queue = asyncio.Queue()

        def progress_callback(event_type: str, data: Dict[str, Any]):
            asyncio.create_task(progress_queue.put({"type": event_type, "data": data}))

        # Start processing in background
        process_task = asyncio.create_task(
            self.process_dataset(db, processed_dataset_id, progress_callback)
        )

        try:
            while not process_task.done():
                try:
                    update = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    yield update
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield {"type": "heartbeat", "data": {}}

            # Get final result
            result = await process_task
            yield {"type": "completed", "data": result}

        except Exception as e:
            yield {"type": "error", "data": {"error": str(e)}}

    async def _process_file(
        self,
        db: Session,
        raw_file: RawFile,
        processed_ds: ProcessedDataset,
        config: PreprocessingConfig,
        metadata_extractor: Optional[LLMMetadataExtractor],
        vectorstore: Any,
    ) -> Dict[str, int]:
        """Process a single file through the pipeline."""
        stats = {"documents": 0, "chunks": 0}

        # Step 1: Load file into documents
        documents = self._load_file(raw_file)
        stats["documents"] = len(documents)

        # Step 2: Clean documents (if enabled)
        if config.cleaning.enabled:
            documents = self._clean_documents(documents, config.cleaning)

        # Step 3: Extract LLM metadata (if enabled)
        if metadata_extractor and config.llm_metadata.enabled:
            documents = await self._extract_metadata(
                db=db,
                documents=documents,
                raw_file=raw_file,
                config=config.llm_metadata,
                extractor=metadata_extractor,
                processed_dataset_id=processed_ds.id,
            )

        # Step 4: Chunk documents
        chunks = self._chunk_documents(documents, config.chunking)
        stats["chunks"] = len(chunks)

        # Add source metadata to chunks
        for chunk in chunks:
            chunk.metadata["raw_file_id"] = raw_file.id
            chunk.metadata["original_filename"] = raw_file.filename
            chunk.metadata["processed_dataset_id"] = processed_ds.id

        # Step 5: Index in vector store
        if chunks:
            self._index_documents(chunks, vectorstore)

        logger.info(
            f"Processed {raw_file.filename}: "
            f"{stats['documents']} docs -> {stats['chunks']} chunks"
        )

        return stats

    def _load_file(self, raw_file: RawFile) -> List[Document]:
        """Load raw file content into LangChain documents."""
        return self.file_loader.load_file(
            content=raw_file.file_content,
            filename=raw_file.filename,
            file_type=raw_file.file_type,
            metadata={
                "raw_file_id": raw_file.id,
                "original_filename": raw_file.filename,
            },
        )

    def _clean_documents(
        self, documents: List[Document], config: CleaningConfig
    ) -> List[Document]:
        """Apply cleaning transformations to documents."""
        import re

        cleaned_docs = []

        for doc in documents:
            text = doc.page_content

            if config.normalize_whitespace:
                # Normalize whitespace
                text = re.sub(r"\s+", " ", text)
                text = re.sub(r"\n\s*\n", "\n\n", text)
                text = text.strip()

            if config.remove_page_numbers:
                # Remove common page number patterns
                text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
                text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

            if config.remove_headers_footers:
                # Remove lines that are very short and appear to be headers/footers
                lines = text.split("\n")
                filtered_lines = [
                    line for line in lines
                    if len(line) > 30 or not line.strip()
                ]
                text = "\n".join(filtered_lines)

            # Apply custom patterns
            for pattern in config.custom_patterns:
                try:
                    text = re.sub(pattern, "", text)
                except re.error:
                    logger.warning(f"Invalid regex pattern: {pattern}")

            if text.strip():
                cleaned_docs.append(
                    Document(page_content=text, metadata=doc.metadata)
                )

        return cleaned_docs

    async def _extract_metadata(
        self,
        db: Session,
        documents: List[Document],
        raw_file: RawFile,
        config: LLMMetadataConfig,
        extractor: LLMMetadataExtractor,
        processed_dataset_id: int,
    ) -> List[Document]:
        """Extract and store LLM metadata, enrich document metadata."""
        # Combine document content for extraction
        combined_content = "\n\n".join(doc.page_content for doc in documents)

        # Extract metadata
        metadata = await extractor.extract_metadata(combined_content, config)

        # Store in database
        llm_metadata = LLMExtractedMetadata(
            raw_file_id=raw_file.id,
            processed_dataset_id=processed_dataset_id,
            extraction_model=config.model,
            summary=metadata.get("summary"),
            keywords=metadata.get("keywords"),
            entities=metadata.get("entities"),
            categories=metadata.get("categories"),
        )
        db.add(llm_metadata)
        db.commit()

        # Enrich document metadata
        for doc in documents:
            if metadata.get("keywords"):
                doc.metadata["keywords"] = metadata["keywords"]
            if metadata.get("categories"):
                doc.metadata["categories"] = metadata["categories"]
            if metadata.get("summary"):
                doc.metadata["document_summary"] = metadata["summary"]

        return documents

    def _chunk_documents(
        self, documents: List[Document], config: Any
    ) -> List[Document]:
        """Chunk documents using configured strategy."""
        return self.chunking_service.chunk_documents(
            documents=documents,
            method=config.method,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def _index_documents(self, chunks: List[Document], vectorstore: Any) -> int:
        """Add chunks to vector store."""
        vectorstore.add_documents(chunks)
        return len(chunks)

    def _get_vectorstore(self, processed_ds: ProcessedDataset) -> Any:
        """Get or create the vector store for the processed dataset."""
        from langchain_huggingface import HuggingFaceEmbeddings

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=processed_ds.embedder_model_name,
            model_kwargs=processed_ds.embedder_model_kwargs or {},
        )

        # Get vectorstore from factory
        return VectorStoreFactory.create_vectorstore(
            backend=processed_ds.vector_backend,
            embeddings=embeddings,
            collection_name=processed_ds.collection_name,
        )


# Singleton instance
preprocessing_pipeline_service = PreprocessingPipelineService()
