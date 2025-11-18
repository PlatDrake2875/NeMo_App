"""
Upload service layer containing all document upload business logic.
Separated from the web layer for better testability and maintainability.
"""

import os
from typing import Optional

from fastapi import HTTPException, UploadFile
from langchain_postgres.vectorstores import PGVector as LangchainPGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_embedder_manager import get_multi_embedder_manager
from schemas import UploadResponse
from services.chunking import ChunkingService
from services.dataset_registry import DatasetRegistryService


class UploadService:
    """Service class handling all document upload business logic."""

    def __init__(self):
        self.temp_dir = "/app/temp_uploads"
        self.supported_content_types = {"application/pdf"}
        self.chunking_service = ChunkingService()

    async def process_document_upload(
        self,
        file: UploadFile,
        default_vectorstore: LangchainPGVector,
        dataset_name: Optional[str] = None,
        chunking_method: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **chunking_kwargs
    ) -> UploadResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        original_filename = file.filename
        temp_file_path = self._prepare_temp_file_path(original_filename)

        # Determine which vectorstore to use
        if dataset_name:
            # Get dataset info and create appropriate vectorstore
            dataset_registry = DatasetRegistryService()
            dataset_info = dataset_registry.get_dataset(dataset_name)

            # Get vectorstore for this dataset
            multi_embedder_manager = get_multi_embedder_manager()
            vectorstore = multi_embedder_manager.get_vectorstore(
                collection_name=dataset_info.collection_name,
                embedder_config=dataset_info.embedder_config
            )
        else:
            # Use default vectorstore
            vectorstore = default_vectorstore

        file_content = await self._read_and_validate_file(file)
        self._save_temp_file(file_content, temp_file_path)
        docs = self._load_document(file, temp_file_path, original_filename)
        chunks = self._split_document_into_chunks(
            docs,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **chunking_kwargs
        )
        self._add_to_vector_store(chunks, vectorstore)

        # Update dataset counts if using a specific dataset
        if dataset_name:
            dataset_registry.update_dataset_counts(dataset_name)

        self._cleanup_temp_file(temp_file_path)
        await file.close()

        return UploadResponse(
            message=f"Document processed with {chunking_method} chunking and added to {'dataset ' + dataset_name if dataset_name else 'vector store'} successfully.",
            filename=original_filename,
            chunks_added=len(chunks),
        )

    def _prepare_temp_file_path(self, filename: str) -> str:
        """Prepare the temporary file path."""
        os.makedirs(self.temp_dir, exist_ok=True)
        return os.path.join(self.temp_dir, filename)

    async def _read_and_validate_file(self, file: UploadFile) -> bytes:
        """Read and validate the uploaded file content."""
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return file_content

    def _save_temp_file(self, file_content: bytes, temp_file_path: str) -> None:
        """Save the file content to a temporary file."""
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)

    def _load_document(
        self, file: UploadFile, temp_file_path: str, original_filename: str
    ) -> list:
        """Load the document based on its content type."""
        if file.content_type not in self.supported_content_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Only PDF is currently supported.",
            )

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the document.",
            )

        for doc in docs:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["original_filename"] = original_filename

        return docs

    def _split_document_into_chunks(
        self,
        docs: list,
        chunking_method: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **chunking_kwargs
    ) -> list:
        """Split the document into chunks for vectorization using specified method."""
        chunks = self.chunking_service.chunk_documents(
            docs,
            method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **chunking_kwargs
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted for vectorization.",
            )

        return chunks

    def _add_to_vector_store(self, chunks: list, vectorstore: LangchainPGVector) -> None:
        """Add the document chunks to the vector store."""
        vectorstore.add_documents(chunks)

    def _cleanup_temp_file(self, temp_file_path: str) -> None:
        """Clean up the temporary file."""
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def main():
    """Test the UploadService with a mock PDF document."""
    import shutil
    import tempfile
    from io import BytesIO

    from langchain_huggingface import HuggingFaceEmbeddings
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    test_temp_dir = tempfile.mkdtemp(prefix="upload_service_test_")

    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        story = [
            Paragraph("Test Document for Upload Service", styles["Title"]),
            Paragraph(
                "This is a test document created for testing the upload service functionality.",
                styles["Normal"],
            ),
            Paragraph(
                "It contains multiple paragraphs to test text splitting and chunking.",
                styles["Normal"],
            ),
            Paragraph(
                "The document should be processed, split into chunks, and added to a vector store.",
                styles["Normal"],
            ),
            Paragraph(
                "This paragraph contains some additional content to ensure we have enough text for meaningful chunks.",
                styles["Normal"],
            ),
        ]

        doc.build(story)
        pdf_content = pdf_buffer.getvalue()
        pdf_buffer.close()

        class MockUploadFile:
            def __init__(self, content: bytes, filename: str, content_type: str):
                self.content = content
                self.filename = filename
                self.content_type = content_type
                self._closed = False

            async def read(self) -> bytes:
                if self._closed:
                    raise ValueError("File is closed")
                return self.content

            async def close(self):
                self._closed = True

        mock_file = MockUploadFile(
            content=pdf_content,
            filename="test_document.pdf",
            content_type="application/pdf",
        )

        upload_service = UploadService()
        upload_service.temp_dir = test_temp_dir

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Note: For testing, you'd need a real PostgreSQL connection string
        # vectorstore = LangchainPGVector(
        #     connection="postgresql+psycopg://user:pass@localhost:5432/db",
        #     embeddings=embeddings,
        #     collection_name="test_collection"
        # )

        # Skipping actual upload test as it requires PostgreSQL connection
        # await upload_service.process_document_upload(mock_file, vectorstore)

    except Exception:
        pass

    finally:
        try:
            shutil.rmtree(test_temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
