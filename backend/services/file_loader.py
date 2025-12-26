"""
File Loader Service - Load various file types into LangChain Documents.
Supports: PDF, JSON, MD, TXT, CSV
"""

import csv
import json
import logging
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FileLoaderService:
    """Load different file types into LangChain Documents."""

    SUPPORTED_TYPES = {"pdf", "json", "md", "txt", "csv"}

    MIME_TYPE_MAP = {
        "application/pdf": "pdf",
        "application/json": "json",
        "text/markdown": "md",
        "text/x-markdown": "md",
        "text/plain": "txt",
        "text/csv": "csv",
        "application/csv": "csv",
    }

    def load_file(
        self,
        content: bytes,
        filename: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load file content into documents based on type.

        Args:
            content: Raw file content as bytes
            filename: Original filename
            file_type: File type (pdf, json, md, txt, csv)
            metadata: Optional additional metadata to include

        Returns:
            List of LangChain Documents
        """
        if file_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported file type: {file_type}. Supported: {self.SUPPORTED_TYPES}")

        base_metadata = {
            "source": filename,
            "file_type": file_type,
        }
        if metadata:
            base_metadata.update(metadata)

        loader_map = {
            "pdf": self._load_pdf,
            "json": self._load_json,
            "md": self._load_markdown,
            "txt": self._load_text,
            "csv": self._load_csv,
        }

        loader = loader_map.get(file_type)
        if not loader:
            raise ValueError(f"No loader found for file type: {file_type}")

        try:
            return loader(content, base_metadata)
        except Exception as e:
            logger.error(f"Error loading {filename} as {file_type}: {e}")
            raise

    def _load_pdf(self, content: bytes, metadata: Dict[str, Any]) -> List[Document]:
        """Load PDF content into documents."""
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error("pypdf not installed. Install with: pip install pypdf")
            raise ImportError("pypdf is required to load PDF files. Install with: pip install pypdf")

        documents = []
        pdf_file = BytesIO(content)
        reader = PdfReader(pdf_file)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                page_metadata = {
                    **metadata,
                    "page": page_num,
                    "total_pages": len(reader.pages),
                }
                documents.append(Document(page_content=text, metadata=page_metadata))

        if not documents:
            # If no text extracted, create a single empty document to track the file
            documents.append(Document(
                page_content="[PDF content could not be extracted]",
                metadata={**metadata, "extraction_error": True}
            ))

        logger.info(f"Loaded PDF with {len(documents)} pages")
        return documents

    def _load_json(self, content: bytes, metadata: Dict[str, Any]) -> List[Document]:
        """Load JSON content into documents."""
        text = content.decode("utf-8")
        data = json.loads(text)

        documents = []

        if isinstance(data, list):
            # Handle array of objects - each becomes a document
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    # Try common text fields
                    text_content = self._extract_text_from_dict(item)
                    if text_content:
                        item_metadata = {
                            **metadata,
                            "json_index": idx,
                            "json_keys": list(item.keys()),
                        }
                        documents.append(Document(page_content=text_content, metadata=item_metadata))
                elif isinstance(item, str):
                    documents.append(Document(
                        page_content=item,
                        metadata={**metadata, "json_index": idx}
                    ))
        elif isinstance(data, dict):
            # Single object - extract text or serialize
            text_content = self._extract_text_from_dict(data)
            if text_content:
                documents.append(Document(page_content=text_content, metadata=metadata))
            else:
                # Fallback: serialize the entire JSON
                documents.append(Document(
                    page_content=json.dumps(data, indent=2),
                    metadata={**metadata, "serialized": True}
                ))
        else:
            # Primitive value - convert to string
            documents.append(Document(page_content=str(data), metadata=metadata))

        if not documents:
            documents.append(Document(
                page_content=json.dumps(data, indent=2),
                metadata={**metadata, "fallback": True}
            ))

        logger.info(f"Loaded JSON with {len(documents)} documents")
        return documents

    def _extract_text_from_dict(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text from a dictionary, trying common text field names."""
        # Common text field names in order of preference
        text_fields = [
            "text", "content", "body", "message", "description",
            "question", "answer", "context", "passage", "document",
            "title", "summary", "abstract"
        ]

        for field in text_fields:
            if field in data and isinstance(data[field], str):
                return data[field]

        # Try to concatenate all string values
        string_values = [str(v) for v in data.values() if isinstance(v, str) and len(str(v)) > 10]
        if string_values:
            return "\n\n".join(string_values)

        return None

    def _load_markdown(self, content: bytes, metadata: Dict[str, Any]) -> List[Document]:
        """Load Markdown content into documents."""
        text = content.decode("utf-8")

        # For markdown, we keep it as a single document
        # The chunking service will handle splitting if needed
        documents = [Document(page_content=text, metadata=metadata)]

        logger.info(f"Loaded Markdown with {len(text)} characters")
        return documents

    def _load_text(self, content: bytes, metadata: Dict[str, Any]) -> List[Document]:
        """Load plain text content into documents."""
        # Try different encodings
        text = None
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError("Could not decode text file with any supported encoding")

        documents = [Document(page_content=text, metadata=metadata)]

        logger.info(f"Loaded text file with {len(text)} characters")
        return documents

    def _load_csv(self, content: bytes, metadata: Dict[str, Any]) -> List[Document]:
        """Load CSV content into documents."""
        text = content.decode("utf-8")
        reader = csv.DictReader(StringIO(text))

        documents = []
        rows = list(reader)

        # Detect text columns (columns with string content > 50 chars on average)
        text_columns = []
        if rows:
            for col in reader.fieldnames or []:
                avg_len = sum(len(str(row.get(col, ""))) for row in rows) / len(rows)
                if avg_len > 50:
                    text_columns.append(col)

        # If no obvious text columns, use all columns
        if not text_columns:
            text_columns = list(reader.fieldnames or [])

        for idx, row in enumerate(rows):
            # Combine text from text columns
            text_parts = []
            for col in text_columns:
                value = row.get(col, "")
                if value:
                    text_parts.append(f"{col}: {value}")

            if text_parts:
                row_metadata = {
                    **metadata,
                    "csv_row": idx,
                    "csv_columns": list(row.keys()),
                }
                documents.append(Document(
                    page_content="\n".join(text_parts),
                    metadata=row_metadata
                ))

        if not documents:
            # Fallback: create single document with CSV content
            documents.append(Document(page_content=text, metadata=metadata))

        logger.info(f"Loaded CSV with {len(documents)} rows")
        return documents

    @classmethod
    def detect_file_type(cls, filename: str, mime_type: Optional[str] = None) -> str:
        """
        Detect file type from filename or MIME type.

        Args:
            filename: The filename with extension
            mime_type: Optional MIME type

        Returns:
            File type string (pdf, json, md, txt, csv)

        Raises:
            ValueError: If file type cannot be determined or is not supported
        """
        # Try MIME type first
        if mime_type and mime_type in cls.MIME_TYPE_MAP:
            return cls.MIME_TYPE_MAP[mime_type]

        # Fall back to extension
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        extension_map = {
            "pdf": "pdf",
            "json": "json",
            "md": "md",
            "markdown": "md",
            "txt": "txt",
            "text": "txt",
            "csv": "csv",
        }

        if ext in extension_map:
            return extension_map[ext]

        raise ValueError(f"Cannot determine file type for: {filename} (mime: {mime_type})")

    @classmethod
    def get_mime_type(cls, file_type: str) -> str:
        """Get the MIME type for a file type."""
        mime_map = {
            "pdf": "application/pdf",
            "json": "application/json",
            "md": "text/markdown",
            "txt": "text/plain",
            "csv": "text/csv",
        }
        return mime_map.get(file_type, "application/octet-stream")


# Singleton instance
file_loader_service = FileLoaderService()
