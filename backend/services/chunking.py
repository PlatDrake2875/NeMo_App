"""
Chunking service with multiple text splitting strategies.
Provides factory pattern for selecting and configuring different chunking methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings

# Try to import SemanticChunker, but make it optional
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False
    SemanticChunker = None


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration used for this chunking strategy."""
        pass


class RecursiveChunking(ChunkingStrategy):
    """Recursive character text splitting strategy."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        # Add chunking metadata
        for chunk in chunks:
            if not chunk.metadata:
                chunk.metadata = {}
            chunk.metadata.update(self.get_config())
        return chunks

    def get_config(self) -> Dict[str, Any]:
        return {
            "chunking_method": "recursive",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size character chunking strategy."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        # Add chunking metadata
        for chunk in chunks:
            if not chunk.metadata:
                chunk.metadata = {}
            chunk.metadata.update(self.get_config())
        return chunks

    def get_config(self) -> Dict[str, Any]:
        return {
            "chunking_method": "fixed",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


class SemanticChunking(ChunkingStrategy):
    """Semantic chunking strategy using embeddings.

    Includes post-processing to enforce max_chunk_size, since semantic
    chunking can produce arbitrarily large chunks based on similarity.
    """

    def __init__(
        self,
        embedder_model: str = "all-MiniLM-L6-v2",
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        max_chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ):
        if not SEMANTIC_CHUNKING_AVAILABLE:
            raise ImportError(
                "Semantic chunking requires langchain_experimental package. "
                "Install it with: pip install langchain-experimental"
            )

        self.embedder_model = embedder_model
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedder_model)

        # Create semantic chunker
        if breakpoint_threshold_amount is not None:
            self.splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
            )
        else:
            self.splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type,
            )

        # Fallback splitter for oversized chunks
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        semantic_chunks = self.splitter.split_documents(documents)

        # Post-process: split oversized chunks using fallback splitter
        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk.page_content) > self.max_chunk_size:
                sub_chunks = self._fallback_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        # Add chunking metadata
        for chunk in final_chunks:
            if not chunk.metadata:
                chunk.metadata = {}
            chunk.metadata.update(self.get_config())
        return final_chunks

    def get_config(self) -> Dict[str, Any]:
        config = {
            "chunking_method": "semantic",
            "embedder_model": self.embedder_model,
            "breakpoint_threshold_type": self.breakpoint_threshold_type,
            "max_chunk_size": self.max_chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if self.breakpoint_threshold_amount is not None:
            config["breakpoint_threshold_amount"] = self.breakpoint_threshold_amount
        return config


class ChunkingService:
    """Service for managing different chunking strategies."""

    @staticmethod
    def _get_available_methods_dict():
        """Get available methods dynamically based on installed packages."""
        methods = {
            "recursive": {
                "name": "Recursive Character Splitter",
                "description": "Splits text recursively by trying different separators",
                "default_params": {"chunk_size": 1000, "chunk_overlap": 200},
            },
            "fixed": {
                "name": "Fixed-Size Chunking",
                "description": "Simple fixed-size chunks with configurable overlap",
                "default_params": {"chunk_size": 1000, "chunk_overlap": 200},
            },
        }

        # Only add semantic chunking if available
        if SEMANTIC_CHUNKING_AVAILABLE:
            methods["semantic"] = {
                "name": "Semantic Chunking",
                "description": "Chunks based on semantic similarity using embeddings",
                "default_params": {
                    "embedder_model": "all-MiniLM-L6-v2",
                    "breakpoint_threshold_type": "percentile",
                },
            }

        return methods

    AVAILABLE_METHODS = _get_available_methods_dict.__func__()

    @staticmethod
    def get_chunking_strategy(
        method: str = "recursive", **kwargs
    ) -> ChunkingStrategy:
        """
        Factory method to create a chunking strategy.

        Args:
            method: The chunking method ('recursive', 'fixed', 'semantic')
            **kwargs: Additional parameters specific to the chunking method

        Returns:
            ChunkingStrategy instance
        """
        if method == "recursive":
            chunk_size = kwargs.get("chunk_size", 1000)
            chunk_overlap = kwargs.get("chunk_overlap", 200)
            return RecursiveChunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        elif method == "fixed":
            chunk_size = kwargs.get("chunk_size", 1000)
            chunk_overlap = kwargs.get("chunk_overlap", 200)
            return FixedSizeChunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        elif method == "semantic":
            embedder_model = kwargs.get("embedder_model", "all-MiniLM-L6-v2")
            breakpoint_threshold_type = kwargs.get("breakpoint_threshold_type", "percentile")
            breakpoint_threshold_amount = kwargs.get("breakpoint_threshold_amount")
            # Pass chunk_size/overlap for enforcing max size on semantic chunks
            max_chunk_size = kwargs.get("chunk_size", 2000)
            chunk_overlap = kwargs.get("chunk_overlap", 200)
            return SemanticChunking(
                embedder_model=embedder_model,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )

        else:
            raise ValueError(f"Unknown chunking method: {method}")

    @staticmethod
    def get_available_methods() -> Dict[str, Dict[str, Any]]:
        """Get information about all available chunking methods."""
        return ChunkingService.AVAILABLE_METHODS

    @staticmethod
    def chunk_documents(
        documents: List[Document], method: str = "recursive", **kwargs
    ) -> List[Document]:
        """
        Convenience method to chunk documents using a specified method.

        Args:
            documents: List of documents to chunk
            method: The chunking method to use
            **kwargs: Additional parameters for the chunking method

        Returns:
            List of chunked documents
        """
        strategy = ChunkingService.get_chunking_strategy(method, **kwargs)
        return strategy.split_documents(documents)
