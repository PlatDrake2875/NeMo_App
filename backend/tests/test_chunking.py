"""Tests for ChunkingService - testing chunking configuration and strategy selection."""

import pytest
from langchain_core.documents import Document

from services.chunking import (
    ChunkingService,
    RecursiveChunking,
    FixedSizeChunking,
    SEMANTIC_CHUNKING_AVAILABLE,
)


class TestRecursiveChunkingConfig:
    """Tests for RecursiveChunking.get_config()."""

    def test_get_config_returns_correct_structure(self):
        """get_config should return dict with method, size, and overlap."""
        strategy = RecursiveChunking(chunk_size=500, chunk_overlap=100)
        config = strategy.get_config()

        assert config["chunking_method"] == "recursive"
        assert config["chunk_size"] == 500
        assert config["chunk_overlap"] == 100

    def test_get_config_uses_default_values(self):
        """Default values should be 1000 size and 200 overlap."""
        strategy = RecursiveChunking()
        config = strategy.get_config()

        assert config["chunk_size"] == 1000
        assert config["chunk_overlap"] == 200

    def test_get_config_custom_values(self):
        """Custom values should be reflected in config."""
        strategy = RecursiveChunking(chunk_size=2000, chunk_overlap=400)
        config = strategy.get_config()

        assert config["chunk_size"] == 2000
        assert config["chunk_overlap"] == 400


class TestFixedSizeChunkingConfig:
    """Tests for FixedSizeChunking.get_config()."""

    def test_get_config_returns_correct_structure(self):
        """get_config should return dict with method, size, and overlap."""
        strategy = FixedSizeChunking(chunk_size=500, chunk_overlap=100)
        config = strategy.get_config()

        assert config["chunking_method"] == "fixed"
        assert config["chunk_size"] == 500
        assert config["chunk_overlap"] == 100

    def test_get_config_uses_default_values(self):
        """Default values should be 1000 size and 200 overlap."""
        strategy = FixedSizeChunking()
        config = strategy.get_config()

        assert config["chunk_size"] == 1000
        assert config["chunk_overlap"] == 200


class TestSemanticChunkingConfig:
    """Tests for SemanticChunking.get_config() - conditional on availability."""

    @pytest.mark.skipif(not SEMANTIC_CHUNKING_AVAILABLE, reason="Semantic chunking not installed")
    def test_get_config_without_threshold(self):
        """Config without threshold amount should not include that field."""
        from services.chunking import SemanticChunking

        strategy = SemanticChunking(
            embedder_model="all-MiniLM-L6-v2",
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=None,
        )
        config = strategy.get_config()

        assert config["chunking_method"] == "semantic"
        assert config["embedder_model"] == "all-MiniLM-L6-v2"
        assert config["breakpoint_threshold_type"] == "percentile"
        assert "breakpoint_threshold_amount" not in config

    @pytest.mark.skipif(not SEMANTIC_CHUNKING_AVAILABLE, reason="Semantic chunking not installed")
    def test_get_config_with_threshold(self):
        """Config with threshold amount should include that field."""
        from services.chunking import SemanticChunking

        strategy = SemanticChunking(
            embedder_model="all-MiniLM-L6-v2",
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.95,
        )
        config = strategy.get_config()

        assert config["breakpoint_threshold_amount"] == 0.95


class TestChunkingServiceFactory:
    """Tests for ChunkingService.get_chunking_strategy() factory method."""

    def test_returns_recursive_chunking(self):
        """'recursive' method should return RecursiveChunking instance."""
        strategy = ChunkingService.get_chunking_strategy("recursive")
        assert isinstance(strategy, RecursiveChunking)

    def test_returns_fixed_size_chunking(self):
        """'fixed' method should return FixedSizeChunking instance."""
        strategy = ChunkingService.get_chunking_strategy("fixed")
        assert isinstance(strategy, FixedSizeChunking)

    def test_applies_custom_chunk_size(self):
        """Custom chunk_size should be applied to strategy."""
        strategy = ChunkingService.get_chunking_strategy("recursive", chunk_size=500)
        config = strategy.get_config()
        assert config["chunk_size"] == 500

    def test_applies_custom_chunk_overlap(self):
        """Custom chunk_overlap should be applied to strategy."""
        strategy = ChunkingService.get_chunking_strategy("recursive", chunk_overlap=100)
        config = strategy.get_config()
        assert config["chunk_overlap"] == 100

    def test_applies_both_custom_params(self):
        """Both custom params should be applied."""
        strategy = ChunkingService.get_chunking_strategy(
            "fixed", chunk_size=800, chunk_overlap=150
        )
        config = strategy.get_config()
        assert config["chunk_size"] == 800
        assert config["chunk_overlap"] == 150

    def test_invalid_method_raises_error(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ChunkingService.get_chunking_strategy("unknown_method")

        assert "Unknown chunking method" in str(exc_info.value)

    def test_default_method_is_recursive(self):
        """Default method should be 'recursive'."""
        strategy = ChunkingService.get_chunking_strategy()
        assert isinstance(strategy, RecursiveChunking)

    @pytest.mark.skipif(not SEMANTIC_CHUNKING_AVAILABLE, reason="Semantic chunking not installed")
    def test_returns_semantic_chunking(self):
        """'semantic' method should return SemanticChunking instance."""
        from services.chunking import SemanticChunking

        strategy = ChunkingService.get_chunking_strategy("semantic")
        assert isinstance(strategy, SemanticChunking)


class TestGetAvailableMethods:
    """Tests for ChunkingService.get_available_methods()."""

    def test_includes_recursive(self):
        """Available methods should include 'recursive'."""
        methods = ChunkingService.get_available_methods()
        assert "recursive" in methods

    def test_includes_fixed(self):
        """Available methods should include 'fixed'."""
        methods = ChunkingService.get_available_methods()
        assert "fixed" in methods

    def test_method_has_required_fields(self):
        """Each method should have name, description, and default_params."""
        methods = ChunkingService.get_available_methods()

        for method_name, method_info in methods.items():
            assert "name" in method_info, f"{method_name} missing 'name'"
            assert "description" in method_info, f"{method_name} missing 'description'"
            assert "default_params" in method_info, f"{method_name} missing 'default_params'"

    def test_semantic_conditional_availability(self):
        """Semantic method should be present only if package is installed."""
        methods = ChunkingService.get_available_methods()

        if SEMANTIC_CHUNKING_AVAILABLE:
            assert "semantic" in methods
        else:
            assert "semantic" not in methods


class TestChunkDocuments:
    """Tests for ChunkingService.chunk_documents() convenience method."""

    def test_chunks_with_default_method(self):
        """chunk_documents should work with default method."""
        docs = [Document(page_content="This is a test document. " * 100)]
        chunks = ChunkingService.chunk_documents(docs)

        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunks_with_specified_method(self):
        """chunk_documents should work with specified method."""
        docs = [Document(page_content="This is a test document. " * 100)]
        chunks = ChunkingService.chunk_documents(docs, method="fixed")

        assert len(chunks) > 0

    def test_adds_chunking_metadata(self):
        """Chunked documents should have chunking metadata."""
        docs = [Document(page_content="This is a test document. " * 100)]
        chunks = ChunkingService.chunk_documents(docs, method="recursive")

        for chunk in chunks:
            assert "chunking_method" in chunk.metadata
            assert chunk.metadata["chunking_method"] == "recursive"

    def test_preserves_original_metadata(self):
        """Original document metadata should be preserved."""
        docs = [Document(
            page_content="This is a test document. " * 100,
            metadata={"source": "test.txt", "custom": "value"}
        )]
        chunks = ChunkingService.chunk_documents(docs)

        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.txt"
            assert chunk.metadata.get("custom") == "value"

    def test_respects_chunk_size(self):
        """Chunks should respect the specified chunk_size."""
        docs = [Document(page_content="Word " * 1000)]  # 5000 chars
        chunks = ChunkingService.chunk_documents(docs, chunk_size=500, chunk_overlap=50)

        # With 5000 chars and 500 chunk size, we should get multiple chunks
        assert len(chunks) > 1

        # Most chunks should be around the chunk_size (allowing some variation for word boundaries)
        for chunk in chunks[:-1]:  # Exclude last chunk which may be smaller
            assert len(chunk.page_content) <= 600  # Allow some buffer


class TestSplitDocumentsMetadata:
    """Tests for metadata enrichment during split_documents."""

    def test_recursive_adds_config_to_metadata(self):
        """RecursiveChunking should add config to chunk metadata."""
        strategy = RecursiveChunking(chunk_size=500, chunk_overlap=100)
        docs = [Document(page_content="Content " * 200)]

        chunks = strategy.split_documents(docs)

        for chunk in chunks:
            assert chunk.metadata["chunking_method"] == "recursive"
            assert chunk.metadata["chunk_size"] == 500
            assert chunk.metadata["chunk_overlap"] == 100

    def test_fixed_adds_config_to_metadata(self):
        """FixedSizeChunking should add config to chunk metadata."""
        strategy = FixedSizeChunking(chunk_size=500, chunk_overlap=100)
        docs = [Document(page_content="Content " * 200)]

        chunks = strategy.split_documents(docs)

        for chunk in chunks:
            assert chunk.metadata["chunking_method"] == "fixed"
            assert chunk.metadata["chunk_size"] == 500
            assert chunk.metadata["chunk_overlap"] == 100

    def test_handles_empty_metadata(self):
        """Documents with no metadata should get metadata dict created."""
        strategy = RecursiveChunking()
        # Document requires metadata to be a dict (not None), use empty dict
        docs = [Document(page_content="Content " * 200, metadata={})]

        # Should not raise
        chunks = strategy.split_documents(docs)

        for chunk in chunks:
            assert chunk.metadata is not None
            assert "chunking_method" in chunk.metadata
