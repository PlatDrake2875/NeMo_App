"""Tests for Pydantic schema validators."""

import pytest
from pydantic import ValidationError

from schemas import ChunkingConfigSchema


class TestChunkingConfigSchema:
    """Tests for ChunkingConfigSchema validators."""

    def test_valid_config(self):
        """Test valid chunking configuration."""
        config = ChunkingConfigSchema(
            method="recursive",
            chunk_size=1000,
            chunk_overlap=200,
        )
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_overlap_must_be_less_than_size(self):
        """Test that chunk_overlap must be less than chunk_size."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfigSchema(
                method="recursive",
                chunk_size=500,
                chunk_overlap=500,  # Equal to size - should fail
            )
        assert "chunk_overlap must be less than chunk_size" in str(exc_info.value)

    def test_overlap_greater_than_size_fails(self):
        """Test that chunk_overlap greater than chunk_size fails."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfigSchema(
                method="recursive",
                chunk_size=500,
                chunk_overlap=600,  # Greater than size - should fail
            )
        assert "chunk_overlap must be less than chunk_size" in str(exc_info.value)

    def test_zero_overlap_is_valid(self):
        """Test that zero overlap is valid."""
        config = ChunkingConfigSchema(
            method="recursive",
            chunk_size=500,
            chunk_overlap=0,
        )
        assert config.chunk_overlap == 0

    def test_default_values(self):
        """Test default values are applied."""
        config = ChunkingConfigSchema()
        assert config.method == "recursive"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_valid_methods(self):
        """Test various valid chunking methods."""
        for method in ["character", "recursive", "token", "semantic"]:
            config = ChunkingConfigSchema(method=method)
            assert config.method == method
