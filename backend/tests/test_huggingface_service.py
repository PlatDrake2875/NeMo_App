"""Tests for HuggingFace dataset service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.huggingface_dataset import HuggingFaceDatasetService


class TestHuggingFaceDatasetService:
    """Tests for HuggingFaceDatasetService."""

    @pytest.fixture
    def service(self):
        """Create a HuggingFaceDatasetService instance for testing."""
        return HuggingFaceDatasetService(token=None)

    @pytest.mark.asyncio
    async def test_get_dataset_metadata_returns_columns_format(self, service):
        """Test that get_dataset_metadata returns columns in correct format."""
        mock_info = MagicMock()
        mock_info.description = "Test dataset"
        mock_info.size_in_bytes = 1000
        mock_info.splits = {"train": MagicMock(num_examples=100)}
        mock_info.features = {
            "text": MagicMock(),
            "label": MagicMock(),
        }
        # Set the type names
        type(mock_info.features["text"]).__name__ = "Value"
        type(mock_info.features["label"]).__name__ = "ClassLabel"

        mock_builder = MagicMock()
        mock_builder.info = mock_info

        with patch("services.huggingface_dataset.load_dataset_builder", return_value=mock_builder):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_builder)

                metadata = await service.get_dataset_metadata("test/dataset")

                # Verify columns is a list of HFColumnInfo objects
                assert hasattr(metadata, "columns")
                assert isinstance(metadata.columns, list)
                assert len(metadata.columns) == 2

                # Verify column format
                column_names = [col.name for col in metadata.columns]
                assert "text" in column_names
                assert "label" in column_names

    @pytest.mark.asyncio
    async def test_search_datasets_returns_search_succeeded_flag(self, service):
        """Test that search_datasets returns search_succeeded flag."""
        mock_dataset = MagicMock()
        mock_dataset.id = "test/dataset"
        mock_dataset.author = "test"
        mock_dataset.downloads = 100
        mock_dataset.likes = 10
        mock_dataset.tags = ["nlp"]

        with patch("services.huggingface_dataset.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.list_datasets.return_value = [mock_dataset]
            mock_api_class.return_value = mock_api

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=[mock_dataset])

                result = await service.search_datasets("test")

                assert "search_succeeded" in result
                assert result["search_succeeded"] is True
                assert "results" in result
                assert "error" in result
                assert result["error"] is None

    @pytest.mark.asyncio
    async def test_search_datasets_error_returns_search_succeeded_false(self, service):
        """Test that search error returns search_succeeded=False."""
        with patch("services.huggingface_dataset.HfApi") as mock_api_class:
            mock_api_class.side_effect = Exception("API Error")

            result = await service.search_datasets("test")

            assert result["search_succeeded"] is False
            assert result["error"] is not None
            assert "API Error" in result["error"]
            assert result["results"] == []

    @pytest.mark.asyncio
    async def test_validate_dataset_returns_tuple(self, service):
        """Test that validate_dataset returns (is_valid, error_message) tuple."""
        with patch.object(service, "get_dataset_metadata") as mock_get:
            mock_get.return_value = MagicMock()

            is_valid, error = await service.validate_dataset("test/dataset")

            assert is_valid is True
            assert error is None

    @pytest.mark.asyncio
    async def test_validate_dataset_invalid_returns_error(self, service):
        """Test that validate_dataset returns error for invalid dataset."""
        with patch.object(service, "get_dataset_metadata") as mock_get:
            mock_get.side_effect = ValueError("Dataset not found")

            is_valid, error = await service.validate_dataset("invalid/dataset")

            assert is_valid is False
            assert error is not None
            assert "Dataset not found" in error
