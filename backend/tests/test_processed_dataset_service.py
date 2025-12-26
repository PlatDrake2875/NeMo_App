"""Tests for ProcessedDatasetService, especially the get_chunks endpoint."""

from unittest.mock import MagicMock, patch

import pytest

from services.processed_dataset import ProcessedDatasetService


class TestProcessedDatasetService:
    """Tests for ProcessedDatasetService."""

    @pytest.fixture
    def service(self):
        """Create a ProcessedDatasetService instance for testing."""
        return ProcessedDatasetService()

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()


class TestGetChunks:
    """Tests for get_chunks method."""

    @pytest.fixture
    def service(self):
        """Create a ProcessedDatasetService instance for testing."""
        return ProcessedDatasetService()

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_get_chunks_returns_correct_format(self, service, mock_db):
        """Test that get_chunks returns correct response format."""
        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_dataset.vector_backend = "pgvector"
        mock_dataset.collection_name = "test_collection"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        with patch.object(service, "_get_chunks_pgvector") as mock_pgvector:
            mock_pgvector.return_value = {
                "chunks": [
                    {"id": "1", "content": "test content", "metadata": {}},
                ],
                "total": 1,
            }

            result = service.get_chunks(mock_db, 1, page=1, limit=20)

            assert "chunks" in result
            assert "total" in result
            assert isinstance(result["chunks"], list)
            assert isinstance(result["total"], int)

    def test_get_chunks_raises_for_missing_dataset(self, service, mock_db):
        """Test that get_chunks raises ValueError for missing dataset."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError) as exc_info:
            service.get_chunks(mock_db, 999, page=1, limit=20)

        assert "not found" in str(exc_info.value)

    def test_get_chunks_pagination(self, service, mock_db):
        """Test that get_chunks respects pagination parameters."""
        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_dataset.vector_backend = "pgvector"
        mock_dataset.collection_name = "test_collection"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        with patch.object(service, "_get_chunks_pgvector") as mock_pgvector:
            mock_pgvector.return_value = {"chunks": [], "total": 100}

            service.get_chunks(mock_db, 1, page=2, limit=10)

            # Verify the call was made with correct offset
            call_args = mock_pgvector.call_args
            assert call_args[1]["page"] == 2
            assert call_args[1]["limit"] == 10

    def test_get_chunks_with_search(self, service, mock_db):
        """Test that get_chunks passes search parameter."""
        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_dataset.vector_backend = "pgvector"
        mock_dataset.collection_name = "test_collection"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        with patch.object(service, "_get_chunks_pgvector") as mock_pgvector:
            mock_pgvector.return_value = {"chunks": [], "total": 0}

            service.get_chunks(mock_db, 1, page=1, limit=20, search="test query")

            call_args = mock_pgvector.call_args
            assert call_args[1]["search"] == "test query"

    def test_get_chunks_qdrant_backend(self, service, mock_db):
        """Test that get_chunks uses qdrant backend correctly."""
        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_dataset.vector_backend = "qdrant"
        mock_dataset.collection_name = "test_collection"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        with patch.object(service, "_get_chunks_qdrant") as mock_qdrant:
            mock_qdrant.return_value = {"chunks": [], "total": 0}

            service.get_chunks(mock_db, 1, page=1, limit=20)

            mock_qdrant.assert_called_once()

    def test_get_chunks_unsupported_backend(self, service, mock_db):
        """Test that get_chunks raises for unsupported backend."""
        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_dataset.vector_backend = "unsupported"
        mock_dataset.collection_name = "test_collection"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        with pytest.raises(ValueError) as exc_info:
            service.get_chunks(mock_db, 1, page=1, limit=20)

        assert "Unsupported" in str(exc_info.value)


class TestCreateDataset:
    """Tests for create_dataset method."""

    @pytest.fixture
    def service(self):
        """Create a ProcessedDatasetService instance for testing."""
        return ProcessedDatasetService()

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_create_dataset_generates_collection_name(self, service, mock_db):
        """Test that create_dataset generates a unique collection name."""
        from schemas import ProcessedDatasetCreate

        mock_raw_dataset = MagicMock()
        mock_raw_dataset.id = 1
        mock_raw_dataset.name = "test_raw"

        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_raw_dataset,  # Raw dataset lookup
            None,  # Collection name uniqueness check
        ]

        request = ProcessedDatasetCreate(
            name="test_processed",
            description="Test",
            raw_dataset_id=1,
        )

        with patch.object(service, "_generate_collection_name") as mock_gen:
            mock_gen.return_value = "test_collection_123"

            service.create_dataset(mock_db, request)

            mock_gen.assert_called_once()
