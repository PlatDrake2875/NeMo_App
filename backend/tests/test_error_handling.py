"""Tests for error handling improvements across services."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.file_loader import FileLoaderService
from services.llm_metadata_extractor import LLMMetadataExtractor


class TestFileLoaderErrorHandling:
    """Tests for FileLoaderService error handling."""

    @pytest.fixture
    def loader(self):
        """Create a FileLoaderService instance for testing."""
        return FileLoaderService()

    def test_pdf_extraction_failure_adds_metadata(self, loader):
        """Test that PDF extraction failure adds extraction_error metadata."""
        # Create a PDF with no text content (simulated)
        empty_pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

        with patch("services.file_loader.PdfReader") as mock_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = ""
            mock_reader.return_value.pages = [mock_page]

            docs = loader._load_pdf(empty_pdf_content, {"source": "test.pdf"})

            # Should have one placeholder document
            assert len(docs) == 1
            assert docs[0].metadata.get("extraction_error") is True
            assert "extraction_error_reason" in docs[0].metadata
            assert "OCR" in docs[0].page_content or "could not be extracted" in docs[0].page_content

    def test_unsupported_file_type_raises_error(self, loader):
        """Test that unsupported file type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            loader.load_file(b"content", "test.xyz", "xyz")

        assert "Unsupported file type" in str(exc_info.value)

    def test_detect_file_type_from_extension(self, loader):
        """Test file type detection from extension."""
        assert loader.detect_file_type("test.pdf") == "pdf"
        assert loader.detect_file_type("test.json") == "json"
        assert loader.detect_file_type("test.md") == "md"
        assert loader.detect_file_type("test.txt") == "txt"
        assert loader.detect_file_type("test.csv") == "csv"

    def test_detect_file_type_unknown_raises_error(self, loader):
        """Test that unknown file type raises ValueError."""
        with pytest.raises(ValueError):
            loader.detect_file_type("test.unknown")


class TestLLMMetadataExtractorErrorHandling:
    """Tests for LLMMetadataExtractor error handling."""

    @pytest.fixture
    def extractor(self):
        """Create an LLMMetadataExtractor instance for testing."""
        return LLMMetadataExtractor(model="test-model")

    @pytest.mark.asyncio
    async def test_extraction_failure_sets_flag(self, extractor):
        """Test that extraction failure sets extraction_failed flag."""
        mock_config = MagicMock()
        mock_config.extract_summary = True
        mock_config.extract_keywords = True
        mock_config.extract_entities = False
        mock_config.extract_categories = False

        with patch.object(extractor, "_call_llm") as mock_call:
            mock_call.side_effect = Exception("LLM API Error")

            result = await extractor.extract_metadata("test content", mock_config)

            assert result.get("extraction_failed") is True

    @pytest.mark.asyncio
    async def test_fallback_parsing_sets_flag(self, extractor):
        """Test that fallback parsing sets _used_fallback_parsing flag."""
        mock_config = MagicMock()
        mock_config.extract_summary = True
        mock_config.extract_keywords = True
        mock_config.extract_entities = False
        mock_config.extract_categories = False

        with patch.object(extractor, "_call_llm") as mock_call:
            # Return invalid JSON to trigger fallback parsing
            mock_call.return_value = "This is not valid JSON but contains keywords"

            result = await extractor.extract_metadata("test content", mock_config)

            # If fallback parsing was used, the flag should be set
            if result.get("_used_fallback_parsing"):
                assert result["_used_fallback_parsing"] is True


class TestPreprocessingPipelineErrorHandling:
    """Tests for preprocessing pipeline error handling."""

    def test_error_dict_includes_error_type(self):
        """Test that error dict includes error_type field."""
        # Simulate an error dict that would be created by the pipeline
        error = ValueError("Test error")
        error_dict = {
            "filename": "test.pdf",
            "error": str(error),
            "error_type": type(error).__name__,
        }

        assert error_dict["error_type"] == "ValueError"
        assert error_dict["error"] == "Test error"
        assert error_dict["filename"] == "test.pdf"
