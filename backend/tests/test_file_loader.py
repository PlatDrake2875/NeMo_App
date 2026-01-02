"""Tests for FileLoaderService - testing actual file parsing logic with real data."""

import pytest

from services.file_loader import FileLoaderService


class TestExtractTextFromDict:
    """Tests for _extract_text_from_dict method."""

    @pytest.fixture
    def loader(self):
        return FileLoaderService()

    def test_prefers_text_field(self, loader):
        """Dict with 'text' key should return that value."""
        data = {"text": "This is the text content", "other": "ignored"}
        result = loader._extract_text_from_dict(data)
        assert result == "This is the text content"

    def test_prefers_content_field(self, loader):
        """Dict with 'content' key (no 'text') should return content."""
        data = {"content": "This is the content", "title": "Title"}
        result = loader._extract_text_from_dict(data)
        assert result == "This is the content"

    def test_prefers_body_field(self, loader):
        """Dict with 'body' key should return body."""
        data = {"body": "This is the body text", "id": 123}
        result = loader._extract_text_from_dict(data)
        assert result == "This is the body text"

    def test_field_priority_order(self, loader):
        """'text' should take priority over 'content' and 'body'."""
        data = {"text": "text value", "content": "content value", "body": "body value"}
        result = loader._extract_text_from_dict(data)
        assert result == "text value"

    def test_concatenates_long_strings_as_fallback(self, loader):
        """When no standard fields, concatenate string values > 10 chars."""
        data = {
            "field1": "This is a long string value",
            "field2": "Another long string value",
            "short": "tiny",
        }
        result = loader._extract_text_from_dict(data)
        assert "This is a long string value" in result
        assert "Another long string value" in result
        assert "tiny" not in result  # Too short

    def test_returns_none_for_empty_dict(self, loader):
        """Empty dict should return None."""
        result = loader._extract_text_from_dict({})
        assert result is None

    def test_returns_none_for_short_values_only(self, loader):
        """Dict with only short values should return None."""
        data = {"a": "hi", "b": "bye", "c": 123}
        result = loader._extract_text_from_dict(data)
        assert result is None


class TestLoadJson:
    """Tests for _load_json method with real JSON data."""

    @pytest.fixture
    def loader(self):
        return FileLoaderService()

    def test_array_creates_multiple_docs(self, loader):
        """JSON array should create one document per object."""
        json_data = b'[{"text": "First doc"}, {"text": "Second doc"}]'
        docs = loader._load_json(json_data, {"source": "test.json"})

        assert len(docs) == 2
        assert docs[0].page_content == "First doc"
        assert docs[1].page_content == "Second doc"

    def test_object_creates_single_doc(self, loader):
        """JSON object should create one document."""
        json_data = b'{"content": "Single document content", "title": "Title"}'
        docs = loader._load_json(json_data, {"source": "test.json"})

        assert len(docs) == 1
        assert docs[0].page_content == "Single document content"

    def test_array_of_strings(self, loader):
        """Array of strings should create docs from each string."""
        json_data = b'["First string", "Second string"]'
        docs = loader._load_json(json_data, {"source": "test.json"})

        assert len(docs) == 2
        assert docs[0].page_content == "First string"
        assert docs[1].page_content == "Second string"

    def test_primitive_string(self, loader):
        """Primitive JSON string should create one document."""
        json_data = b'"Just a plain string"'
        docs = loader._load_json(json_data, {"source": "test.json"})

        assert len(docs) == 1
        assert docs[0].page_content == "Just a plain string"

    def test_preserves_metadata(self, loader):
        """Metadata should be passed through to documents."""
        json_data = b'{"text": "Content"}'
        metadata = {"source": "test.json", "custom_field": "custom_value"}
        docs = loader._load_json(json_data, metadata)

        assert docs[0].metadata["source"] == "test.json"
        assert docs[0].metadata["custom_field"] == "custom_value"

    def test_adds_json_index_for_array_items(self, loader):
        """Array items should have json_index in metadata."""
        json_data = b'[{"text": "First"}, {"text": "Second"}]'
        docs = loader._load_json(json_data, {"source": "test.json"})

        assert docs[0].metadata["json_index"] == 0
        assert docs[1].metadata["json_index"] == 1


class TestLoadCsv:
    """Tests for _load_csv method with real CSV data."""

    @pytest.fixture
    def loader(self):
        return FileLoaderService()

    def test_creates_doc_per_row(self, loader):
        """Each CSV row should create one document."""
        csv_data = b"id,name\n1,Alice\n2,Bob\n3,Charlie"
        docs = loader._load_csv(csv_data, {"source": "test.csv"})

        assert len(docs) == 3

    def test_detects_text_columns_by_length(self, loader):
        """Columns with avg length > 50 should be detected as text columns."""
        csv_data = b"""id,name,description
1,Item1,This is a very long description that definitely exceeds fifty characters in total length
2,Item2,Another extremely long description that also exceeds the fifty character threshold for detection"""

        docs = loader._load_csv(csv_data, {"source": "test.csv"})

        # Only description column should be in content (long enough)
        assert len(docs) == 2
        # The long description should be in the content
        assert "description" in docs[0].page_content.lower()

    def test_uses_all_columns_when_none_long(self, loader):
        """When no columns have avg length > 50, use all columns."""
        csv_data = b"id,code,value\n1,A,10\n2,B,20"
        docs = loader._load_csv(csv_data, {"source": "test.csv"})

        assert len(docs) == 2
        # All columns should be included
        assert "id" in docs[0].page_content
        assert "code" in docs[0].page_content
        assert "value" in docs[0].page_content

    def test_adds_csv_metadata(self, loader):
        """CSV documents should have csv_row and csv_columns in metadata."""
        csv_data = b"id,name\n1,Alice\n2,Bob"
        docs = loader._load_csv(csv_data, {"source": "test.csv"})

        assert docs[0].metadata["csv_row"] == 0
        assert docs[1].metadata["csv_row"] == 1
        assert "csv_columns" in docs[0].metadata


class TestLoadText:
    """Tests for _load_text method."""

    @pytest.fixture
    def loader(self):
        return FileLoaderService()

    def test_handles_utf8(self, loader):
        """UTF-8 encoded text should be loaded correctly."""
        text_data = "Hello, 世界! Привет мир!".encode("utf-8")
        docs = loader._load_text(text_data, {"source": "test.txt"})

        assert len(docs) == 1
        assert "世界" in docs[0].page_content
        assert "Привет" in docs[0].page_content

    def test_handles_latin1_fallback(self, loader):
        """Latin-1 encoded text should be handled via fallback."""
        # Create Latin-1 encoded text with special chars
        text_data = "Café résumé naïve".encode("latin-1")
        docs = loader._load_text(text_data, {"source": "test.txt"})

        assert len(docs) == 1
        assert "Caf" in docs[0].page_content  # Some chars may differ but should not crash

    def test_preserves_content(self, loader):
        """Text content should be preserved exactly."""
        text_data = b"Line 1\nLine 2\nLine 3"
        docs = loader._load_text(text_data, {"source": "test.txt"})

        assert docs[0].page_content == "Line 1\nLine 2\nLine 3"


class TestLoadMarkdown:
    """Tests for _load_markdown method."""

    @pytest.fixture
    def loader(self):
        return FileLoaderService()

    def test_preserves_markdown_content(self, loader):
        """Markdown content should be preserved as-is."""
        md_content = b"# Heading\n\nParagraph with **bold** and *italic*.\n\n- List item"
        docs = loader._load_markdown(md_content, {"source": "test.md"})

        assert len(docs) == 1
        assert "# Heading" in docs[0].page_content
        assert "**bold**" in docs[0].page_content

    def test_creates_single_document(self, loader):
        """Markdown should create exactly one document."""
        md_content = b"# Title\n\nContent here."
        docs = loader._load_markdown(md_content, {"source": "test.md"})

        assert len(docs) == 1


class TestDetectFileType:
    """Tests for detect_file_type class method."""

    def test_from_extension_pdf(self):
        assert FileLoaderService.detect_file_type("document.pdf") == "pdf"

    def test_from_extension_json(self):
        assert FileLoaderService.detect_file_type("data.json") == "json"

    def test_from_extension_markdown(self):
        assert FileLoaderService.detect_file_type("readme.md") == "md"

    def test_from_extension_markdown_full(self):
        assert FileLoaderService.detect_file_type("readme.markdown") == "md"

    def test_from_extension_txt(self):
        assert FileLoaderService.detect_file_type("notes.txt") == "txt"

    def test_from_extension_csv(self):
        assert FileLoaderService.detect_file_type("data.csv") == "csv"

    def test_from_mime_type(self):
        assert FileLoaderService.detect_file_type("file", "application/pdf") == "pdf"
        assert FileLoaderService.detect_file_type("file", "application/json") == "json"
        assert FileLoaderService.detect_file_type("file", "text/markdown") == "md"
        assert FileLoaderService.detect_file_type("file", "text/plain") == "txt"
        assert FileLoaderService.detect_file_type("file", "text/csv") == "csv"

    def test_mime_type_takes_priority(self):
        """MIME type should take priority over extension."""
        # File with .txt extension but JSON mime type
        result = FileLoaderService.detect_file_type("file.txt", "application/json")
        assert result == "json"

    def test_unknown_raises_error(self):
        """Unknown file type should raise ValueError."""
        with pytest.raises(ValueError):
            FileLoaderService.detect_file_type("file.xyz")

        with pytest.raises(ValueError):
            FileLoaderService.detect_file_type("file_no_extension")


class TestLoadFile:
    """Integration tests for load_file method."""

    @pytest.fixture
    def loader(self):
        return FileLoaderService()

    def test_loads_json_file(self, loader):
        """load_file should correctly route JSON files."""
        json_data = b'{"text": "Hello world"}'
        docs = loader.load_file(json_data, "test.json", "json")

        assert len(docs) == 1
        assert docs[0].page_content == "Hello world"

    def test_loads_txt_file(self, loader):
        """load_file should correctly route text files."""
        text_data = b"Plain text content"
        docs = loader.load_file(text_data, "test.txt", "txt")

        assert len(docs) == 1
        assert docs[0].page_content == "Plain text content"

    def test_unsupported_type_raises(self, loader):
        """Unsupported file type should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            loader.load_file(b"content", "test.xyz", "xyz")

        assert "Unsupported file type" in str(exc_info.value)

    def test_adds_base_metadata(self, loader):
        """load_file should add source and file_type to metadata."""
        docs = loader.load_file(b'{"text": "content"}', "myfile.json", "json")

        assert docs[0].metadata["source"] == "myfile.json"
        assert docs[0].metadata["file_type"] == "json"

    def test_merges_custom_metadata(self, loader):
        """Custom metadata should be merged with base metadata."""
        custom = {"custom_key": "custom_value"}
        docs = loader.load_file(b'{"text": "content"}', "test.json", "json", metadata=custom)

        assert docs[0].metadata["custom_key"] == "custom_value"
        assert docs[0].metadata["source"] == "test.json"
