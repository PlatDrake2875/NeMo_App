"""Tests for SSE formatting utilities."""

import json
import time

import pytest

from utils.sse import (
    SSE_DATA_PREFIX,
    SSE_DONE_SIGNAL,
    SSEChunk,
    SSEFormatter,
    generate_response_id,
)


class TestSSEChunk:
    """Tests for SSEChunk dataclass."""

    def test_to_sse_simple_dict(self):
        """Test SSE formatting with a simple dictionary."""
        chunk = SSEChunk({"token": "Hello"})
        result = chunk.to_sse()

        assert result.startswith(SSE_DATA_PREFIX)
        assert result.endswith("\n\n")
        assert json.loads(result[6:-2]) == {"token": "Hello"}

    def test_to_sse_complex_dict(self):
        """Test SSE formatting with a complex dictionary."""
        data = {
            "id": "test-123",
            "choices": [{"delta": {"content": "test"}}],
            "model": "llama",
        }
        chunk = SSEChunk(data)
        result = chunk.to_sse()

        parsed = json.loads(result[6:-2])
        assert parsed == data


class TestSSEFormatter:
    """Tests for SSEFormatter utility class."""

    def test_format_token(self):
        """Test token formatting."""
        result = SSEFormatter.format_token("Hello")

        assert SSE_DATA_PREFIX in result
        parsed = json.loads(result[6:-2])
        assert parsed == {"token": "Hello"}

    def test_format_token_with_special_chars(self):
        """Test token formatting with special characters."""
        result = SSEFormatter.format_token('Hello "world"')

        parsed = json.loads(result[6:-2])
        assert parsed["token"] == 'Hello "world"'

    def test_format_error(self):
        """Test error formatting."""
        result = SSEFormatter.format_error("Connection failed")

        parsed = json.loads(result[6:-2])
        assert parsed == {"error": "Connection failed"}

    def test_format_done(self):
        """Test done signal formatting."""
        result = SSEFormatter.format_done()

        parsed = json.loads(result[6:-2])
        assert parsed == {"status": "done"}

    def test_format_chat_completion_chunk_with_content(self):
        """Test chat completion chunk with content."""
        result = SSEFormatter.format_chat_completion_chunk(
            response_id="test-123",
            model="llama",
            content="Hello",
            created=1234567890,
        )

        parsed = json.loads(result[6:-2])
        assert parsed["id"] == "test-123"
        assert parsed["model"] == "llama"
        assert parsed["created"] == 1234567890
        assert parsed["object"] == "chat.completion.chunk"
        assert parsed["choices"][0]["delta"]["content"] == "Hello"
        assert parsed["choices"][0]["finish_reason"] is None

    def test_format_chat_completion_chunk_final(self):
        """Test final chat completion chunk with finish_reason."""
        result = SSEFormatter.format_chat_completion_chunk(
            response_id="test-123",
            model="llama",
            finish_reason="stop",
        )

        parsed = json.loads(result[6:-2])
        assert parsed["choices"][0]["delta"] == {}
        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_format_chat_completion_chunk_default_created(self):
        """Test that created defaults to current time."""
        before = int(time.time())
        result = SSEFormatter.format_chat_completion_chunk(
            response_id="test-123",
            model="llama",
            content="Hello",
        )
        after = int(time.time())

        parsed = json.loads(result[6:-2])
        assert before <= parsed["created"] <= after

    def test_format_openai_done(self):
        """Test OpenAI-style done signal."""
        result = SSEFormatter.format_openai_done()

        assert result == f"{SSE_DATA_PREFIX}{SSE_DONE_SIGNAL}\n\n"

    def test_parse_sse_line_valid(self):
        """Test parsing a valid SSE data line."""
        line = 'data: {"token": "Hello"}'
        result = SSEFormatter.parse_sse_line(line)

        assert result == '{"token": "Hello"}'

    def test_parse_sse_line_done_signal(self):
        """Test parsing the done signal returns None."""
        line = "data: [DONE]"
        result = SSEFormatter.parse_sse_line(line)

        assert result is None

    def test_parse_sse_line_empty_data(self):
        """Test parsing empty data returns None."""
        line = "data: "
        result = SSEFormatter.parse_sse_line(line)

        assert result is None

    def test_parse_sse_line_not_data_line(self):
        """Test parsing non-data line returns None."""
        line = "event: message"
        result = SSEFormatter.parse_sse_line(line)

        assert result is None

    def test_is_done_signal_true(self):
        """Test done signal detection."""
        assert SSEFormatter.is_done_signal("data: [DONE]") is True

    def test_is_done_signal_false(self):
        """Test non-done signal detection."""
        assert SSEFormatter.is_done_signal('data: {"token": "Hi"}') is False
        assert SSEFormatter.is_done_signal("event: message") is False


class TestGenerateResponseId:
    """Tests for response ID generation."""

    def test_default_prefix(self):
        """Test default prefix is used."""
        result = generate_response_id()

        assert result.startswith("chatcmpl-")

    def test_custom_prefix(self):
        """Test custom prefix is used."""
        result = generate_response_id("custom")

        assert result.startswith("custom-")

    def test_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = {generate_response_id() for _ in range(100)}

        # All IDs should be unique (may have some collision if very fast)
        assert len(ids) >= 99

    def test_contains_timestamp(self):
        """Test that ID contains a timestamp component."""
        before = int(time.time() * 1000)
        result = generate_response_id()
        after = int(time.time() * 1000)

        timestamp_str = result.split("-")[1]
        timestamp = int(timestamp_str)
        assert before <= timestamp <= after
