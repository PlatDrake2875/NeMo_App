"""Tests for ChatService."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.chat import ChatService


class TestChatServiceInit:
    """Tests for ChatService initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default configuration."""
        with patch("services.chat.VLLM_BASE_URL", "http://localhost:8000"), \
             patch("services.chat.VLLM_MODEL", "test-model"), \
             patch("services.chat.RAG_ENABLED", True), \
             patch("services.chat.USE_GUARDRAILS", False):

            service = ChatService()

            assert service.vllm_base_url == "http://localhost:8000"
            assert service.default_model == "test-model"
            assert service.rag_enabled is True
            assert service.use_guardrails is False


class TestBuildMessages:
    """Tests for _build_messages method."""

    @pytest.fixture
    def service(self):
        """Create a ChatService instance for testing."""
        with patch("services.chat.VLLM_BASE_URL", "http://localhost:8000"), \
             patch("services.chat.VLLM_MODEL", "test-model"), \
             patch("services.chat.RAG_ENABLED", False), \
             patch("services.chat.USE_GUARDRAILS", False):
            return ChatService()

    @pytest.mark.asyncio
    async def test_build_messages_empty_history(self, service):
        """Test building messages with empty history."""
        messages = await service._build_messages([], "Hello", use_rag=False)

        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_build_messages_with_history(self, service):
        """Test building messages with conversation history."""
        history = [
            {"sender": "user", "text": "Hi"},
            {"sender": "bot", "text": "Hello!"},
        ]
        messages = await service._build_messages(history, "How are you?", use_rag=False)

        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hi"}
        assert messages[1] == {"role": "assistant", "content": "Hello!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    @pytest.mark.asyncio
    async def test_build_messages_with_rag_enabled(self, service):
        """Test building messages with RAG enhancement."""
        service.rag_enabled = True

        with patch("services.chat.get_rag_context_prefix") as mock_rag:
            mock_rag.return_value = "Enhanced query with context"

            messages = await service._build_messages([], "Hello", use_rag=True)

            assert messages[-1]["content"] == "Enhanced query with context"
            mock_rag.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_build_messages_rag_returns_none(self, service):
        """Test building messages when RAG returns no context."""
        service.rag_enabled = True

        with patch("services.chat.get_rag_context_prefix") as mock_rag:
            mock_rag.return_value = None

            messages = await service._build_messages([], "Hello", use_rag=True)

            assert messages[-1]["content"] == "Hello"


class TestExtractTokenFromChunk:
    """Tests for _extract_token_from_chunk static method."""

    def test_extract_token_valid_chunk(self):
        """Test extracting token from valid chunk."""
        chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        result = ChatService._extract_token_from_chunk(chunk)

        assert result == "Hello"

    def test_extract_token_empty_choices(self):
        """Test extracting token with empty choices."""
        chunk = {"choices": []}
        result = ChatService._extract_token_from_chunk(chunk)

        assert result == ""

    def test_extract_token_no_content(self):
        """Test extracting token with no content in delta."""
        chunk = {"choices": [{"delta": {}}]}
        result = ChatService._extract_token_from_chunk(chunk)

        assert result == ""

    def test_extract_token_missing_delta(self):
        """Test extracting token with missing delta."""
        chunk = {"choices": [{}]}
        result = ChatService._extract_token_from_chunk(chunk)

        assert result == ""


class TestExtractFinishReason:
    """Tests for _extract_finish_reason static method."""

    def test_extract_finish_reason_stop(self):
        """Test extracting 'stop' finish reason."""
        chunk = {"choices": [{"finish_reason": "stop"}]}
        result = ChatService._extract_finish_reason(chunk)

        assert result == "stop"

    def test_extract_finish_reason_none(self):
        """Test extracting None finish reason."""
        chunk = {"choices": [{"finish_reason": None}]}
        result = ChatService._extract_finish_reason(chunk)

        assert result is None

    def test_extract_finish_reason_empty_choices(self):
        """Test extracting finish reason with empty choices."""
        chunk = {"choices": []}
        result = ChatService._extract_finish_reason(chunk)

        assert result is None

    def test_extract_finish_reason_missing_key(self):
        """Test extracting finish reason with missing key."""
        chunk = {"choices": [{}]}
        result = ChatService._extract_finish_reason(chunk)

        assert result is None


class TestProcessChatRequest:
    """Tests for process_chat_request method."""

    @pytest.fixture
    def service(self):
        """Create a ChatService instance for testing."""
        with patch("services.chat.VLLM_BASE_URL", "http://localhost:8000"), \
             patch("services.chat.VLLM_MODEL", "test-model"), \
             patch("services.chat.RAG_ENABLED", False), \
             patch("services.chat.USE_GUARDRAILS", False):
            return ChatService()

    @pytest.mark.asyncio
    async def test_process_chat_request_uses_default_model(self, service):
        """Test that default model is used when not specified."""
        with patch.object(service, "_stream_via_vllm") as mock_stream:
            mock_stream.return_value = AsyncMock()
            mock_stream.return_value.__aiter__ = AsyncMock(return_value=iter([]))

            chunks = []
            async for chunk in service.process_chat_request(query="Hello"):
                chunks.append(chunk)

            # Verify default model was used
            call_args = mock_stream.call_args
            assert call_args.kwargs["model_name"] == "test-model"

    @pytest.mark.asyncio
    async def test_process_chat_request_uses_specified_model(self, service):
        """Test that specified model overrides default."""
        with patch.object(service, "_stream_via_vllm") as mock_stream:
            mock_stream.return_value = AsyncMock()
            mock_stream.return_value.__aiter__ = AsyncMock(return_value=iter([]))

            chunks = []
            async for chunk in service.process_chat_request(
                query="Hello",
                model_name="custom-model"
            ):
                chunks.append(chunk)

            call_args = mock_stream.call_args
            assert call_args.kwargs["model_name"] == "custom-model"

    @pytest.mark.asyncio
    async def test_process_chat_request_routes_to_guardrails(self, service):
        """Test that requests are routed to guardrails when enabled."""
        service.use_guardrails = True

        with patch.object(service, "_stream_via_guardrails") as mock_stream:
            mock_stream.return_value = AsyncMock()
            mock_stream.return_value.__aiter__ = AsyncMock(return_value=iter([]))

            chunks = []
            async for chunk in service.process_chat_request(query="Hello"):
                chunks.append(chunk)

            mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_chat_request_routes_to_vllm(self, service):
        """Test that requests are routed to vLLM when guardrails disabled."""
        service.use_guardrails = False

        with patch.object(service, "_stream_via_vllm") as mock_stream:
            mock_stream.return_value = AsyncMock()
            mock_stream.return_value.__aiter__ = AsyncMock(return_value=iter([]))

            chunks = []
            async for chunk in service.process_chat_request(query="Hello"):
                chunks.append(chunk)

            mock_stream.assert_called_once()
