"""Tests for message converter utilities."""

import pytest

from utils.message_converter import (
    ROLE_ASSISTANT,
    ROLE_BOT,
    ROLE_USER,
    MessageConverter,
)


class TestHistoryToLLMMessages:
    """Tests for history_to_llm_messages conversion."""

    def test_empty_history(self):
        """Test conversion of empty history."""
        result = MessageConverter.history_to_llm_messages([])

        assert result == []

    def test_single_user_message(self):
        """Test conversion of single user message."""
        history = [{"sender": "user", "text": "Hello"}]
        result = MessageConverter.history_to_llm_messages(history)

        assert result == [{"role": "user", "content": "Hello"}]

    def test_single_bot_message(self):
        """Test conversion of single bot message to assistant."""
        history = [{"sender": "bot", "text": "Hi there!"}]
        result = MessageConverter.history_to_llm_messages(history)

        assert result == [{"role": "assistant", "content": "Hi there!"}]

    def test_conversation_history(self):
        """Test conversion of multi-turn conversation."""
        history = [
            {"sender": "user", "text": "Hello"},
            {"sender": "bot", "text": "Hi! How can I help?"},
            {"sender": "user", "text": "Tell me a joke"},
            {"sender": "bot", "text": "Why did the chicken cross the road?"},
        ]
        result = MessageConverter.history_to_llm_messages(history)

        assert len(result) == 4
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi! How can I help?"}
        assert result[2] == {"role": "user", "content": "Tell me a joke"}
        assert result[3] == {"role": "assistant", "content": "Why did the chicken cross the road?"}

    def test_with_current_query(self):
        """Test adding current query to messages."""
        history = [{"sender": "user", "text": "Hello"}]
        result = MessageConverter.history_to_llm_messages(history, current_query="How are you?")

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "user", "content": "How are you?"}

    def test_current_query_only(self):
        """Test with only current query, no history."""
        result = MessageConverter.history_to_llm_messages([], current_query="Hello")

        assert result == [{"role": "user", "content": "Hello"}]

    def test_missing_sender_defaults_to_user(self):
        """Test that missing sender defaults to user role."""
        history = [{"text": "Hello"}]
        result = MessageConverter.history_to_llm_messages(history)

        assert result == [{"role": "user", "content": "Hello"}]

    def test_missing_text_uses_empty_string(self):
        """Test that missing text uses empty string."""
        history = [{"sender": "user"}]
        result = MessageConverter.history_to_llm_messages(history)

        assert result == [{"role": "user", "content": ""}]

    def test_case_insensitive_role(self):
        """Test that role comparison is case-insensitive."""
        history = [
            {"sender": "USER", "text": "Hello"},
            {"sender": "Bot", "text": "Hi"},
            {"sender": "BOT", "text": "How can I help?"},
        ]
        result = MessageConverter.history_to_llm_messages(history)

        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "assistant"


class TestExtractLastUserMessage:
    """Tests for extract_last_user_message."""

    def test_single_user_message(self):
        """Test extraction with single user message."""
        messages = [{"role": "user", "content": "Hello"}]
        result = MessageConverter.extract_last_user_message(messages)

        assert result == "Hello"

    def test_last_is_user(self):
        """Test extraction when last message is from user."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = MessageConverter.extract_last_user_message(messages)

        assert result == "How are you?"

    def test_last_is_assistant(self):
        """Test extraction when last message is from assistant."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = MessageConverter.extract_last_user_message(messages)

        assert result == "Hello"

    def test_no_user_message_raises(self):
        """Test that ValueError is raised when no user message exists."""
        messages = [{"role": "assistant", "content": "Hi!"}]

        with pytest.raises(ValueError, match="No user message found"):
            MessageConverter.extract_last_user_message(messages)

    def test_empty_messages_raises(self):
        """Test that ValueError is raised for empty messages list."""
        with pytest.raises(ValueError, match="No user message found"):
            MessageConverter.extract_last_user_message([])

    def test_empty_content_skipped(self):
        """Test that empty content messages are skipped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": ""},
        ]
        result = MessageConverter.extract_last_user_message(messages)

        assert result == "Hello"

    def test_missing_content_skipped(self):
        """Test that messages without content are skipped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user"},
        ]
        result = MessageConverter.extract_last_user_message(messages)

        assert result == "Hello"


class TestNormalizeRole:
    """Tests for normalize_role."""

    def test_user_unchanged(self):
        """Test that 'user' remains unchanged."""
        assert MessageConverter.normalize_role("user") == ROLE_USER

    def test_bot_to_assistant(self):
        """Test that 'bot' is normalized to 'assistant'."""
        assert MessageConverter.normalize_role("bot") == ROLE_ASSISTANT

    def test_assistant_unchanged(self):
        """Test that 'assistant' remains unchanged."""
        assert MessageConverter.normalize_role("assistant") == ROLE_ASSISTANT

    def test_case_insensitive(self):
        """Test case-insensitive normalization."""
        assert MessageConverter.normalize_role("USER") == ROLE_USER
        assert MessageConverter.normalize_role("Bot") == ROLE_ASSISTANT
        assert MessageConverter.normalize_role("ASSISTANT") == ROLE_ASSISTANT

    def test_unknown_role_defaults_to_user(self):
        """Test that unknown roles default to user."""
        assert MessageConverter.normalize_role("system") == ROLE_USER
        assert MessageConverter.normalize_role("unknown") == ROLE_USER
