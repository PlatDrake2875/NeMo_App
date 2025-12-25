"""
Server-Sent Events (SSE) formatting utilities.
Centralizes SSE chunk creation to eliminate duplication across services.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional


# SSE Protocol Constants
SSE_DATA_PREFIX = "data: "
SSE_DONE_SIGNAL = "[DONE]"
SSE_NEWLINES = "\n\n"


@dataclass
class SSEChunk:
    """Represents a single SSE data chunk."""

    data: dict

    def to_sse(self) -> str:
        """Convert to SSE format string."""
        return f"{SSE_DATA_PREFIX}{json.dumps(self.data)}{SSE_NEWLINES}"


class SSEFormatter:
    """Utility class for formatting Server-Sent Events responses."""

    @staticmethod
    def format_token(token: str) -> str:
        """Format a token chunk for SSE streaming."""
        return SSEChunk({"token": token}).to_sse()

    @staticmethod
    def format_error(error_message: str) -> str:
        """Format an error message for SSE streaming."""
        return SSEChunk({"error": error_message}).to_sse()

    @staticmethod
    def format_done() -> str:
        """Format the completion signal for SSE streaming."""
        return SSEChunk({"status": "done"}).to_sse()

    @staticmethod
    def format_chat_completion_chunk(
        response_id: str,
        model: str,
        content: Optional[str] = None,
        finish_reason: Optional[str] = None,
        created: Optional[int] = None,
    ) -> str:
        """
        Format an OpenAI-style chat completion chunk for SSE streaming.

        Args:
            response_id: Unique identifier for the response
            model: Model name
            content: Token content (None for final chunk)
            finish_reason: Reason for completion (None if not finished)
            created: Unix timestamp (defaults to current time)
        """
        chunk_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created or int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }],
        }
        return SSEChunk(chunk_data).to_sse()

    @staticmethod
    def format_openai_done() -> str:
        """Format the OpenAI-style completion signal."""
        return f"{SSE_DATA_PREFIX}{SSE_DONE_SIGNAL}{SSE_NEWLINES}"

    @staticmethod
    def parse_sse_line(line: str) -> Optional[str]:
        """
        Parse an SSE data line and extract the content.

        Returns:
            The JSON content string, or None if not a data line
        """
        if line.startswith(SSE_DATA_PREFIX):
            content = line[len(SSE_DATA_PREFIX):].strip()
            if content and content != SSE_DONE_SIGNAL:
                return content
        return None

    @staticmethod
    def is_done_signal(line: str) -> bool:
        """Check if the line is the SSE done signal."""
        if line.startswith(SSE_DATA_PREFIX):
            content = line[len(SSE_DATA_PREFIX):].strip()
            return content == SSE_DONE_SIGNAL
        return False


def generate_response_id(prefix: str = "chatcmpl") -> str:
    """Generate a unique response ID for chat completions."""
    return f"{prefix}-{int(time.time() * 1000)}"
