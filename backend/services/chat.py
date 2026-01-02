"""
Chat service layer containing all chat-related business logic.
Separated from the web layer for better testability and maintainability.

Refactored to use centralized utilities for SSE formatting and message conversion.
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Optional

import httpx

from config import (
    NEMO_GUARDRAILS_SERVER_URL,
    RAG_ENABLED,
    USE_GUARDRAILS,
    VLLM_BASE_URL,
    VLLM_MODEL,
)
from rag_components import get_rag_context_prefix
from services.nemo import get_local_nemo_instance
from utils.message_converter import MessageConverter
from utils.sse import SSEFormatter, generate_response_id

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service class handling all chat-related business logic.

    Responsibilities:
    - Process chat requests with optional RAG enhancement
    - Route requests to vLLM or NeMo Guardrails
    - Stream responses using Server-Sent Events (SSE)
    """

    def __init__(self):
        self.vllm_base_url = VLLM_BASE_URL
        self.default_model = VLLM_MODEL
        self.rag_enabled = RAG_ENABLED
        self.use_guardrails = USE_GUARDRAILS
        self.nemo_server_url = NEMO_GUARDRAILS_SERVER_URL

    async def process_chat_request(
        self,
        query: str,
        model_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        history: Optional[list[dict]] = None,
        use_rag: Optional[bool] = None,
        request=None,
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat request and stream the response.

        Args:
            query: The user's query
            model_name: Optional model name (defaults to configured model)
            agent_name: Optional NeMo agent name for guardrails
            history: Optional conversation history
            use_rag: Whether to use RAG (defaults to config setting)
            request: FastAPI request for disconnect detection

        Yields:
            SSE-formatted response chunks
        """
        history = history or []
        use_rag = use_rag if use_rag is not None else self.rag_enabled
        effective_model = model_name or self.default_model

        messages = await self._build_messages(history, query, use_rag)

        if self.use_guardrails:
            async for chunk in self._stream_via_guardrails(
                messages=messages,
                model_name=effective_model,
                agent_name=agent_name,
                request=request,
            ):
                yield chunk
        else:
            async for chunk in self._stream_via_vllm(
                model_name=effective_model,
                messages=messages,
                request=request,
            ):
                yield chunk

    async def _build_messages(
        self,
        history: list[dict],
        query: str,
        use_rag: bool,
    ) -> list[dict[str, str]]:
        """
        Build LLM messages from history and query with optional RAG enhancement.

        Args:
            history: Conversation history with 'sender' and 'text' keys
            query: Current user query
            use_rag: Whether to enhance query with RAG context

        Returns:
            List of messages with 'role' and 'content' keys
        """
        messages = MessageConverter.history_to_llm_messages(history)

        if self.rag_enabled and use_rag:
            enhanced_query = await get_rag_context_prefix(query)
            final_query = enhanced_query or query
        else:
            final_query = query

        messages.append({"role": "user", "content": final_query})
        return messages

    async def _stream_via_vllm(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        request=None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream responses directly from vLLM (OpenAI-compatible API).

        Args:
            model_name: The model to use
            messages: Messages to send to the LLM
            request: FastAPI request for disconnect detection

        Yields:
            SSE-formatted response chunks
        """
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.vllm_base_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        yield SSEFormatter.format_error(
                            f"vLLM API error {response.status_code}"
                        )
                        return

                    async for chunk in self._process_vllm_stream(response, request):
                        yield chunk

            except httpx.RequestError as e:
                logger.error("vLLM request error: %s", e)
                yield SSEFormatter.format_error(f"Connection error: {e}")

        yield SSEFormatter.format_done()

    async def _process_vllm_stream(
        self,
        response: httpx.Response,
        request=None,
    ) -> AsyncGenerator[str, None]:
        """
        Process the streaming response from vLLM.

        Args:
            response: The httpx streaming response
            request: FastAPI request for disconnect detection

        Yields:
            SSE-formatted token chunks
        """
        async for line in response.aiter_lines():
            if request and await request.is_disconnected():
                logger.info("Client disconnected, stopping vLLM stream")
                break

            content = SSEFormatter.parse_sse_line(line)
            if content is None:
                if SSEFormatter.is_done_signal(line):
                    break
                continue

            try:
                chunk_data = json.loads(content)
                token = self._extract_token_from_chunk(chunk_data)
                finish_reason = self._extract_finish_reason(chunk_data)

                if token:
                    yield SSEFormatter.format_token(token)
                if finish_reason:
                    break

            except json.JSONDecodeError:
                continue

    async def _stream_via_guardrails(
        self,
        messages: list[dict],
        model_name: str,
        agent_name: Optional[str] = None,
        request=None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion using local NeMo Guardrails integration.

        Args:
            messages: Messages to send to the guardrails
            model_name: The model to use
            agent_name: The NeMo agent configuration to use
            request: FastAPI request for disconnect detection

        Yields:
            SSE-formatted response chunks
        """
        if request and await request.is_disconnected():
            logger.info("Client disconnected before NeMo Guardrails call")
            return

        nemo_instance = await get_local_nemo_instance(agent_name)

        if not nemo_instance.is_available():
            yield SSEFormatter.format_error(
                "Local NeMo Guardrails instance not properly initialized"
            )
            return

        response = await nemo_instance.chat_completion(
            messages=messages,
            model=model_name,
            stream=False,
        )

        if response and response.get("choices"):
            content = response["choices"][0]["message"]["content"]
            response_id = response.get("id", generate_response_id("local"))
            created = response.get("created")

            # Stream content chunk
            yield SSEFormatter.format_chat_completion_chunk(
                response_id=response_id,
                model=model_name,
                content=content,
                created=created,
            )

            # Final chunk with finish_reason
            yield SSEFormatter.format_chat_completion_chunk(
                response_id=response_id,
                model=model_name,
                finish_reason="stop",
                created=created,
            )

            yield SSEFormatter.format_openai_done()
        else:
            yield SSEFormatter.format_error(
                "No valid response from local NeMo Guardrails"
            )

    @staticmethod
    def _extract_token_from_chunk(chunk_data: dict) -> str:
        """Extract token content from an OpenAI-style chunk."""
        choices = chunk_data.get("choices", [{}])
        if choices:
            return choices[0].get("delta", {}).get("content", "")
        return ""

    @staticmethod
    def _extract_finish_reason(chunk_data: dict) -> Optional[str]:
        """Extract finish_reason from an OpenAI-style chunk."""
        choices = chunk_data.get("choices", [{}])
        if choices:
            return choices[0].get("finish_reason")
        return None


async def main():
    """Test function for the ChatService."""
    chat_service = ChatService()
    print(f"Initialized: {chat_service.default_model}, RAG: {chat_service.rag_enabled}")

    history = [
        {"sender": "user", "text": "Hi"},
        {"sender": "bot", "text": "Hello!"},
    ]

    # Test message building
    messages = await chat_service._build_messages(history, "Test query", False)
    print(f"Messages: {len(messages)}")
    print("Content:", [f"{m['role']}: {m['content'][:20]}..." for m in messages])

    # Test streaming (requires vLLM to be running)
    chunks = 0
    async for chunk in chat_service.process_chat_request(
        query="Test",
        model_name=None,
        history=history,
        use_rag=False,
    ):
        chunks += 1
        if chunks <= 2:
            print(chunk[:50])

    print(f"Total chunks: {chunks}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
