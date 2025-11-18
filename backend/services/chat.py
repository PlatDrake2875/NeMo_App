"""
Chat service layer containing all chat-related business logic.
Separated from the web layer for better testability and maintainability.
"""

import json
import time
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


class ChatService:
    """Service class handling all chat-related business logic."""

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
    ) -> AsyncGenerator[str, None]:
        if history is None:
            history = []
        if use_rag is None:
            use_rag = self.rag_enabled

        effective_model_name = model_name or self.default_model
        request_id = f"req_{int(time.time() * 1000)}"
        stream_id = f"stream_{request_id}"

        # Build messages for LLM
        messages_for_llm = await self._build_messages_with_rag(history, query, use_rag)

        # Route to appropriate streaming method
        if self.use_guardrails:
            async for chunk in self._local_nemo_guardrails_stream(
                messages_for_llm=messages_for_llm,
                stream_id=stream_id,
                model_name=effective_model_name,
                agent_name=agent_name,
            ):
                yield chunk
        else:
            async for chunk in self._direct_vllm_stream(
                model_name=effective_model_name,
                messages_payload=messages_for_llm,
            ):
                yield chunk

    def _build_messages_for_llm(
        self, history: list[dict], query: str, use_rag: bool
    ) -> list[dict[str, str]]:
        """Build the messages list for the LLM based on history and current query."""
        messages_for_llm: list[dict[str, str]] = []

        # Add conversation history
        for msg in history:
            role = msg.get("sender", "user").lower()
            content = msg.get("text", "")
            if role == "bot":
                messages_for_llm.append({"role": "assistant", "content": content})
            else:
                messages_for_llm.append({"role": "user", "content": content})

        # Add current query with optional RAG enhancement
        current_user_query_content = query

        if self.rag_enabled and use_rag:
            # This will be async, so we need to handle it in the calling method
            # For now, let's keep the sync interface and handle RAG in the caller
            pass

        messages_for_llm.append({"role": "user", "content": current_user_query_content})
        return messages_for_llm

    async def _build_messages_with_rag(
        self, history: list[dict], query: str, use_rag: bool
    ) -> list[dict[str, str]]:
        """Build messages with RAG enhancement - async version."""
        messages_for_llm: list[dict[str, str]] = []

        # Add conversation history
        for msg in history:
            role = msg.get("sender", "user").lower()
            content = msg.get("text", "")
            if role == "bot":
                messages_for_llm.append({"role": "assistant", "content": content})
            else:
                messages_for_llm.append({"role": "user", "content": content})

        # Add current query with optional RAG enhancement
        if self.rag_enabled and use_rag:
            rag_enhanced_prompt = await get_rag_context_prefix(query)
            if rag_enhanced_prompt:
                messages_for_llm.append(
                    {"role": "user", "content": rag_enhanced_prompt}
                )
            else:
                messages_for_llm.append({"role": "user", "content": query})
        else:
            messages_for_llm.append({"role": "user", "content": query})

        return messages_for_llm

    async def _direct_vllm_stream(
        self, model_name: str, messages_payload: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream responses directly from vLLM (OpenAI-compatible API)."""
        request_payload = {
            "model": model_name,
            "messages": messages_payload,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.vllm_base_url}/v1/chat/completions",
                json=request_payload,
            ) as response:
                if response.status_code != 200:
                    error_msg = f"vLLM API error {response.status_code}"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        line_content = line[len("data: "):].strip()
                        if line_content == "[DONE]":
                            break
                        if not line_content:
                            continue

                        try:
                            chunk_data = json.loads(line_content)
                            # Extract token from OpenAI-style response
                            token = (
                                chunk_data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            finish_reason = chunk_data.get("choices", [{}])[0].get(
                                "finish_reason"
                            )

                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if finish_reason:
                                break
                        except json.JSONDecodeError:
                            continue

        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    async def _local_nemo_guardrails_stream(
        self,
        messages_for_llm: list[dict],
        stream_id: str,
        model_name: str,
        agent_name: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion using local NeMo Guardrails integration."""
        nemo_instance = await get_local_nemo_instance(agent_name)

        if not nemo_instance.is_available():
            error_msg = "Local NeMo Guardrails instance not properly initialized"
            yield f'data: {{"error": "{error_msg}"}}\n\n'
            return

        response = await nemo_instance.chat_completion(
            messages=messages_for_llm, model=model_name, stream=False
        )

        if response and "choices" in response and response["choices"]:
            content = response["choices"][0]["message"]["content"]

            # Stream the response in SSE format
            chunk_data = {
                "id": response.get("id", f"local-{int(time.time())}"),
                "object": "chat.completion.chunk",
                "created": response.get("created", int(time.time())),
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

            # Final chunk
            final_chunk = {
                "id": response.get("id", f"local-{int(time.time())}"),
                "object": "chat.completion.chunk",
                "created": response.get("created", int(time.time())),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        else:
            error_msg = "No valid response from local NeMo Guardrails"
            yield f'data: {{"error": "{error_msg}"}}\n\n'

    async def _guardrails_stream(
        self,
        model_name_for_guardrails: str,
        messages_payload: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Stream responses through NeMo Guardrails server."""
        guardrails_endpoint = f"{self.nemo_server_url}/v1/chat/completions"

        guardrails_payload = {
            "model": model_name_for_guardrails,
            "messages": messages_payload,
            "config_id": "mybot",
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", guardrails_endpoint, json=guardrails_payload
            ) as response:
                if response.status_code != 200:
                    error_content_bytes = await response.aread()
                    raw_response_content_for_debug = error_content_bytes.decode(
                        errors="replace"
                    )
                    err_msg = f"NeMo Guardrails API error {response.status_code}: {raw_response_content_for_debug}"
                    yield f"data: {json.dumps({'error': err_msg})}\n\n"
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        line_content = line[len("data: ") :].strip()
                        if line_content == "[DONE]":
                            break
                        if not line_content:
                            continue

                        chunk_data = json.loads(line_content)

                        if "object" in chunk_data and chunk_data["object"] == "error":
                            err_msg = chunk_data.get(
                                "message", "Unknown error from Guardrails stream"
                            )
                            yield f"data: {json.dumps({'error': err_msg})}\n\n"
                            continue

                        if "detail" in chunk_data:
                            err_detail = chunk_data["detail"]
                            if (
                                isinstance(err_detail, list)
                                and err_detail
                                and "msg" in err_detail[0]
                            ):
                                err_msg = f"Error from Guardrails stream: {err_detail[0]['msg']}"
                            elif isinstance(err_detail, str):
                                err_msg = f"Error from Guardrails stream: {err_detail}"
                            else:
                                err_msg = f"Unknown error structure from Guardrails stream: {err_detail}"
                            yield f"data: {json.dumps({'error': err_msg})}\n\n"
                            continue

                        token = (
                            chunk_data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        finish_reason = chunk_data.get("choices", [{}])[0].get(
                            "finish_reason"
                        )

                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        if finish_reason:
                            break

        yield f"data: {json.dumps({'status': 'done'})}\n\n"


async def main():
    chat_service = ChatService()
    print(f"Initialized: {chat_service.default_model}, RAG: {chat_service.rag_enabled}")

    history = [{"sender": "user", "text": "Hi"}, {"sender": "bot", "text": "Hello!"}]

    # Test message building
    sync_messages = chat_service._build_messages_for_llm(history, "Test query", False)
    async_messages = await chat_service._build_messages_with_rag(
        history, "Test query", False
    )
    print(f"Sync messages: {len(sync_messages)}, Async messages: {len(async_messages)}")
    print("Messages:", [f"{m['role']}: {m['content'][:20]}..." for m in sync_messages])

    chunks = 0
    async for chunk in chat_service.process_chat_request(
        query="Test", model_name=None, history=history, use_rag=False
    ):
        chunks += 1
        if chunks <= 2:
            print(chunk[:50])

    print(f"Total chunks: {chunks}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
