"""
NeMo Guardrails service layer containing all guardrails business logic.
Separated from the web layer for better testability and maintainability.

Refactored to use centralized utilities for SSE formatting and message conversion.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from nemoguardrails import LLMRails, RailsConfig

from config import VLLM_BASE_URL
from utils.message_converter import MessageConverter
from utils.sse import SSEFormatter, generate_response_id

logger = logging.getLogger("nemo_service")


class NemoService:
    """
    Service class handling all NeMo Guardrails business logic.

    Responsibilities:
    - Initialize and manage NeMo Guardrails configuration
    - Process chat completions through guardrails
    - Stream responses with SSE formatting
    """

    def __init__(self, agent_name: str = "math_assistant"):
        self.agent_name = agent_name
        self.config_path = self._get_default_config_path(agent_name)
        self.rails: Optional[LLMRails] = None
        self._is_initialized = False
        self.base_url = VLLM_BASE_URL

    def _get_default_config_path(self, agent_name: str) -> str:
        """Get the default config path relative to the backend directory."""
        current_dir = Path(__file__).parent.parent
        config_dir = current_dir / "guardrails_config" / agent_name

        if not config_dir.exists():
            available_agents = self._list_available_agents(current_dir)
            raise FileNotFoundError(
                f"Configuration directory not found: {config_dir}. "
                f"Available agents: {available_agents}"
            )

        return str(config_dir)

    @staticmethod
    def _list_available_agents(backend_dir: Path) -> list[str]:
        """List available agent configurations."""
        config_root = backend_dir / "guardrails_config"
        if config_root.exists():
            return [d.name for d in config_root.iterdir() if d.is_dir()]
        return []

    async def initialize(self) -> bool:
        """Initialize the NeMo Guardrails instance."""
        if self._is_initialized:
            return True

        logger.info("Initializing NeMo Guardrails with config from %s", self.config_path)

        config_file = Path(self.config_path) / "config.yml"
        colang_file = Path(self.config_path) / "config.co"

        yaml_content = config_file.read_text(encoding="utf-8")
        colang_content = colang_file.read_text(encoding="utf-8")

        # Replace hardcoded localhost URL with environment variable
        yaml_content = yaml_content.replace(
            "http://localhost:8000/v1", f"{self.base_url}/v1"
        )

        rails_config = RailsConfig.from_content(
            colang_content=colang_content,
            yaml_content=yaml_content,
        )

        logger.info("Rails config created programmatically")

        self.rails = LLMRails(rails_config)
        logger.info("LLM Rails initialized successfully")

        self._is_initialized = True
        return True

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3",
        **kwargs,
    ) -> dict:
        """
        Process a chat completion through guardrails.

        Args:
            messages: List of messages with 'role' and 'content' keys
            model: Model name for the response
            **kwargs: Additional parameters (unused)

        Returns:
            OpenAI-style chat completion response
        """
        try:
            user_message = MessageConverter.extract_last_user_message(messages)
            logger.info("Processing message through guardrails: %s...", user_message[:100])

            result = await self.rails.generate_async(user_message)

            response = self._build_completion_response(
                user_message=user_message,
                result=result,
                model=model,
            )

            logger.info("Generated response: %s...", result[:100] if result else "None")
            return response

        except ValueError as e:
            logger.error("Invalid message format: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("Error in chat completion: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Chat completion failed: {e}"
            ) from e

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3",
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion through guardrails.

        Args:
            messages: List of messages with 'role' and 'content' keys
            model: Model name for the response
            **kwargs: Additional parameters (unused)

        Yields:
            SSE-formatted response chunks
        """
        response_id = generate_response_id("chatcmpl-local")
        created_time = int(time.time())

        if not self._is_initialized:
            yield self._format_error_chunk(
                response_id=response_id,
                model=model,
                created=created_time,
                message="NeMo Guardrails not initialized",
                error_type="initialization_error",
            )
            yield SSEFormatter.format_openai_done()
            return

        try:
            # Initial processing chunk
            yield SSEFormatter.format_chat_completion_chunk(
                response_id=response_id,
                model=model,
                content=None,
                created=created_time,
            )

            user_message = MessageConverter.extract_last_user_message(messages)
            logger.info("Processing message through guardrails: %s...", user_message[:100])

            result = await self.rails.generate_async(user_message)

            # Stream result in chunks for realistic streaming
            if result:
                for chunk_content in self._split_into_chunks(result):
                    yield SSEFormatter.format_chat_completion_chunk(
                        response_id=response_id,
                        model=model,
                        content=chunk_content,
                        created=created_time,
                    )

            # Final chunk with usage
            yield self._format_final_chunk(
                response_id=response_id,
                model=model,
                created=created_time,
                user_message=user_message,
                result=result,
            )

            yield SSEFormatter.format_openai_done()
            logger.info("Generated response: %s...", result[:100] if result else "None")

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in streaming chat completion: %s", e)
            yield self._format_error_chunk(
                response_id=response_id,
                model=model,
                created=created_time,
                message=str(e),
                error_type="internal_error",
            )
            yield SSEFormatter.format_openai_done()

    def _build_completion_response(
        self,
        user_message: str,
        result: str,
        model: str,
    ) -> dict:
        """Build an OpenAI-style chat completion response."""
        return {
            "id": f"chatcmpl-local-{hash(user_message)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }],
            "usage": self._calculate_usage(user_message, result),
        }

    @staticmethod
    def _calculate_usage(user_message: str, result: Optional[str]) -> dict:
        """Calculate approximate token usage for the response."""
        prompt_tokens = len(user_message.split())
        completion_tokens = len(result.split()) if result else 0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _split_into_chunks(text: str, num_chunks: int = 10) -> list[str]:
        """Split text into chunks for realistic streaming."""
        words = text.split()
        chunk_size = max(1, len(words) // num_chunks)
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            # Add leading space for non-first chunks
            if i > 0:
                chunk_content = " " + chunk_content
            chunks.append(chunk_content)

        return chunks

    def _format_error_chunk(
        self,
        response_id: str,
        model: str,
        created: int,
        message: str,
        error_type: str,
    ) -> str:
        """Format an error as an SSE chunk."""
        error_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "error": {"message": message, "type": error_type},
        }
        return f"data: {json.dumps(error_data)}\n\n"

    def _format_final_chunk(
        self,
        response_id: str,
        model: str,
        created: int,
        user_message: str,
        result: Optional[str],
    ) -> str:
        """Format the final chunk with finish_reason and usage."""
        final_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": self._calculate_usage(user_message, result),
        }
        return f"data: {json.dumps(final_data)}\n\n"

    def is_available(self) -> bool:
        """Check if NeMo Guardrails is available and initialized."""
        return self._is_initialized and self.rails is not None

    async def health_check(self) -> dict:
        """Perform a health check on the NeMo Guardrails instance."""
        try:
            if not self.is_available():
                return {
                    "status": "unhealthy",
                    "message": "NeMo Guardrails not initialized",
                }

            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.chat_completion(messages=test_messages)

            return {
                "status": "healthy",
                "message": "NeMo Guardrails is working correctly",
                "test_response_length": len(response["choices"][0]["message"]["content"]),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}",
            }


# Global instance management
_nemo_service_instance: Optional[NemoService] = None


async def get_nemo_service(agent_name: str = "math_assistant") -> NemoService:
    """
    Get or create the global NemoService instance.

    Args:
        agent_name: Name of the agent configuration to use

    Returns:
        Initialized NemoService instance
    """
    global _nemo_service_instance

    if (
        _nemo_service_instance is None
        or _nemo_service_instance.agent_name != agent_name
    ):
        _nemo_service_instance = NemoService(agent_name)
        await _nemo_service_instance.initialize()

    return _nemo_service_instance


# Backward compatibility alias
async def get_local_nemo_instance(agent_name: str = "math_assistant") -> NemoService:
    """Legacy function name for backward compatibility."""
    return await get_nemo_service(agent_name)


async def test_nemo_service(agent_name: str = "math_assistant"):
    """Test function for the NeMo Guardrails service."""
    logger.info("Testing NeMo Guardrails service with agent: %s", agent_name)

    try:
        nemo = await get_nemo_service(agent_name)

        if not nemo.is_available():
            logger.error("NeMo Guardrails service not available")
            return False

        test_messages = [{"role": "user", "content": "Hello, this is a test message."}]
        response = await nemo.chat_completion(messages=test_messages)
        logger.info("Test response: %s", response)

        health = await nemo.health_check()
        logger.info("Health check: %s", health)

        return True

    except Exception as e:
        logger.error("Test failed: %s", e)
        return False


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_nemo_service())
