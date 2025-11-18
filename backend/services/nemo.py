"""
NeMo Guardrails service layer containing all guardrails business logic.
Separated from the web layer for better testability and maintainability.
"""

import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from nemoguardrails import LLMRails, RailsConfig

from config import VLLM_BASE_URL

logger = logging.getLogger("nemo_service")


class NemoService:
    """Service class handling all NeMo Guardrails business logic."""

    def __init__(self, agent_name: str = "math_assistant"):
        self.agent_name = agent_name
        self.config_path = self._get_default_config_path(agent_name)
        self.rails: Optional[LLMRails] = None
        self._is_initialized = False
        self.base_url = VLLM_BASE_URL

    def _get_default_config_path(self, agent_name: str) -> str:
        """Get the default config path relative to the backend directory."""
        current_dir = Path(__file__).parent.parent
        config_dir = current_dir / "guardrails_config" / f"{agent_name}"

        if not config_dir.exists():
            raise FileNotFoundError(
                f"Configuration directory not found: {config_dir}. "
                f"Available agents: {[d.name for d in (current_dir / 'guardrails_config').iterdir() if d.is_dir()]}"
            )

        return str(config_dir)

    async def initialize(self) -> bool:
        if self._is_initialized:
            return True

        logger.info(
            "Initializing NeMo Guardrails with config from %s", self.config_path
        )

        # Read the actual YAML config file
        config_file = Path(self.config_path) / "config.yml"
        colang_file = Path(self.config_path) / "config.co"

        yaml_content = config_file.read_text(encoding="utf-8")
        colang_content = colang_file.read_text(encoding="utf-8")

        # Replace hardcoded localhost URL with environment variable
        # This allows the config to work in both Docker and local development
        yaml_content = yaml_content.replace(
            "http://localhost:8000/v1", f"{self.base_url}/v1"
        )

        # Create rails configuration from YAML and Colang content
        rails_config = RailsConfig.from_content(
            colang_content=colang_content, yaml_content=yaml_content
        )

        logger.info("Rails config created programmatically")

        # Initialize the LLM Rails
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
        try:
            user_message = self._extract_user_message(messages)

            logger.info(
                "Processing message through guardrails: %s...", user_message[:100]
            )

            result = await self.rails.generate_async(user_message)

            # Format response to match OpenAI-style response
            response = {
                "id": f"chatcmpl-local-{hash(user_message)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result},
                        "finish_reason": "stop",
                    }
                ],
                "usage": self._calculate_usage(user_message, result),
            }

            logger.info("Generated response: %s...", result[:100] if result else "None")
            return response

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
        # Create response ID and timestamp upfront
        response_id = f"chatcmpl-local-{int(time.time())}"
        created_time = int(time.time())

        if not self._is_initialized:
            # Yield initialization error in SSE format
            error_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "error": {
                    "message": "NeMo Guardrails not initialized",
                    "type": "initialization_error",
                },
            }
            yield f"data: {self._json_dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            # Yield initial processing chunk
            processing_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {self._json_dumps(processing_chunk)}\n\n"

            # Extract user message
            user_message = self._extract_user_message(messages)

            logger.info(
                "Processing message through guardrails: %s...", user_message[:100]
            )

            # Generate response through NeMo Guardrails
            result = await self.rails.generate_async(user_message)

            # Split content into chunks for more realistic streaming
            if result:
                words = result.split()
                chunk_size = max(1, len(words) // 10)  # Split into ~10 chunks

                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i : i + chunk_size]
                    chunk_content = " ".join(chunk_words)

                    # Add space before chunk if not first chunk
                    if i > 0:
                        chunk_content = " " + chunk_content

                    chunk_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {self._json_dumps(chunk_data)}\n\n"

            # Final chunk to indicate completion
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": self._calculate_usage(user_message, result),
            }

            yield f"data: {self._json_dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

            logger.info("Generated response: %s...", result[:100] if result else "None")

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in streaming chat completion: %s", e)
            # Yield error in SSE format
            error_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "error": {"message": str(e), "type": "internal_error"},
            }
            yield f"data: {self._json_dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    def _extract_user_message(self, messages: list[dict[str, str]]) -> str:
        """Extract the user message from the messages list."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                if user_message:
                    return user_message

        raise HTTPException(status_code=400, detail="No user message found in messages")

    def _calculate_usage(self, user_message: str, result: Optional[str]) -> dict:
        """Calculate token usage for the response."""
        return {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(result.split()) if result else 0,
            "total_tokens": len(user_message.split())
            + (len(result.split()) if result else 0),
        }

    def _json_dumps(self, data: dict) -> str:
        """Helper method for JSON serialization."""
        import json

        return json.dumps(data)

    def is_available(self) -> bool:
        """Check if NeMo Guardrails is available and initialized."""
        return self._is_initialized and self.rails is not None

    async def health_check(self) -> dict:
        try:
            if not self.is_available():
                return {
                    "status": "unhealthy",
                    "message": "NeMo Guardrails not initialized",
                }

            # Try a simple test message
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.chat_completion(messages=test_messages)

            return {
                "status": "healthy",
                "message": "NeMo Guardrails is working correctly",
                "test_response_length": len(
                    response["choices"][0]["message"]["content"]
                ),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}",
            }


# Global instance for use throughout the application
_nemo_service_instance: Optional[NemoService] = None


async def get_nemo_service(agent_name: str = "math_assistant") -> NemoService:
    """
    Get or create the global NemoService instance.

    Args:
        agent_name: Name of the agent configuration to use (math_assistant, bank_assistant, aviation_assistant)

    Returns:
        NemoService: The initialized service instance
    """
    global _nemo_service_instance  # noqa: PLW0603

    if (
        _nemo_service_instance is None
        or _nemo_service_instance.agent_name != agent_name
    ):
        _nemo_service_instance = NemoService(agent_name)
        await _nemo_service_instance.initialize()

    return _nemo_service_instance


# Legacy function name for backward compatibility
async def get_local_nemo_instance(agent_name: str = "math_assistant") -> NemoService:
    """
    Legacy function name for backward compatibility.

    Args:
        agent_name: Name of the agent configuration to use

    Returns:
        NemoService: The initialized service instance
    """
    return await get_nemo_service(agent_name)


async def test_nemo_service(agent_name: str = "math_assistant"):
    """Test function for the NeMo Guardrails service."""
    logger.info("Testing NeMo Guardrails service with agent: %s", agent_name)

    try:
        nemo = await get_nemo_service(agent_name)

        if not nemo.is_available():
            logger.error("NeMo Guardrails service not available")
            return False

        # Test a simple message
        test_messages = [{"role": "user", "content": "Hello, this is a test message."}]

        response = await nemo.chat_completion(messages=test_messages)
        logger.info("Test response: %s", response)

        # Test health check
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
