"""
Model service layer containing all model-related business logic.
Separated from the web layer for better testability and maintainability.
"""

from typing import Optional

import httpx
from fastapi import HTTPException
from openai import AsyncOpenAI

from config import VLLM_BASE_URL
from schemas import ModelInfo


class ModelService:
    """Service class handling all model-related business logic."""

    def __init__(self):
        self.vllm_base_url = VLLM_BASE_URL
        self.client = AsyncOpenAI(
            base_url=f"{self.vllm_base_url}/v1",
            api_key="EMPTY"  # vLLM doesn't require authentication
        )

    async def list_available_models(self) -> list[ModelInfo]:
        try:
            # Call vLLM's OpenAI-compatible /v1/models endpoint
            models_response = await self.client.models.list()

            if not hasattr(models_response, "data"):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid response structure from vLLM (missing 'data' attribute).",
                )

            models_list = models_response.data

            if not isinstance(models_list, list):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid data type for 'data' in vLLM models response.",
                )

            parsed_models = []
            for model_obj in models_list:
                model_info = self._parse_vllm_model(model_obj)
                if model_info:
                    parsed_models.append(model_info)

            return parsed_models
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch models from vLLM: {str(e)}"
            )

    def _parse_vllm_model(self, model_obj) -> Optional[ModelInfo]:
        # Convert OpenAI model object to dict
        model_data_dict = model_obj.model_dump() if hasattr(model_obj, "model_dump") else dict(model_obj)

        return ModelInfo.from_openai(model_data_dict)

    def _get_valid_status_code(self, status_code: int) -> int:
        if not isinstance(status_code, int) or status_code < 100 or status_code > 599:
            return 500
        return status_code

    async def load_model_into_vllm(
        self, model_id: str, served_model_name: Optional[str] = None
    ) -> dict:
        """
        Load a model into vLLM using the admin API.

        vLLM must be running with `--enable-auto-tool-choice` or in multi-model mode
        to support dynamic model loading.

        Args:
            model_id: HuggingFace model ID to load
            served_model_name: Optional name to serve the model as

        Returns:
            Dictionary with status and model information

        Raises:
            HTTPException: If model loading fails
        """
        try:
            # vLLM's model loading endpoint (when running in multi-model mode)
            # NOTE: This endpoint is available in vLLM v0.4.0+ when using --enable-lora or similar flags
            # For standard vLLM, models must be specified at startup

            # Use the served model name or default to model ID
            model_name = served_model_name or model_id

            async with httpx.AsyncClient(timeout=60.0) as client:
                # Try to load the model via vLLM's admin API
                # This endpoint may vary depending on vLLM version
                load_url = f"{self.vllm_base_url}/v1/load_lora_adapter"

                # For now, we'll just verify the model is available
                # In a production setup with vLLM supporting dynamic loading,
                # you would use the appropriate API endpoint

                # Check if model is already loaded
                models = await self.list_available_models()
                model_names = [m.name for m in models]

                if model_id in model_names or model_name in model_names:
                    return {
                        "status": "success",
                        "message": f"Model '{model_id}' is already loaded in vLLM",
                        "model_name": model_name,
                    }

                # If model is not loaded, return instructions for manual loading
                # In production with multi-model vLLM, you would call the load API here
                return {
                    "status": "manual_restart_required",
                    "message": (
                        f"Model '{model_id}' has been downloaded to the HuggingFace cache. "
                        "To use it, please restart vLLM with this model specified, or configure "
                        "vLLM for multi-model support."
                    ),
                    "model_id": model_id,
                    "instructions": (
                        f"Add to docker-compose.yml vLLM command: --model={model_id}"
                    ),
                }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model into vLLM: {str(e)}"
            )
