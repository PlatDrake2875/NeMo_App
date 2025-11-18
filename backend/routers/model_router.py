# backend/routers/model_router.py
"""
Model router - thin web layer that delegates to ModelService.
Handles HTTP concerns only, business logic is in services.model.ModelService.
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from deps import get_model_service
from schemas import (
    ModelDownloadRequest,
    ModelInfo,
    ModelLoadRequest,
    ModelMetadataResponse,
    ModelValidationRequest,
)
from services.huggingface import get_hf_service
from services.model import ModelService

router = APIRouter(
    prefix="/models",  # Prefix relative to the /api added in main.py
    tags=["models"],
)


@router.get(
    "", response_model=list[ModelInfo]
)  # Path is relative to prefix -> /api/models
async def list_vllm_models_endpoint(
    model_service: ModelService = Depends(get_model_service),
):
    """
    List available vLLM models via OpenAI-compatible API.

    Args:
        model_service: Injected ModelService instance

    Returns:
        List of available models
    """
    # Delegate all business logic to the service
    return await model_service.list_available_models()


@router.post("/validate", response_model=ModelMetadataResponse)
async def validate_model_endpoint(request: ModelValidationRequest):
    """
    Validate a HuggingFace model and get its metadata.

    Args:
        request: Model validation request with model ID and optional token

    Returns:
        Model metadata including size, gated status, etc.

    Raises:
        HTTPException: If model validation fails
    """
    hf_service = get_hf_service(token=request.token)

    try:
        metadata = await hf_service.get_model_metadata(request.model_id)
        return ModelMetadataResponse(**metadata.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/download")
async def download_model_endpoint(request: ModelDownloadRequest):
    """
    Download a model from HuggingFace with progress updates.

    This endpoint streams progress updates as Server-Sent Events (SSE).

    Args:
        request: Model download request with model ID and optional token

    Returns:
        StreamingResponse with progress updates
    """
    hf_service = get_hf_service(token=request.token)

    async def event_generator():
        """Generate SSE events for download progress."""
        try:
            async for progress_update in hf_service.download_model_with_progress(
                model_id=request.model_id, token=request.token
            ):
                # Format as SSE event
                yield f"data: {json.dumps(progress_update)}\n\n"
        except Exception as e:
            # Send error event
            error_event = {
                "stage": "error",
                "progress": 0,
                "message": "Download failed",
                "error": str(e),
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/load")
async def load_model_endpoint(
    request: ModelLoadRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Load a downloaded model into vLLM.

    Args:
        request: Model load request with model ID
        model_service: Injected ModelService instance

    Returns:
        Success message with loaded model info

    Raises:
        HTTPException: If model loading fails
    """
    try:
        result = await model_service.load_model_into_vllm(
            model_id=request.model_id,
            served_model_name=request.served_model_name,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
