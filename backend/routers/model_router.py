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


# =============================================================================
# Model Switch Endpoints - Dynamic model switching via Docker control
# =============================================================================

from pydantic import BaseModel, Field
from typing import Optional, List
from services.model_switch import get_model_switch_service


class ModelSwitchRequest(BaseModel):
    """Request to switch the active vLLM model."""
    model_id: str = Field(..., description="HuggingFace model ID to switch to")


class ModelSwitchResponse(BaseModel):
    """Response with model switch task status."""
    id: str
    from_model: Optional[str]
    to_model: str
    status: str
    progress: int
    current_step: Optional[str]
    estimated_time: Optional[int]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]


class CachedModelInfo(BaseModel):
    """Information about a cached model."""
    model_id: str
    size_gb: float
    cached_at: Optional[str]
    is_loaded: bool = False


@router.post("/switch", response_model=ModelSwitchResponse)
async def switch_model(
    request: ModelSwitchRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Switch the active vLLM model.

    This initiates a background task to:
    1. Check if model is cached
    2. Stop current vLLM container
    3. Restart vLLM with the new model
    4. Wait for the model to be ready

    Returns task ID for tracking progress.

    Raises:
        HTTPException 409: If a switch is already in progress
        HTTPException 400: If model is not cached
        HTTPException 503: If evaluations are running
    """
    switch_service = get_model_switch_service()

    # Get current model for display
    try:
        models = await model_service.list_available_models()
        current_model = models[0].name if models else None
    except Exception:
        current_model = None

    try:
        task_id = await switch_service.start_switch(
            target_model=request.model_id,
            current_model=current_model
        )
        task = switch_service.get_task(task_id)
        return ModelSwitchResponse(**task)

    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start switch: {str(e)}")


@router.get("/switch/status", response_model=Optional[ModelSwitchResponse])
async def get_switch_status():
    """
    Get the status of the current model switch (if any).

    Returns None if no switch is in progress.
    """
    switch_service = get_model_switch_service()
    task = switch_service.get_current_task()
    if task:
        return ModelSwitchResponse(**task)
    return None


@router.get("/switch/{task_id}", response_model=ModelSwitchResponse)
async def get_switch_task(task_id: str):
    """Get the status of a specific model switch task."""
    switch_service = get_model_switch_service()
    task = switch_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return ModelSwitchResponse(**task)


@router.delete("/switch/{task_id}")
async def cancel_switch(task_id: str):
    """Cancel an in-progress model switch."""
    switch_service = get_model_switch_service()
    success = await switch_service.cancel_switch(task_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Task not found or not running"
        )
    return {"status": "cancelled", "task_id": task_id}


@router.get("/cached", response_model=List[CachedModelInfo])
async def list_cached_models(
    model_service: ModelService = Depends(get_model_service),
):
    """
    List models that are cached in the HuggingFace cache.

    These models can be loaded quickly without download.
    """
    switch_service = get_model_switch_service()
    cached = switch_service.list_cached_models()

    # Get currently loaded model
    try:
        loaded_models = await model_service.list_available_models()
        loaded_ids = {m.name for m in loaded_models}
    except Exception:
        loaded_ids = set()

    # Add is_loaded flag
    result = []
    for model in cached:
        result.append(CachedModelInfo(
            model_id=model["model_id"],
            size_gb=model["size_gb"],
            cached_at=model.get("cached_at"),
            is_loaded=model["model_id"] in loaded_ids,
        ))

    return result


@router.get("/switch/history", response_model=List[ModelSwitchResponse])
async def get_switch_history(limit: int = 10):
    """Get recent model switch tasks."""
    switch_service = get_model_switch_service()
    tasks = switch_service.list_recent_tasks(limit=limit)
    return [ModelSwitchResponse(**task) for task in tasks]
