# backend/routers/health_router.py
"""
Health router - thin web layer that delegates to HealthService.
Handles HTTP concerns only, business logic is in services.health.HealthService.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI

from deps import get_health_service
from rag_components import (
    get_optional_pg_connection,
    get_optional_vllm_chat_for_rag,
)
from schemas import HealthResponse
from services.health import HealthService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check_endpoint(
    health_service: HealthService = Depends(get_health_service),
    pg_connection: Optional[str] = Depends(get_optional_pg_connection),
    vllm_chat_for_rag: Optional[ChatOpenAI] = Depends(
        get_optional_vllm_chat_for_rag
    ),
):
    # Service returns either HealthResponse or JSONResponse with 503 status
    return await health_service.perform_health_check(
        pg_connection, vllm_chat_for_rag
    )


@router.get("/live")
async def liveness_check():
    """Simple liveness check for container orchestration - returns 200 if server is up"""
    return {"status": "ok"}
