"""Health check schemas for monitoring endpoints."""

from typing import Optional

from pydantic import BaseModel


class HealthStatusDetail(BaseModel):
    """Health status for an individual service."""

    status: str
    details: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall health response for the application."""

    status: str
    vllm: HealthStatusDetail
    postgres: HealthStatusDetail
