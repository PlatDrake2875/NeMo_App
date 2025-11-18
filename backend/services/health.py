"""
Health service layer containing all health check business logic.
Separated from the web layer for better testability and maintainability.
"""

from typing import Optional

import psycopg
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from config import VLLM_BASE_URL
from schemas import HealthResponse, HealthStatusDetail


class HealthService:
    """Service class handling all health check business logic."""

    def __init__(self):
        self.vllm_base_url = VLLM_BASE_URL

    async def perform_health_check(
        self,
        pg_connection: Optional[str] = None,
        vllm_chat_for_rag: Optional[ChatOpenAI] = None,
    ) -> HealthResponse | JSONResponse:
        # Check vLLM status
        vllm_status = await self._check_vllm_status(vllm_chat_for_rag)

        # Check PostgreSQL status
        postgres_status = self._check_postgres_status(pg_connection)

        # Determine overall status
        overall_status = (
            "ok"
            if vllm_status.status == "connected"
            and postgres_status.status == "connected"
            else "error"
        )

        response_payload = HealthResponse(
            status=overall_status, vllm=vllm_status, postgres=postgres_status
        )

        if overall_status == "error":
            detail_payload = (
                response_payload.model_dump()
                if hasattr(response_payload, "model_dump")
                else response_payload.dict()
            )
            return JSONResponse(status_code=503, content=detail_payload)

        return response_payload

    async def _check_vllm_status(
        self, vllm_chat_for_rag: Optional[ChatOpenAI]
    ) -> HealthStatusDetail:
        """Check the status of the vLLM service."""
        if not vllm_chat_for_rag or not self.vllm_base_url:
            return HealthStatusDetail(
                status="disconnected",
                details="ChatOpenAI (for RAG) component failed to initialize or VLLM_BASE_URL not set.",
            )

        try:
            client = AsyncOpenAI(
                base_url=f"{self.vllm_base_url}/v1",
                api_key="EMPTY",
                timeout=5.0
            )
            await client.models.list()
            return HealthStatusDetail(
                status="connected", details="vLLM service is responsive."
            )
        except Exception as e:
            error_detail = f"vLLM connection or list models failed: {str(e)}"
            return HealthStatusDetail(status="disconnected", details=error_detail)

    def _check_postgres_status(
        self, pg_connection: Optional[str]
    ) -> HealthStatusDetail:
        """Check the status of the PostgreSQL service."""
        if not pg_connection:
            return HealthStatusDetail(
                status="disconnected",
                details="PostgreSQL connection string not initialized or connection failed.",
            )

        try:
            # Attempt to connect to PostgreSQL
            with psycopg.connect(pg_connection) as conn:
                with conn.cursor() as cur:
                    # Execute a simple query to verify the connection
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return HealthStatusDetail(
                status="connected", details="PostgreSQL service is responsive."
            )
        except Exception as e:
            error_detail = f"PostgreSQL connection error: {str(e)}"
            return HealthStatusDetail(status="disconnected", details=error_detail)
