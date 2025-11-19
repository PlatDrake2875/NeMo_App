# backend/routers/chat_router.py
"""
Chat router - thin web layer that delegates to ChatService.
Handles HTTP concerns only, business logic is in services.chat.ChatService.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from deps import get_chat_service
from schemas import ChatRequest
from services.chat import ChatService

# --- Router Setup ---
router = APIRouter(
    tags=["chat"],
)


# --- API Endpoints ---
@router.post("/chat")
async def chat_endpoint(
    chat_request: ChatRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Chat endpoint that streams responses from the configured LLM.

    Args:
        chat_request: The chat request containing query, model, history, etc.
        request: FastAPI Request object to detect client disconnection
        chat_service: Injected ChatService instance

    Returns:
        StreamingResponse with Server-Sent Events (SSE) format
    """
    # Convert Pydantic models to simple dicts for the service layer
    history_dicts = []
    if chat_request.history:
        history_dicts = [
            {"sender": msg.sender, "text": msg.text} for msg in chat_request.history
        ]

    # Delegate all business logic to the service
    generator = chat_service.process_chat_request(
        query=chat_request.query,
        model_name=chat_request.model,
        agent_name=chat_request.agent_name,
        history=history_dicts,
        use_rag=chat_request.use_rag,
        request=request,
    )

    return StreamingResponse(generator, media_type="text/event-stream")
