# backend/routers/chat_router.py
"""
Chat router - thin web layer that delegates to ChatService.
Handles HTTP concerns only, business logic is in services.chat.ChatService.
"""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from deps import get_chat_service
from schemas import ChatRequest
from services.chat import ChatService

# Directory for storing chat attachments
ATTACHMENTS_DIR = Path("data/attachments")
ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file types and max size
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf", ".txt", ".md", ".csv", ".json"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

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
        collection_name=chat_request.collection_name,
        use_colbert=chat_request.use_colbert,
        request=request,
    )

    return StreamingResponse(generator, media_type="text/event-stream")


@router.post("/chat/attachments")
async def upload_attachment(file: UploadFile = File(...)):
    """
    Upload a file attachment for use in chat messages.

    Args:
        file: The uploaded file

    Returns:
        Dict with file info including the URL to access it
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024)}MB"
        )

    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    file_path = ATTACHMENTS_DIR / safe_filename

    # Save file
    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "id": unique_id,
        "filename": file.filename,
        "stored_filename": safe_filename,
        "size": len(content),
        "content_type": file.content_type,
        "url": f"/api/chat/attachments/{safe_filename}"
    }


@router.get("/chat/attachments/{filename}")
async def get_attachment(filename: str):
    """
    Retrieve a previously uploaded attachment.

    Args:
        filename: The stored filename

    Returns:
        The file content
    """
    from fastapi.responses import FileResponse

    file_path = ATTACHMENTS_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Attachment not found")

    # Validate path traversal
    if not file_path.resolve().is_relative_to(ATTACHMENTS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    return FileResponse(file_path, filename=filename)
