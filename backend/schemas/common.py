"""Common/shared schemas used across the application."""

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Generic message model for conversation histories."""

    role: str = Field(
        ...,
        description="Role of the message sender (e.g., 'user', 'assistant', 'system')",
    )
    content: str = Field(..., description="Content of the message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"role": "user", "content": "Hello, how are you?"}
        }
    )


class UploadResponse(BaseModel):
    """Response model for file upload operations."""

    message: str
    filename: str
    chunks_added: int | None = None
