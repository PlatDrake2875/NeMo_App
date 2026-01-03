"""Chat-related schemas for the chat API endpoints."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LegacyChatRequest(BaseModel):
    """Legacy chat request model (deprecated)."""

    query: str
    model: Optional[str] = None


class RAGChatRequest(BaseModel):
    """Chat request with RAG (Retrieval Augmented Generation) support."""

    session_id: str
    prompt: str
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Conversation history as a list of role/content dicts.",
    )
    use_rag: bool = True


class RAGChatResponse(BaseModel):
    """Response from a RAG-enabled chat request."""

    session_id: str
    response: str


class RAGStreamRequest(BaseModel):
    """Streaming chat request with RAG support."""

    session_id: str
    prompt: str
    history: list[dict[str, str]] = Field(
        default_factory=list, description="Conversation history for RAG stream."
    )
    use_rag: bool = True


class ChatResponseToken(BaseModel):
    """Token response for non-streaming chat (deprecated)."""

    token: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class HistoryMessage(BaseModel):
    """Message in chat history with sender and text fields."""

    sender: str = Field(..., description="Sender of the message ('user' or 'bot')")
    text: str = Field(..., description="Content of the message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"sender": "user", "text": "Hello, how are you?"}
        }
    )


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    query: str = Field(..., description="User's question or message")
    model: Optional[str] = Field(None, description="Model to use for the response")
    agent_name: Optional[str] = Field(
        None, description="Name of the NeMo Guardrails agent to use"
    )
    history: Optional[list[HistoryMessage]] = Field(
        default=[], description="Previous conversation history"
    )
    use_rag: Optional[bool] = Field(
        None, description="Whether to use RAG for this request"
    )
    collection_name: Optional[str] = Field(
        None, description="Name of the vector store collection to use for RAG"
    )
    use_colbert: Optional[bool] = Field(
        None, description="Whether to use ColBERT reranking for RAG"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is the weather like today?",
                "model": "gemma3:4b-it-q4_K_M",
                "agent_name": "math_assistant",
                "history": [
                    {"sender": "user", "text": "Hello"},
                    {"sender": "bot", "text": "Hi there! How can I help you?"},
                ],
                "use_rag": True,
                "collection_name": "rag_documents",
                "use_colbert": True,
            }
        }
    )


class ChatStreamChunk(BaseModel):
    """Individual chunk in a streaming chat response."""

    token: Optional[str] = Field(None, description="Text token being streamed")
    error: Optional[str] = Field(
        None, description="Error message if something went wrong"
    )
    status: Optional[str] = Field(None, description="Status information")

    model_config = ConfigDict(
        json_schema_extra={"example": {"token": "Hello, I'm doing well!"}}
    )
