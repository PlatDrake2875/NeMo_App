# Backend Development Guide

Complete guide to the FastAPI backend architecture and implementation.

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Layered Architecture](#layered-architecture)
5. [Routers](#routers)
6. [Services](#services)
7. [Schemas](#schemas)
8. [Configuration](#configuration)
9. [Dependencies](#dependencies)
10. [Best Practices](#best-practices)

## Overview

The backend is a FastAPI application following a layered architecture pattern with clear separation of concerns.

### Key Principles

- **Layered Architecture**: Routers → Services → Components
- **Dependency Injection**: FastAPI dependencies for testability
- **Type Safety**: Pydantic models for validation
- **Async-First**: Async/await throughout
- **Single Responsibility**: Each module has one clear purpose

## Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | Latest | Web framework |
| Uvicorn | Latest | ASGI server |
| Pydantic | 2.x | Data validation |
| NeMo Guardrails | 0.13.0 | LLM guardrails |
| LangChain | Latest | LLM orchestration |
| ChromaDB | Latest | Vector database |
| Ollama | Latest (client) | LLM provider |
| PyPDF | Latest | PDF processing |
| uv | Latest | Package manager |

**Why These Choices?**
- **FastAPI**: Auto-documentation, async support, validation
- **Pydantic**: Type safety, runtime validation
- **uv**: Faster than pip, better dependency resolution
- **Async**: Efficient for I/O-bound operations (LLM calls, DB queries)

## Project Structure

```
backend/
├── main.py                      # FastAPI app, startup
├── config.py                    # Configuration management
├── schemas.py                   # Pydantic models
├── deps.py                      # Dependency injection
├── rag_components.py            # RAG singleton manager
│
├── routers/                     # HTTP endpoint handlers
│   ├── __init__.py
│   ├── chat_router.py          # Chat streaming
│   ├── agents_router.py        # Agent management
│   ├── model_router.py         # Model listing
│   ├── upload_router.py        # File uploads
│   ├── document_router.py      # Document queries
│   ├── automate_router.py      # Automation tasks
│   └── health_router.py        # Health checks
│
├── services/                    # Business logic layer
│   ├── __init__.py
│   ├── chat.py                 # Chat processing
│   ├── nemo.py                 # NeMo Guardrails
│   ├── model.py                # Model management
│   ├── upload.py               # File processing
│   ├── document.py             # Document retrieval
│   ├── automate.py             # Automation logic
│   └── health.py               # Health checks
│
├── guardrails_config/           # NeMo agent configs
│   ├── metadata.yaml           # Agent metadata
│   ├── aviation_assistant/
│   │   ├── config.yml
│   │   └── config.co
│   └── ...
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── conftest.py             # Test fixtures
│   ├── test_chat.py
│   └── test_agents.py
│
├── pyproject.toml               # Project metadata
├── uv.lock                      # Dependency lock
├── Dockerfile                   # Container image
└── .python-version              # Python version
```

## Layered Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────┐
│              HTTP Layer                     │
│  Routers: Receive requests, return responses│
│  - Request validation (Pydantic)            │
│  - Response serialization                   │
│  - HTTP status codes                        │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│             Service Layer                   │
│  Services: Business logic, orchestration    │
│  - Chat processing                          │
│  - RAG retrieval                            │
│  - NeMo Guardrails integration              │
│  - File processing                          │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│           Component Layer                   │
│  Components: Shared resources, singletons   │
│  - RAG Components (singleton)               │
│  - Database connections                     │
│  - External service clients                 │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│         External Services                   │
│  - ChromaDB (vector store)                  │
│  - Ollama (LLM provider)                    │
│  - NeMo Guardrails (local library)          │
└─────────────────────────────────────────────┘
```

### Layer Responsibilities

**1. Routers (HTTP Layer)**
- Thin layer, minimal logic
- Request validation
- Response formatting
- Delegate to services

**2. Services (Business Logic)**
- Core application logic
- Orchestrate components
- Error handling
- Business rules

**3. Components (Data Access)**
- Singleton resources
- Database clients
- External API clients
- Shared utilities

## Routers

### Router Pattern

**File**: `backend/routers/chat_router.py`

```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas import ChatRequest, ChatResponse
from services.chat import chat_service

# Create router with prefix and tags
router = APIRouter(
    prefix="/api",
    tags=["chat"]
)

@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> StreamingResponse:
    """
    Stream chat responses with optional guardrails and RAG

    Args:
        request: ChatRequest with query, model, agent, etc.

    Returns:
        StreamingResponse with SSE events

    Raises:
        HTTPException: If validation fails or service error
    """
    try:
        # Validate request (automatic via Pydantic)
        # Delegate to service
        stream = chat_service.process_chat(request)

        # Return streaming response
        return StreamingResponse(
            stream,
            media_type="text/event-stream"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Router Best Practices

✅ **Do**: Keep routers thin
```python
@router.post("/chat")
async def chat(request: ChatRequest):
    # Good: Delegate to service
    return await chat_service.process(request)
```

❌ **Don't**: Put business logic in routers
```python
@router.post("/chat")
async def chat(request: ChatRequest):
    # Bad: Business logic in router
    if request.use_rag:
        docs = vector_store.search(...)
        context = format_docs(docs)
        # ... more logic
```

### All Routers Reference

| Router | File | Endpoints | Purpose |
|--------|------|-----------|---------|
| Chat | `chat_router.py:15-68` | POST `/api/chat` | Streaming chat |
| Agents | `agents_router.py:10-85` | GET `/api/agents/*` | Agent management |
| Models | `model_router.py:8-45` | GET `/api/models` | Model listing |
| Upload | `upload_router.py:12-68` | POST `/api/upload` | File uploads |
| Documents | `document_router.py:9-52` | GET `/api/documents` | Document queries |
| Automate | `automate_router.py:15-78` | POST `/api/automate` | Automation |
| Health | `health_router.py:7-35` | GET `/health` | Health checks |

## Services

### Service Pattern

**File**: `backend/services/chat.py`

```python
from config import Config
from rag_components import RAGComponents
from services.nemo import nemo_service
import json

class ChatService:
    """
    Handle chat message processing with optional RAG and guardrails
    """

    def __init__(self):
        self.config = Config
        self.rag = RAGComponents()

    async def process_chat(self, request: ChatRequest):
        """
        Process chat request and stream response

        Args:
            request: ChatRequest with query, model, options

        Yields:
            SSE formatted events with tokens
        """

        # Step 1: RAG retrieval (if enabled)
        context = None
        if request.use_rag:
            context = await self._retrieve_context(request.query)

        # Step 2: Build prompt
        prompt = self._build_prompt(request.query, context)

        # Step 3: Route to appropriate LLM service
        if self.config.USE_GUARDRAILS and request.agent_name:
            # Use NeMo Guardrails
            response_stream = nemo_service.generate_stream(
                agent_name=request.agent_name,
                message=prompt,
                history=request.history
            )
        else:
            # Direct Ollama
            response_stream = await self._call_ollama(
                model=request.model,
                prompt=prompt
            )

        # Step 4: Stream response as SSE
        try:
            async for token in response_stream:
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'status': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant documents from RAG"""
        docs = self.rag.retriever.get_relevant_documents(query)

        context_parts = [
            f"[{doc.metadata['source']}]\n{doc.page_content}"
            for doc in docs
        ]

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str = None) -> str:
        """Build final prompt with optional context"""
        if context:
            return self.config.RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=query
            )
        else:
            return query

    async def _call_ollama(self, model: str, prompt: str):
        """Call Ollama API directly"""
        # Implementation...
        pass

# Singleton instance
chat_service = ChatService()
```

### Service Best Practices

✅ **Do**: Single responsibility per service
```python
class ChatService:
    """Handles chat processing only"""
    pass

class UploadService:
    """Handles file uploads only"""
    pass
```

❌ **Don't**: God objects
```python
class ApplicationService:
    """Handles everything - BAD"""
    def chat(self): pass
    def upload(self): pass
    def process_docs(self): pass
    # ... 50 more methods
```

✅ **Do**: Dependency injection
```python
class ChatService:
    def __init__(self, rag: RAGComponents, nemo: NeMoService):
        self.rag = rag
        self.nemo = nemo
```

✅ **Do**: Error handling
```python
async def process(self, request):
    try:
        result = await self._do_work(request)
        return result
    except ValueError as e:
        # Business logic error
        raise
    except Exception as e:
        # Unexpected error, log and re-raise
        logger.error(f"Unexpected error: {e}")
        raise
```

### All Services Reference

| Service | File | Responsibilities |
|---------|------|------------------|
| Chat | `chat.py:20-180` | Message processing, RAG, streaming |
| NeMo | `nemo.py:15-145` | Guardrails initialization, generation |
| Model | `model.py:10-65` | Ollama model queries |
| Upload | `upload.py:18-95` | PDF processing, embedding, storage |
| Document | `document.py:12-78` | ChromaDB document queries |
| Health | `health.py:8-55` | Service connectivity checks |
| Automate | `automate.py:15-92` | Conversation summarization, actions |

## Schemas

### Pydantic Models

**File**: `backend/schemas.py`

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""

    query: str = Field(..., min_length=1, description="User's message")
    model: str = Field(
        default="gemma3:4b-it-q4_K_M",
        description="Ollama model name"
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Guardrails agent to use"
    )
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation history"
    )
    use_rag: bool = Field(
        default=False,
        description="Enable RAG retrieval"
    )

    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

    @validator('history')
    def validate_history(cls, v):
        for msg in v:
            if 'sender' not in msg or 'text' not in msg:
                raise ValueError('History messages must have sender and text')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is V1 speed?",
                "model": "gemma3:latest",
                "agent_name": "aviation_assistant",
                "use_rag": True,
                "history": [
                    {"sender": "user", "text": "Hello"},
                    {"sender": "bot", "text": "Hi! How can I help?"}
                ]
            }
        }

class OllamaModelInfo(BaseModel):
    """Ollama model information"""

    name: str
    modified_at: str
    size: int

class UploadResponse(BaseModel):
    """Response from file upload"""

    message: str
    filename: str
    chunks_added: int

class HealthResponse(BaseModel):
    """Health check response"""

    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    services: Dict[str, Any]
```

### Schema Best Practices

✅ **Do**: Use validators
```python
@validator('email')
def validate_email(cls, v):
    if '@' not in v:
        raise ValueError('Invalid email')
    return v.lower()
```

✅ **Do**: Provide examples
```python
class Config:
    json_schema_extra = {
        "example": {...}
    }
```

✅ **Do**: Use Field for metadata
```python
field: str = Field(..., description="...", min_length=1)
```

❌ **Don't**: Skip validation
```python
# Bad: No validation
data: dict  # Any dict, no structure
```

## Configuration

### Environment-Based Config

**File**: `backend/config.py`

```python
import os
from typing import Optional

class Config:
    """Application configuration from environment variables"""

    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )
    OLLAMA_MODEL_FOR_RAG: str = os.getenv(
        "OLLAMA_MODEL_FOR_RAG",
        "gemma3:latest"
    )

    # ChromaDB settings
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))

    # Feature flags
    USE_GUARDRAILS: bool = os.getenv(
        "USE_GUARDRAILS",
        "true"
    ).lower() == "true"

    RAG_ENABLED: bool = os.getenv(
        "RAG_ENABLED",
        "true"
    ).lower() == "true"

    # RAG configuration
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
    RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Prompts
    RAG_PROMPT_TEMPLATE: str = """
    You are a helpful assistant. Answer based on the provided context.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    SIMPLE_PROMPT_TEMPLATE: str = "{query}"

    # Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/uploads")
    GUARDRAILS_CONFIG_DIR: str = "guardrails_config"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @classmethod
    def validate(cls):
        """Validate configuration on startup"""
        required = ["OLLAMA_BASE_URL", "CHROMA_HOST"]
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Required config {var} not set")

# Validate on import
Config.validate()
```

### Usage

```python
from config import Config

# Access settings
url = Config.OLLAMA_BASE_URL
use_guardrails = Config.USE_GUARDRAILS
```

## Dependencies

### FastAPI Dependency Injection

**File**: `backend/deps.py`

```python
from fastapi import Depends
from rag_components import RAGComponents
from services.nemo import nemo_service

def get_rag_components() -> RAGComponents:
    """
    Dependency for RAG components singleton
    """
    return RAGComponents()

def get_nemo_service():
    """
    Dependency for NeMo Guardrails service
    """
    return nemo_service

# Usage in routers
from deps import get_rag_components

@router.post("/endpoint")
async def endpoint(
    request: MyRequest,
    rag: RAGComponents = Depends(get_rag_components)
):
    # rag is injected automatically
    docs = rag.retriever.get_relevant_documents(request.query)
    return {"docs": docs}
```

### Testing with Dependencies

```python
# tests/test_chat.py
from fastapi.testclient import TestClient
from main import app
from deps import get_rag_components

class MockRAG:
    def get_relevant_documents(self, query):
        return [{"content": "mock doc"}]

# Override dependency
app.dependency_overrides[get_rag_components] = lambda: MockRAG()

client = TestClient(app)

def test_chat():
    response = client.post("/api/chat", json={"query": "test"})
    assert response.status_code == 200
```

## Application Lifecycle

### Startup

**File**: `backend/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import chat_router, agents_router, model_router
from rag_components import RAGComponents
from config import Config

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    """
    # Startup
    print("Starting up...")

    # Initialize RAG components
    rag = RAGComponents()
    rag.initialize()

    # Validate configuration
    Config.validate()

    yield

    # Shutdown
    print("Shutting down...")
    # Cleanup if needed

app = FastAPI(
    title="NeMo Guardrails Testing API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router.router)
app.include_router(agents_router.router)
app.include_router(model_router.router)

@app.get("/")
async def root():
    return {
        "message": "NeMo Guardrails Testing API",
        "version": "1.0.0",
        "docs": "/docs"
    }
```

## Error Handling

### Custom Exception Handlers

```python
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

### Service-Level Error Handling

```python
class ChatService:
    async def process(self, request):
        try:
            # Processing logic
            pass
        except OllamaConnectionError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama: {e}"
            )
        except ChromaDBError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Database error: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in chat service: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal processing error"
            )
```

## Best Practices

### Async/Await

✅ **Do**: Use async for I/O operations
```python
async def fetch_data():
    response = await httpx_client.get(url)
    return response.json()
```

❌ **Don't**: Mix sync and async unnecessarily
```python
# Bad: Blocking call in async function
async def fetch_data():
    return requests.get(url)  # Blocks event loop!
```

### Type Hints

✅ **Do**: Use type hints everywhere
```python
def process_message(message: str, user_id: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {...}
    return result
```

### Logging

✅ **Do**: Structured logging
```python
import logging

logger = logging.getLogger(__name__)

logger.info(
    "Chat request processed",
    extra={
        "user_id": request.user_id,
        "model": request.model,
        "duration_ms": duration
    }
)
```

### Testing

✅ **Do**: Write unit tests
```python
import pytest
from services.chat import chat_service

@pytest.mark.asyncio
async def test_chat_service():
    request = ChatRequest(query="test", model="gemma3")
    result = await chat_service.process(request)
    assert result is not None
```

### Documentation

✅ **Do**: Document functions
```python
async def process_chat(request: ChatRequest) -> AsyncGenerator:
    """
    Process chat request with optional RAG and guardrails

    Args:
        request: ChatRequest with query, model, and options

    Returns:
        AsyncGenerator yielding SSE formatted events

    Raises:
        ValueError: If request validation fails
        HTTPException: If service is unavailable
    """
    pass
```

## Related Documentation

- [Architecture Overview](./ARCHITECTURE.md) - Backend architecture
- [Development Guide](./DEVELOPMENT.md) - Local backend development
- [API Reference](./API-REFERENCE.md) - API endpoints
- [RAG System](./RAG-SYSTEM.md) - RAG implementation
- [Guardrails Guide](./GUARDRAILS-GUIDE.md) - NeMo integration
- [Troubleshooting](./TROUBLESHOOTING.md) - Backend issues
