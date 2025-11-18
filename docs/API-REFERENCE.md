# API Reference

Complete documentation for all backend API endpoints in the NeMo Guardrails Testing Application.

## Table of Contents

1. [Overview](#overview)
2. [Base URL](#base-url)
3. [Authentication](#authentication)
4. [Common Headers](#common-headers)
5. [Response Formats](#response-formats)
6. [Endpoints](#endpoints)
   - [Chat](#chat-endpoints)
   - [Agents](#agent-endpoints)
   - [Models](#model-endpoints)
   - [Documents](#document-endpoints)
   - [Upload](#upload-endpoints)
   - [Automation](#automation-endpoints)
   - [Health](#health-endpoints)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)

## Overview

The backend API is built with FastAPI and provides RESTful endpoints for chat functionality, agent management, document handling, and system health monitoring.

**Interactive Documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Base URL

```
Development: http://localhost:8000
Production: Configure via environment
```

All endpoints are prefixed with `/api` except health endpoints.

## Authentication

**Current**: None (development only)

**Production Requirements**:
- API key in headers
- JWT tokens
- Rate limiting per user/IP

## Common Headers

```http
Content-Type: application/json
Accept: application/json
```

For file uploads:
```http
Content-Type: multipart/form-data
```

For SSE streaming:
```http
Accept: text/event-stream
```

## Response Formats

### Standard JSON Response
```json
{
  "status": "success",
  "data": { ... }
}
```

### Error Response
```json
{
  "detail": "Error message description"
}
```

### SSE Stream Response
```
data: {"token": "Hello "}
data: {"token": "World"}
data: {"status": "done"}
```

## Endpoints

## Chat Endpoints

### POST /api/chat

Stream a chat response with optional guardrails and RAG.

**Implementation**: `backend/routers/chat_router.py:25-68`

**Request Body**:
```json
{
  "query": "What is the weather like?",
  "model": "gemma3:4b-it-q4_K_M",
  "agent_name": "math_assistant",
  "history": [
    {
      "sender": "user",
      "text": "Hello"
    },
    {
      "sender": "bot",
      "text": "Hi! How can I help you?"
    }
  ],
  "use_rag": true
}
```

**Request Schema** (`backend/schemas.py:15-35`):
```python
class ChatRequest(BaseModel):
    query: str  # User's message
    model: str = "gemma3:4b-it-q4_K_M"  # Ollama model name
    agent_name: Optional[str] = None  # Guardrails agent
    history: List[Dict[str, str]] = []  # Conversation history
    use_rag: bool = False  # Enable RAG retrieval
```

**Response**: Server-Sent Events (SSE) stream

**SSE Event Format**:
```
# Token event (multiple)
data: {"token": "text fragment"}

# Completion event (final)
data: {"status": "done"}

# Error event
data: {"error": "error message"}
```

**Example Request** (curl):
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain lift in aviation",
    "model": "gemma3:latest",
    "agent_name": "aviation_assistant",
    "use_rag": false,
    "history": []
  }'
```

**Example Response** (SSE stream):
```
data: {"token": "Lift "}
data: {"token": "is "}
data: {"token": "the "}
data: {"token": "force "}
data: {"token": "that "}
data: {"token": "keeps "}
data: {"token": "an "}
data: {"token": "aircraft "}
data: {"token": "airborne"}
data: {"token": "."}
data: {"status": "done"}
```

**Behavior**:
- If `USE_GUARDRAILS=true` and `agent_name` provided → Routes through NeMo Guardrails
- If `use_rag=true` → Retrieves context from ChromaDB before generating
- If `history` provided → Includes conversation context
- Streams response in real-time via SSE

**Status Codes**:
- `200 OK` - Stream started successfully
- `400 Bad Request` - Invalid request parameters
- `500 Internal Server Error` - Server error during processing

---

## Agent Endpoints

### GET /api/agents/metadata

Get metadata for all configured guardrail agents.

**Implementation**: `backend/routers/agents_router.py:15-38`

**Request**: No parameters

**Response**:
```json
{
  "agents": [
    {
      "id": "aviation_assistant",
      "name": "Aviation Assistant",
      "description": "Specialized in flight operations, aircraft systems, and aviation regulations",
      "icon": "✈️",
      "persona": "Expert aviation professional with deep knowledge of flight operations"
    },
    {
      "id": "bank_assistant",
      "name": "Banking Assistant",
      "description": "Expert in banking services, financial products, and transactions",
      "icon": "🏦",
      "persona": "Professional banking advisor"
    },
    {
      "id": "math_assistant",
      "name": "Math Assistant",
      "description": "Solves mathematical problems and explains concepts",
      "icon": "🔢",
      "persona": "Patient mathematics tutor"
    }
  ]
}
```

**Example Request**:
```bash
curl http://localhost:8000/api/agents/metadata
```

**Status Codes**:
- `200 OK` - Success
- `500 Internal Server Error` - Failed to read metadata

**Source**: `backend/guardrails_config/metadata.yaml`

---

### GET /api/agents/available

List all available (valid) guardrail agent configurations.

**Implementation**: `backend/routers/agents_router.py:42-58`

**Response**:
```json
{
  "agents": [
    "aviation_assistant",
    "bank_assistant",
    "math_assistant"
  ]
}
```

**Validation**: Only returns agents with valid `config.yml` and `config.co` files

**Example Request**:
```bash
curl http://localhost:8000/api/agents/available
```

**Status Codes**:
- `200 OK` - Success
- `500 Internal Server Error` - Failed to list agents

---

### GET /api/agents/validate/{agent_name}

Validate a specific agent configuration.

**Implementation**: `backend/routers/agents_router.py:62-85`

**Path Parameters**:
- `agent_name` (string, required) - Agent identifier

**Response** (valid):
```json
{
  "valid": true,
  "agent_name": "aviation_assistant",
  "config_path": "/home/ldg/NeMo_App/backend/guardrails_config/aviation_assistant"
}
```

**Response** (invalid):
```json
{
  "valid": false,
  "agent_name": "invalid_agent",
  "error": "Missing config.yml or config.co file"
}
```

**Example Request**:
```bash
curl http://localhost:8000/api/agents/validate/aviation_assistant
```

**Status Codes**:
- `200 OK` - Validation completed (check `valid` field)
- `500 Internal Server Error` - Validation error

---

## Model Endpoints

### GET /api/models

List all available Ollama models.

**Implementation**: `backend/routers/model_router.py:15-45`

**Response**:
```json
{
  "models": [
    {
      "name": "gemma3:4b-it-q4_K_M",
      "modified_at": "2024-12-15T10:30:00Z",
      "size": 4200000000
    },
    {
      "name": "gemma3:latest",
      "modified_at": "2024-12-14T08:15:00Z",
      "size": 8500000000
    },
    {
      "name": "llama2:13b",
      "modified_at": "2024-12-10T14:20:00Z",
      "size": 13000000000
    }
  ]
}
```

**Response Schema** (`backend/schemas.py:45-55`):
```python
class OllamaModelInfo(BaseModel):
    name: str  # Model identifier
    modified_at: str  # ISO 8601 timestamp
    size: int  # Size in bytes
```

**Example Request**:
```bash
curl http://localhost:8000/api/models
```

**Status Codes**:
- `200 OK` - Success
- `500 Internal Server Error` - Cannot connect to Ollama
- `503 Service Unavailable` - Ollama not running

**Troubleshooting**:
- Ensure Ollama is running: `ollama list`
- Check Ollama URL: `OLLAMA_BASE_URL` in config
- Default: http://localhost:11434

---

## Document Endpoints

### GET /api/documents

List all documents stored in the vector database.

**Implementation**: `backend/routers/document_router.py:15-52`

**Response**:
```json
{
  "count": 2,
  "documents": [
    {
      "filename": "aviation_manual.pdf",
      "chunks": [
        {
          "id": "doc_1_chunk_0",
          "content": "Chapter 1: Introduction to Flight...",
          "metadata": {
            "source": "aviation_manual.pdf",
            "page": 1,
            "chunk_index": 0
          }
        },
        {
          "id": "doc_1_chunk_1",
          "content": "Lift is generated when air flows...",
          "metadata": {
            "source": "aviation_manual.pdf",
            "page": 2,
            "chunk_index": 1
          }
        }
      ]
    },
    {
      "filename": "banking_guide.pdf",
      "chunks": [
        {
          "id": "doc_2_chunk_0",
          "content": "Types of bank accounts...",
          "metadata": {
            "source": "banking_guide.pdf",
            "page": 1,
            "chunk_index": 0
          }
        }
      ]
    }
  ]
}
```

**Response Schema** (`backend/schemas.py:65-85`):
```python
class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]

class DocumentInfo(BaseModel):
    filename: str
    chunks: List[DocumentChunk]

class DocumentListResponse(BaseModel):
    count: int
    documents: List[DocumentInfo]
```

**Example Request**:
```bash
curl http://localhost:8000/api/documents
```

**Status Codes**:
- `200 OK` - Success (returns empty list if no documents)
- `500 Internal Server Error` - ChromaDB connection error

---

## Upload Endpoints

### POST /api/upload

Upload a PDF file for RAG processing.

**Implementation**: `backend/routers/upload_router.py:18-68`

**Request**: Multipart form data

**Form Fields**:
- `file` (file, required) - PDF file to upload

**Response**:
```json
{
  "message": "File uploaded and processed successfully",
  "filename": "aviation_manual.pdf",
  "chunks_added": 15
}
```

**Response Schema** (`backend/schemas.py:95-105`):
```python
class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int
```

**Example Request** (curl):
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/document.pdf"
```

**Example Request** (JavaScript):
```javascript
const formData = new FormData();
formData.append('file', pdfFile);

fetch('http://localhost:8000/api/upload', {
  method: 'POST',
  body: formData
})
  .then(res => res.json())
  .then(data => console.log(data));
```

**Processing Steps**:
1. Validate file is PDF
2. Save temporarily
3. Extract text with PyPDF
4. Chunk into segments (configurable size)
5. Generate embeddings (all-MiniLM-L6-v2)
6. Store in ChromaDB with metadata
7. Return chunk count

**Chunk Configuration** (`backend/config.py:55-60`):
```python
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
```

**Status Codes**:
- `200 OK` - File processed successfully
- `400 Bad Request` - Invalid file type or missing file
- `413 Payload Too Large` - File exceeds size limit
- `500 Internal Server Error` - Processing error

**Limitations**:
- Only PDF files supported
- File size limit (configure in production)
- Text-based PDFs only (no OCR for scanned documents)

---

## Automation Endpoints

### POST /api/automate

Automate conversation tasks like summarization or action extraction.

**Implementation**: `backend/routers/automate_router.py:15-78`

**Request Body**:
```json
{
  "conversation_history": [
    {
      "sender": "user",
      "text": "We need to update the flight manual"
    },
    {
      "sender": "bot",
      "text": "I can help with that. What sections?"
    },
    {
      "sender": "user",
      "text": "Chapters 3 and 5 on emergency procedures"
    }
  ],
  "model": "gemma3:latest",
  "task": "summarize",
  "config": {
    "max_length": 150
  }
}
```

**Request Schema** (`backend/schemas.py:115-135`):
```python
class AutomateRequest(BaseModel):
    conversation_history: List[Dict[str, str]]
    model: str = "gemma3:latest"
    task: str = "summarize"  # "summarize" or "extract_actions"
    config: Optional[Dict[str, Any]] = {}
```

**Response**:
```json
{
  "status": "success",
  "task": "summarize",
  "data": {
    "summary": "User requested updates to flight manual chapters 3 and 5, specifically emergency procedures sections.",
    "action_items": [
      "Update flight manual chapter 3",
      "Update flight manual chapter 5",
      "Focus on emergency procedures"
    ]
  }
}
```

**Response Schema** (`backend/schemas.py:145-155`):
```python
class AutomateResponse(BaseModel):
    status: str
    task: str
    data: Dict[str, Any]
```

**Supported Tasks**:

| Task | Description | Output |
|------|-------------|--------|
| `summarize` | Generate conversation summary | `data.summary` (string) |
| `extract_actions` | Extract action items/todos | `data.action_items` (array) |

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/automate \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_history": [...],
    "task": "summarize"
  }'
```

**Status Codes**:
- `200 OK` - Success
- `400 Bad Request` - Invalid task type or empty history
- `500 Internal Server Error` - Processing error

---

## Health Endpoints

### GET /health

Check system health and service connectivity.

**Implementation**: `backend/routers/health_router.py:12-35`

**Response** (healthy):
```json
{
  "status": "healthy",
  "timestamp": "2024-12-15T14:30:00Z",
  "services": {
    "ollama": {
      "status": "connected",
      "url": "http://localhost:11434"
    },
    "chromadb": {
      "status": "connected",
      "url": "http://localhost:8001"
    }
  }
}
```

**Response** (degraded):
```json
{
  "status": "degraded",
  "timestamp": "2024-12-15T14:30:00Z",
  "services": {
    "ollama": {
      "status": "disconnected",
      "url": "http://localhost:11434",
      "error": "Connection refused"
    },
    "chromadb": {
      "status": "connected",
      "url": "http://localhost:8001"
    }
  }
}
```

**Response Schema** (`backend/schemas.py:165-185`):
```python
class ServiceStatus(BaseModel):
    status: str  # "connected" or "disconnected"
    url: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str  # ISO 8601
    services: Dict[str, ServiceStatus]
```

**Example Request**:
```bash
curl http://localhost:8000/health
```

**Status Determination**:
- `healthy` - All services connected
- `degraded` - Some services disconnected
- `unhealthy` - All services disconnected

**Status Codes**:
- `200 OK` - Health check completed (check `status` field)
- `503 Service Unavailable` - Backend cannot start

**Use Cases**:
- Deployment health checks
- Monitoring/alerting
- Pre-flight validation
- Troubleshooting connectivity

---

### GET /

Root endpoint with API information.

**Implementation**: `backend/main.py:35-42`

**Response**:
```json
{
  "message": "NeMo Guardrails Testing API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

**Example Request**:
```bash
curl http://localhost:8000/
```

---

## Error Handling

### Standard Error Response

All endpoints return errors in FastAPI's standard format:

```json
{
  "detail": "Descriptive error message"
}
```

### Common HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 400 | Bad Request | Invalid parameters, missing fields |
| 404 | Not Found | Invalid endpoint or resource |
| 413 | Payload Too Large | File upload exceeds size limit |
| 422 | Unprocessable Entity | Pydantic validation failure |
| 500 | Internal Server Error | Server-side processing error |
| 503 | Service Unavailable | External service (Ollama/ChromaDB) down |

### Validation Errors (422)

When request validation fails:

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Example**: Missing `query` field in chat request

### Service Connection Errors (503)

When external services are unavailable:

```json
{
  "detail": "Cannot connect to Ollama at http://localhost:11434"
}
```

**Troubleshooting**:
1. Check service status: `curl http://localhost:8000/health`
2. Verify Ollama: `ollama list`
3. Verify ChromaDB: `curl http://localhost:8001/api/v1/heartbeat`

## Rate Limiting

**Current**: No rate limiting (development only)

**Production Implementation** (recommended):

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_endpoint(...):
    pass
```

**Suggested Limits**:
- `/api/chat`: 10 requests/minute per IP
- `/api/upload`: 5 requests/minute per IP
- `/api/models`: 30 requests/minute per IP
- Other endpoints: 60 requests/minute per IP

## CORS Configuration

**Current** (`backend/main.py:25-32`):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development only!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production Configuration**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
```

## API Versioning

**Current**: No versioning (v1 implicit)

**Future Versioning Strategy**:
```
/api/v1/chat
/api/v2/chat
```

Or via headers:
```http
Accept: application/vnd.api+json; version=1
```

## Client Examples

### Python Client

```python
import requests
import json

# Chat with streaming
def chat_stream(query, model="gemma3:latest"):
    url = "http://localhost:8000/api/chat"
    payload = {
        "query": query,
        "model": model,
        "use_rag": False
    }

    response = requests.post(url, json=payload, stream=True)

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data = json.loads(line_str[6:])
                if 'token' in data:
                    print(data['token'], end='', flush=True)
                elif 'status' in data and data['status'] == 'done':
                    print()
                    break

# Upload file
def upload_pdf(file_path):
    url = "http://localhost:8000/api/upload"
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        return response.json()

# Get health
def check_health():
    response = requests.get("http://localhost:8000/health")
    return response.json()
```

### JavaScript/TypeScript Client

```typescript
// Chat with SSE
async function chatStream(query: string, model: string = "gemma3:latest") {
  const eventSource = new EventSource(
    `http://localhost:8000/api/chat?${new URLSearchParams({
      query,
      model
    })}`
  );

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.token) {
      console.log(data.token);
    } else if (data.status === 'done') {
      eventSource.close();
    }
  };

  eventSource.onerror = (error) => {
    console.error('SSE Error:', error);
    eventSource.close();
  };
}

// Upload file
async function uploadPDF(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/api/upload', {
    method: 'POST',
    body: formData
  });

  return await response.json();
}

// Get models
async function getModels() {
  const response = await fetch('http://localhost:8000/api/models');
  return await response.json();
}
```

## Related Documentation

- [Architecture Overview](./ARCHITECTURE.md) - System design and data flow
- [Backend Guide](./BACKEND-GUIDE.md) - Implementation details
- [Development Guide](./DEVELOPMENT.md) - Local development setup
- [Troubleshooting](./TROUBLESHOOTING.md) - Common API issues
