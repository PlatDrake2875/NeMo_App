# System Architecture

This document describes the complete architecture of the NeMo Guardrails Testing Application, including system design, component interactions, data flow, and architectural patterns.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Backend Architecture](#backend-architecture)
3. [Frontend Architecture](#frontend-architecture)
4. [Data Flow](#data-flow)
5. [Component Interactions](#component-interactions)
6. [Design Patterns](#design-patterns)
7. [Technology Choices](#technology-choices)

## High-Level Architecture

The application follows a **microservices architecture** with clear separation between frontend, backend, and supporting services.

```
┌─────────────────────────────────────────────────────────────┐
│                      User Browser                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         React Frontend (Port 5173)                    │  │
│  │  - Components (Chat, Agents, Documents)               │  │
│  │  - State Management (Sessions, Theme)                 │  │
│  │  - SSE Client (Streaming)                             │  │
│  └────────────────────┬──────────────────────────────────┘  │
└────────────────────────┼──────────────────────────────────────┘
                        │ HTTP/SSE
                        ▼
┌─────────────────────────────────────────────────────────────┐
│            FastAPI Backend (Port 8000)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Routers (HTTP Layer)                               │   │
│  │  - Chat, Agents, Models, Upload, Documents          │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Services (Business Logic)                          │   │
│  │  - Chat Service, NeMo Service, RAG Service          │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  RAG Components (Singleton)                         │   │
│  │  - ChromaDB Client, Embeddings, Retriever           │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────┬──────────────────────┬─────────────────────────┘
             │                      │
             ▼                      ▼
┌────────────────────┐   ┌──────────────────────┐
│  ChromaDB          │   │  Ollama              │
│  (Port 8001)       │   │  (Port 11434 - Host) │
│  - Vector Store    │   │  - LLM Provider      │
│  - Embeddings      │   │  - Multiple Models   │
└────────────────────┘   └──────────────────────┘
             │                      │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │  NeMo Guardrails     │
             │  - Agent Configs     │
             │  - YAML + Colang     │
             │  - Input/Output Rails│
             └──────────────────────┘
```

### Service Responsibilities

| Service | Port | Responsibility |
|---------|------|----------------|
| Frontend | 5173 | User interface, session management, SSE streaming client |
| Backend | 8000 | API gateway, business logic, guardrails orchestration |
| ChromaDB | 8001 | Vector storage, similarity search, document embeddings |
| Ollama | 11434 | LLM inference, model management |

## Backend Architecture

The backend follows a **layered architecture** with clear separation of concerns:

### Layer Structure

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Layer                           │
│  Routers: Handle HTTP requests/responses, validation    │
│  Files: backend/routers/*.py                            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Service Layer                          │
│  Services: Business logic, orchestration                │
│  Files: backend/services/*.py                           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Component Layer                          │
│  RAG Components: Singleton instances, shared resources  │
│  File: backend/rag_components.py                        │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              External Services                          │
│  ChromaDB, Ollama, NeMo Guardrails                      │
└─────────────────────────────────────────────────────────┘
```

### Routers (HTTP Layer)

Located in `backend/routers/`, routers are thin HTTP handlers responsible for:
- Request validation (Pydantic schemas)
- Response formatting
- HTTP status codes
- Delegating to services

**Key Routers:**

| Router | File | Endpoints | Purpose |
|--------|------|-----------|---------|
| Chat | `chat_router.py:15-120` | POST `/api/chat` | Streaming chat interface |
| Agents | `agents_router.py:10-85` | GET `/api/agents/*` | Agent metadata and validation |
| Models | `model_router.py:8-45` | GET `/api/models` | Ollama model listing |
| Upload | `upload_router.py:12-68` | POST `/api/upload` | PDF upload handling |
| Documents | `document_router.py:9-52` | GET `/api/documents` | Document retrieval |
| Health | `health_router.py:7-35` | GET `/health` | Health checks |

**Example Router Pattern:**

```python
# backend/routers/chat_router.py:25-45
@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    1. Validate request (Pydantic)
    2. Delegate to service
    3. Stream response via SSE
    """
    return StreamingResponse(
        chat_service.process_chat(request),
        media_type="text/event-stream"
    )
```

### Services (Business Logic Layer)

Located in `backend/services/`, services contain core business logic:

| Service | File | Responsibilities |
|---------|------|------------------|
| Chat | `chat.py:20-180` | Message processing, RAG integration, streaming |
| NeMo | `nemo.py:15-145` | Guardrails initialization, agent management |
| Model | `model.py:10-65` | Ollama model queries |
| Upload | `upload.py:18-95` | PDF processing, chunking, embedding |
| Document | `document.py:12-78` | ChromaDB document queries |
| Health | `health.py:8-55` | Service health verification |

**Chat Service Flow:**

```python
# backend/services/chat.py:45-120
async def process_chat(request: ChatRequest):
    """
    1. Check if RAG is enabled
    2. Retrieve relevant documents if RAG active
    3. Build prompt with context
    4. Route to NeMo Guardrails or direct Ollama
    5. Stream tokens via SSE
    """

    # RAG context retrieval
    if request.use_rag:
        context = retriever.get_relevant_documents(request.query)
        prompt = build_rag_prompt(request.query, context)
    else:
        prompt = request.query

    # Guardrails routing
    if USE_GUARDRAILS and request.agent_name:
        response = nemo_service.generate(request.agent_name, prompt)
    else:
        response = ollama_client.generate(request.model, prompt)

    # Stream response
    for token in response:
        yield f"data: {json.dumps({'token': token})}\n\n"
```

### RAG Components (Singleton Layer)

File: `backend/rag_components.py:15-125`

**Singleton Pattern Implementation:**

```python
class RAGComponents:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        # Initialize once, reuse everywhere
        self.chroma_client = chromadb.HttpClient(...)
        self.embedding_function = HuggingFaceEmbeddings(...)
        self.vector_store = Chroma(...)
        self.retriever = vector_store.as_retriever(...)
        self._initialized = True
```

**Managed Resources:**
- ChromaDB HTTP client
- HuggingFace embedding model (`all-MiniLM-L6-v2`)
- Vector store instance
- Document retriever
- LLM instances (Ollama)
- Prompt templates

**Benefits:**
- Single initialization (performance)
- Shared across all requests
- Connection pooling
- Memory efficiency

### Configuration Management

File: `backend/config.py:10-85`

**Environment-based Configuration:**

```python
# backend/config.py:15-50
class Config:
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_FOR_RAG = os.getenv("OLLAMA_MODEL_FOR_RAG", "gemma3:latest")

    # ChromaDB settings
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

    # Feature flags
    USE_GUARDRAILS = os.getenv("USE_GUARDRAILS", "true").lower() == "true"
    RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"

    # Embedding settings
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Prompt templates
    RAG_PROMPT_TEMPLATE = """..."""
    SIMPLE_PROMPT_TEMPLATE = """..."""
```

### Dependency Injection

File: `backend/deps.py:8-45`

**FastAPI Dependencies:**

```python
# backend/deps.py:12-30
def get_rag_components() -> RAGComponents:
    """Dependency for RAG components singleton"""
    return RAGComponents()

def get_nemo_service() -> NeMoService:
    """Dependency for NeMo Guardrails service"""
    return NeMoService()

# Usage in routers
@router.post("/chat")
async def chat(
    request: ChatRequest,
    rag: RAGComponents = Depends(get_rag_components)
):
    # rag is injected automatically
    pass
```

## Frontend Architecture

The frontend follows a **component-based React architecture** with hooks for state management.

### Component Hierarchy

```
App.jsx (Root)
├── Header.jsx
│   ├── Theme Toggle
│   └── Title
├── Sidebar.jsx
│   ├── Session List
│   ├── New Session Button
│   ├── Model Selector
│   ├── Agent Selector Button
│   └── Document Viewer Button
└── ChatInterface.jsx
    ├── ChatHistory.jsx
    │   └── MarkdownMessage.jsx (for each message)
    └── ChatForm.jsx
        ├── Text Input
        ├── RAG Toggle
        └── Submit Button

Modals (Overlays)
├── AgentSelector.jsx
└── DocumentViewer.jsx
```

### State Management Strategy

**No Redux/Context API** - Uses custom hooks and local component state:

| Hook | File | Purpose |
|------|------|---------|
| `useChatSessions` | `hooks/useChatSessions.js:10-180` | Session CRUD, localStorage sync |
| `useTheme` | `hooks/useTheme.js:8-45` | Theme toggle, persistence |

**useChatSessions Hook Pattern:**

```javascript
// frontend/src/hooks/useChatSessions.js:25-120
export function useChatSessions() {
  const [sessions, setSessions] = useState(() => {
    // Initialize from localStorage
    const stored = localStorage.getItem('chatSessions');
    return stored ? JSON.parse(stored) : [];
  });

  const [currentSessionId, setCurrentSessionId] = useState(null);

  // Sync to localStorage on change
  useEffect(() => {
    localStorage.setItem('chatSessions', JSON.stringify(sessions));
  }, [sessions]);

  // CRUD operations
  const createSession = () => { /* ... */ };
  const deleteSession = (id) => { /* ... */ };
  const updateSession = (id, updates) => { /* ... */ };

  return { sessions, currentSessionId, createSession, ... };
}
```

### Component Responsibilities

**App.jsx** (`frontend/src/App.jsx:15-180`)
- Root component
- Session state management
- Model and agent selection
- Coordinate child components

**ChatInterface.jsx** (`frontend/src/components/ChatInterface.jsx:20-150`)
- Display current chat
- Handle message submission
- SSE streaming client
- Auto-scroll management

**ChatHistory.jsx** (`frontend/src/components/ChatHistory.jsx:15-95`)
- Render message list
- Scroll container
- Loading states

**ChatForm.jsx** (`frontend/src/components/ChatForm.jsx:12-85`)
- Message input
- RAG toggle
- Submit handling
- Loading state

**Sidebar.jsx** (`frontend/src/components/Sidebar.jsx:18-140`)
- Session list
- Session CRUD operations
- Model selector dropdown
- Control buttons

**AgentSelector.jsx** (`frontend/src/components/AgentSelector.jsx:15-120`)
- Modal overlay
- Agent grid display
- Agent selection
- Metadata display

**DocumentViewer.jsx** (`frontend/src/components/DocumentViewer.jsx:18-105`)
- Document list
- Upload interface
- Document metadata

**MarkdownMessage.jsx** (`frontend/src/components/MarkdownMessage.jsx:10-65`)
- Markdown rendering
- Code syntax highlighting
- User/bot styling

## Data Flow

### Chat Message Flow (with Guardrails and RAG)

```
┌─────────────┐
│  1. User    │
│  Input      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  2. ChatForm Component                  │
│  - Captures message                     │
│  - Reads RAG toggle state               │
│  - Reads selected model                 │
│  - Reads selected agent                 │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  3. ChatInterface Component             │
│  - Builds ChatRequest object            │
│  - Adds conversation history            │
│  - Initiates SSE connection             │
└──────┬──────────────────────────────────┘
       │ POST /api/chat
       ▼
┌─────────────────────────────────────────┐
│  4. Backend: chat_router.py:25          │
│  - Validates ChatRequest (Pydantic)     │
│  - Extracts parameters                  │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  5. Backend: chat.py:45 (Service)       │
│  - Checks use_rag flag                  │
└──────┬──────────────────────────────────┘
       │
       ├─ if use_rag = true ───────────────┐
       │                                   ▼
       │                        ┌──────────────────────┐
       │                        │  6a. RAG Retrieval   │
       │                        │  - Query embeddings  │
       │                        │  - ChromaDB search   │
       │                        │  - Get top-k docs    │
       │                        └──────────┬───────────┘
       │                                   │
       │◄──────────────────────────────────┘
       │  Context documents
       ▼
┌─────────────────────────────────────────┐
│  7. Build Prompt                        │
│  - Use RAG template with context        │
│  - Or simple template without context   │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  8. Route to NeMo or Direct Ollama      │
│  - If USE_GUARDRAILS + agent selected:  │
│    → nemo.py:65 (NeMo Service)          │
│  - Else:                                │
│    → Direct Ollama API call             │
└──────┬──────────────────────────────────┘
       │
       ├─ NeMo Guardrails Path ────────────┐
       │                                   ▼
       │                    ┌──────────────────────────┐
       │                    │  9a. NeMo Processing     │
       │                    │  - Load agent config     │
       │                    │  - Apply input rails     │
       │                    │  - Call Ollama via NeMo  │
       │                    │  - Apply output rails    │
       │                    └──────────┬───────────────┘
       │                               │
       │◄──────────────────────────────┘
       │  Guardrails response
       │
       ▼
┌─────────────────────────────────────────┐
│  10. Stream Response via SSE            │
│  - Yield tokens as JSON                 │
│  - Format: data: {"token": "..."}       │
│  - End with: data: {"status": "done"}   │
└──────┬──────────────────────────────────┘
       │ SSE Stream
       ▼
┌─────────────────────────────────────────┐
│  11. Frontend: SSE Event Handler        │
│  - Parse JSON tokens                    │
│  - Append to current message            │
│  - Update UI in real-time               │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  12. ChatHistory Component              │
│  - Render markdown                      │
│  - Auto-scroll to bottom                │
│  - Save to session                      │
└─────────────────────────────────────────┘
```

### Document Upload Flow

```
┌──────────────┐
│  1. User     │
│  Selects PDF │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────┐
│  2. DocumentViewer Component     │
│  - File input handler            │
│  - FormData creation             │
└──────┬───────────────────────────┘
       │ POST /api/upload (multipart)
       ▼
┌──────────────────────────────────┐
│  3. upload_router.py:25          │
│  - Receive file                  │
│  - Validate PDF format           │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  4. upload.py:35 (Service)       │
│  - Save temporarily              │
│  - Extract text with PyPDF       │
│  - Chunk into segments           │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  5. Generate Embeddings          │
│  - Use HuggingFace model         │
│  - Model: all-MiniLM-L6-v2       │
│  - Create vector for each chunk  │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  6. Store in ChromaDB            │
│  - Add to collection             │
│  - Include metadata              │
│  - Return chunk count            │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  7. Response to Frontend         │
│  - Success message               │
│  - Filename                      │
│  - Chunks added count            │
└──────────────────────────────────┘
```

## Component Interactions

### NeMo Guardrails Integration

File: `backend/services/nemo.py:15-145`

```python
# Initialization
class NeMoService:
    def __init__(self):
        self.rails_apps = {}  # Cache of loaded agents

    def get_or_create_rails(self, agent_name: str):
        """Load and cache NeMo Guardrails instance"""
        if agent_name in self.rails_apps:
            return self.rails_apps[agent_name]

        # Load configuration
        config_path = f"guardrails_config/{agent_name}"
        rails_config = RailsConfig.from_path(config_path)

        # Create LLMRails instance
        rails_app = LLMRails(rails_config)

        # Cache for reuse
        self.rails_apps[agent_name] = rails_app
        return rails_app

    async def generate(self, agent_name: str, message: str):
        """Generate response with guardrails"""
        rails = self.get_or_create_rails(agent_name)
        response = await rails.generate_async(messages=[{
            "role": "user",
            "content": message
        }])
        return response["content"]
```

### ChromaDB Integration

File: `backend/rag_components.py:45-90`

```python
# Initialization
chroma_client = chromadb.HttpClient(
    host=Config.CHROMA_HOST,
    port=Config.CHROMA_PORT
)

embedding_function = HuggingFaceEmbeddings(
    model_name=Config.EMBEDDING_MODEL_NAME
)

vector_store = Chroma(
    client=chroma_client,
    collection_name="documents",
    embedding_function=embedding_function
)

# Retrieval
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Top 3 documents
)

# Usage in chat service
docs = retriever.get_relevant_documents(query)
context = "\n\n".join([doc.page_content for doc in docs])
```

## Design Patterns

### 1. Singleton Pattern
**Where**: `backend/rag_components.py:15-125`
**Why**: Single instance of expensive resources (ChromaDB client, embeddings model)
**Benefit**: Performance, memory efficiency, connection pooling

### 2. Dependency Injection
**Where**: `backend/deps.py:8-45`, all routers
**Why**: Testability, loose coupling
**Benefit**: Easy mocking, service replacement

### 3. Service Layer Pattern
**Where**: `backend/services/*.py`
**Why**: Separate business logic from HTTP handling
**Benefit**: Reusability, testability, maintainability

### 4. Repository Pattern (Implicit)
**Where**: RAG components act as repository for documents
**Why**: Abstract data access
**Benefit**: Can swap ChromaDB for other vector stores

### 5. Streaming Pattern
**Where**: SSE in chat router and frontend
**Why**: Real-time token delivery
**Benefit**: Better UX, progressive rendering

### 6. Hook Pattern (Frontend)
**Where**: `frontend/src/hooks/*.js`
**Why**: Reusable stateful logic
**Benefit**: Clean components, shared behavior

## Technology Choices

### Why FastAPI?
- Native async support for SSE streaming
- Automatic OpenAPI documentation
- Pydantic validation built-in
- High performance (Starlette + Uvicorn)

### Why React with Hooks?
- Component reusability
- Hooks avoid Context API complexity
- Great ecosystem (react-markdown, etc.)
- Excellent developer experience with Vite

### Why ChromaDB?
- Simple HTTP API
- Built-in embedding support
- Good for development/testing
- Easy Docker deployment

### Why Ollama?
- Local LLM deployment
- No external API dependencies
- Multiple model support
- Simple REST API

### Why uv (not pip)?
- Faster dependency resolution
- Better lock file management
- Modern Python tooling
- Compatible with pip/poetry

### Why Server-Sent Events (not WebSockets)?
- Simpler protocol (HTTP)
- Auto-reconnection in browsers
- Perfect for one-way streaming
- No need for bidirectional communication

## Security Considerations

### Current State (Development)
- **No authentication** - Open endpoints
- **No rate limiting** - Unrestricted requests
- **File uploads** - Limited validation
- **CORS** - Wide open

### Production Requirements
Would need:
- API key authentication
- Rate limiting (per user/IP)
- File upload restrictions (size, type, scan)
- CORS configuration
- HTTPS enforcement
- Input sanitization (XSS, injection)
- Secrets management (not in .env files)

## Scalability Considerations

### Current Limitations
- Single Ollama instance
- ChromaDB single node
- No caching layer
- No load balancing

### Scale-Out Strategy
1. **Backend**: Multiple FastAPI instances behind load balancer
2. **ChromaDB**: Distributed deployment or managed service
3. **Ollama**: Multiple instances with routing logic
4. **Caching**: Redis for session/response caching
5. **Queue**: Celery for async document processing

## Deployment Architecture

See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete deployment documentation.

**Development Setup**:
```
Docker Compose
├── Frontend Container (Vite dev server)
├── Backend Container (Uvicorn --reload)
└── ChromaDB Container

Host Machine
└── Ollama (not containerized)
```

**Benefits**:
- Hot reload for both frontend and backend
- Isolated ChromaDB
- Host Ollama for GPU access
- Volume mounts for live code updates

## Related Documentation

- [API Reference](./API-REFERENCE.md) - Complete API documentation
- [Backend Guide](./BACKEND-GUIDE.md) - Detailed backend implementation
- [Frontend Guide](./FRONTEND-GUIDE.md) - Detailed frontend implementation
- [RAG System](./RAG-SYSTEM.md) - RAG architecture and implementation
- [Guardrails Guide](./GUARDRAILS-GUIDE.md) - NeMo Guardrails configuration
