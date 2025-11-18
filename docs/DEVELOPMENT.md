# Development Guide

Complete guide for local development of the NeMo Guardrails Testing Application.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Development Workflows](#development-workflows)
4. [Backend Development](#backend-development)
5. [Frontend Development](#frontend-development)
6. [Testing](#testing)
7. [Debugging](#debugging)
8. [Code Quality](#code-quality)
9. [Hot Reload](#hot-reload)
10. [Common Tasks](#common-tasks)

## Prerequisites

### Required Software

| Software | Minimum Version | Purpose |
|----------|----------------|---------|
| Docker | 20.10+ | Container runtime (includes vLLM) |
| Docker Compose | 2.0+ | Multi-container orchestration |
| Python | 3.9+ | Backend development |
| Node.js | 18+ | Frontend development |
| uv | Latest | Python package manager |
| Git | 2.0+ | Version control |

### Optional Software

| Software | Purpose |
|----------|---------|
| VS Code | Recommended IDE |
| Postman | API testing |
| Chrome DevTools | Frontend debugging |

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| CPU | 4 cores | 8+ cores |
| Disk Space | 20GB | 50GB+ |
| GPU | - | NVIDIA GPU for faster LLM inference |

## Initial Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd NeMo_App
```

### 2. Start Docker Services

vLLM and other services run in Docker containers. Start them with:

```bash
docker-compose up -d
```

This will:
- Start **vLLM** with the default model (`meta-llama/Llama-3.2-3B-Instruct`)
- Start **PostgreSQL** with pgvector extension
- Start **ChromaDB** for vector storage (if configured)

The model will be downloaded automatically on first run. To use a different model, modify `MODEL_NAME` in `docker-compose.yml`.

**Verify vLLM is running**:
```bash
curl http://localhost:8000/v1/models
```

### 3. Install Backend Dependencies

```bash
cd backend

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Verify installation
uv pip list
```

### 5. Install Frontend Dependencies

```bash
cd ../frontend

# Install packages
npm install

# Verify installation
npm list --depth=0
```

### 6. Setup Environment

```bash
cd ..

# Copy environment template
cp deploy/.env.example deploy/.env

# Edit configuration (optional)
nano deploy/.env
```

**Key Variables**:
```bash
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_HOST=localhost
CHROMA_PORT=8001
USE_GUARDRAILS=true
RAG_ENABLED=true
```

### 7. Start Services

**Option A: Using Deployment Script**
```bash
./deploy/start.sh
```

**Option B: Using Docker Compose**
```bash
docker-compose --profile fullstack up -d
```

**Option C: Using Makefile**
```bash
make up-fullstack
```

### 8. Verify Installation

```bash
# Check backend health
curl http://localhost:8000/health

# Check ChromaDB
curl http://localhost:8001/api/v1/heartbeat

# Check vLLM
curl http://localhost:8000/v1/models

# Access frontend
# Open http://localhost:5173 in browser
```

Expected response:
```json
{
  "status": "healthy",
  "vllm": {
    "status": "connected",
    "details": "vLLM service is responsive."
  },
  "postgres": {
    "status": "connected",
    "details": "PostgreSQL service is responsive."
  }
}
```

## Development Workflows

### Full-Stack Development (Docker)

**Best for**: Testing full integration, deployment simulation

```bash
# Start all services
docker-compose --profile fullstack up

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Hot Reload**: Enabled via volume mounts
- Backend: `backend/` → `/app` in container
- Frontend: `frontend/` → `/app` in container

### Backend-Only Development

**Best for**: API development, service logic, guardrails testing

```bash
# Terminal 1: Start ChromaDB
docker-compose up chromadb -d

# Terminal 2: Run backend locally
cd backend
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Benefits**:
- Faster restarts
- Direct debugging
- IDE integration
- No container overhead

### Frontend-Only Development

**Best for**: UI development, component work, styling

```bash
# Ensure backend is running (Docker or local)

# Terminal: Run frontend dev server
cd frontend
npm run dev
```

**Benefits**:
- Vite HMR (instant updates)
- Fast rebuilds
- Browser DevTools
- Component isolation

### Hybrid Development

```bash
# Terminal 1: Backend locally
cd backend
uv run uvicorn main:app --reload --port 8000

# Terminal 2: Frontend locally
cd frontend
npm run dev

# Terminal 3: ChromaDB in Docker
docker-compose up chromadb -d
```

**Best for**: Full-stack features requiring both changes

## Backend Development

### Project Structure

```
backend/
├── main.py                 # FastAPI app entry
├── config.py              # Configuration
├── schemas.py             # Pydantic models
├── deps.py                # Dependencies
├── rag_components.py      # RAG singleton
├── routers/               # HTTP endpoints
│   ├── chat_router.py
│   ├── agents_router.py
│   └── ...
├── services/              # Business logic
│   ├── chat.py
│   ├── nemo.py
│   └── ...
├── guardrails_config/     # NeMo configs
│   ├── aviation_assistant/
│   └── ...
├── pyproject.toml         # Dependencies
├── uv.lock               # Lock file
└── tests/                # Tests
```

### Running Backend

**Development Mode**:
```bash
cd backend
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Options**:
- `--reload`: Auto-restart on file changes
- `--host 0.0.0.0`: Listen on all interfaces
- `--port 8000`: Port number
- `--log-level debug`: Verbose logging

**Production Mode**:
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Adding New Endpoints

**1. Create Router** (`backend/routers/my_router.py`):
```python
from fastapi import APIRouter, Depends
from schemas import MyRequest, MyResponse

router = APIRouter(prefix="/api/my", tags=["My Feature"])

@router.post("/endpoint")
async def my_endpoint(request: MyRequest) -> MyResponse:
    """
    Endpoint description
    """
    # Implementation
    return MyResponse(...)
```

**2. Register Router** (`backend/main.py`):
```python
from routers import my_router

app.include_router(my_router.router)
```

**3. Add Schemas** (`backend/schemas.py`):
```python
from pydantic import BaseModel

class MyRequest(BaseModel):
    field1: str
    field2: int = 0

class MyResponse(BaseModel):
    result: str
```

**4. Test Endpoint**:
```bash
curl -X POST http://localhost:8000/api/my/endpoint \
  -H "Content-Type: application/json" \
  -d '{"field1": "value"}'
```

### Adding New Services

**1. Create Service** (`backend/services/my_service.py`):
```python
from config import Config

class MyService:
    def __init__(self):
        self.config = Config

    async def process(self, data: str) -> str:
        """
        Service logic
        """
        # Implementation
        return result

# Singleton instance
my_service = MyService()
```

**2. Use in Router**:
```python
from services.my_service import my_service

@router.post("/endpoint")
async def endpoint(request: MyRequest):
    result = await my_service.process(request.data)
    return {"result": result}
```

### Working with Dependencies

**Install New Package**:
```bash
uv add package-name

# Development dependency
uv add --dev pytest-asyncio
```

**Update Dependencies**:
```bash
uv sync

# Update specific package
uv add --upgrade package-name
```

**Lock File**:
```bash
# Update lock file
uv lock

# Sync from lock
uv sync
```

## Frontend Development

### Project Structure

```
frontend/
├── src/
│   ├── App.jsx              # Main component
│   ├── main.jsx            # Entry point
│   ├── components/         # React components
│   │   ├── ChatInterface.jsx
│   │   ├── Sidebar.jsx
│   │   └── ...
│   ├── hooks/              # Custom hooks
│   │   ├── useChatSessions.js
│   │   └── useTheme.js
│   └── styles/            # CSS modules
├── public/                # Static assets
├── package.json           # Dependencies
├── vite.config.js        # Vite configuration
└── index.html            # HTML template
```

### Running Frontend

**Development Server**:
```bash
cd frontend
npm run dev
```

**Build for Production**:
```bash
npm run build

# Preview production build
npm run preview
```

**Linting**:
```bash
npm run lint

# Auto-fix
npm run lint:fix
```

### Adding New Components

**1. Create Component** (`frontend/src/components/MyComponent.jsx`):
```jsx
import React, { useState } from 'react';
import './MyComponent.css';

export default function MyComponent({ prop1, prop2 }) {
  const [state, setState] = useState(null);

  const handleAction = () => {
    // Logic
  };

  return (
    <div className="my-component">
      {/* JSX */}
    </div>
  );
}
```

**2. Create Styles** (`frontend/src/components/MyComponent.css`):
```css
.my-component {
  /* Styles */
}
```

**3. Use Component**:
```jsx
import MyComponent from './components/MyComponent';

function App() {
  return (
    <MyComponent prop1="value" prop2={123} />
  );
}
```

### Working with State

**Local State**:
```jsx
const [count, setCount] = useState(0);
const [user, setUser] = useState({ name: '', email: '' });
```

**Custom Hook** (`frontend/src/hooks/useMyHook.js`):
```js
import { useState, useEffect } from 'react';

export function useMyHook(initialValue) {
  const [value, setValue] = useState(initialValue);

  useEffect(() => {
    // Side effects
  }, [value]);

  return { value, setValue };
}
```

**Usage**:
```jsx
import { useMyHook } from './hooks/useMyHook';

function Component() {
  const { value, setValue } = useMyHook('initial');
  // ...
}
```

### API Integration

**Fetch Data**:
```jsx
const [data, setData] = useState(null);
const [loading, setLoading] = useState(false);

const fetchData = async () => {
  setLoading(true);
  try {
    const response = await fetch('http://localhost:8000/api/endpoint');
    const json = await response.json();
    setData(json);
  } catch (error) {
    console.error('Error:', error);
  } finally {
    setLoading(false);
  }
};
```

**SSE Streaming**:
```jsx
const handleStream = (query) => {
  const eventSource = new EventSource(
    `http://localhost:8000/api/chat?query=${encodeURIComponent(query)}`
  );

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.token) {
      // Append token to message
    } else if (data.status === 'done') {
      eventSource.close();
    }
  };

  eventSource.onerror = (error) => {
    console.error('SSE Error:', error);
    eventSource.close();
  };
};
```

## Testing

### Backend Testing

**File**: `backend/tests/`

**Run Tests**:
```bash
cd backend

# All tests
uv run pytest

# Specific file
uv run pytest tests/test_chat.py

# With coverage
uv run pytest --cov=. --cov-report=html

# Verbose
uv run pytest -v
```

**Example Test** (`backend/tests/test_chat.py`):
```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/api/chat", json={
        "query": "test query",
        "model": "gemma3:latest"
    })
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected
```

**Fixtures** (`backend/tests/conftest.py`):
```python
import pytest

@pytest.fixture
def mock_vllm():
    # Setup
    yield mock_client
    # Teardown

@pytest.fixture
def sample_document():
    return {
        "content": "test content",
        "metadata": {"source": "test.pdf"}
    }
```

### Frontend Testing

**Run Tests**:
```bash
cd frontend

# Run tests (if configured)
npm test

# With coverage
npm test -- --coverage
```

**Manual Testing Checklist**:
- [ ] Chat input and submission
- [ ] Message rendering (markdown)
- [ ] Session creation/deletion
- [ ] Model selection
- [ ] Agent selection
- [ ] Document upload
- [ ] Theme toggle
- [ ] RAG toggle
- [ ] Auto-scroll behavior

## Debugging

### Backend Debugging

**1. Print Debugging**:
```python
print(f"Debug: {variable}")
import json
print(json.dumps(data, indent=2))
```

**2. Logging**:
```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")
```

**3. VS Code Debugger**:

Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
      ],
      "cwd": "${workspaceFolder}/backend",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/backend"
      }
    }
  ]
}
```

**4. Interactive Debugging**:
```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use breakpoint() in Python 3.7+
breakpoint()
```

**5. Docker Logs**:
```bash
# Follow logs
docker logs -f backend

# Last 100 lines
docker logs --tail 100 backend

# With timestamps
docker logs -t backend
```

### Frontend Debugging

**1. Console Logging**:
```jsx
console.log('Variable:', variable);
console.table(arrayData);
console.error('Error:', error);
```

**2. React DevTools**:
- Install React DevTools extension
- Inspect component state and props
- Profile performance

**3. Network Tab**:
- Monitor API requests
- Check request/response payloads
- Debug SSE connections

**4. Breakpoints**:
```jsx
// In browser DevTools
debugger;  // Execution pauses here
```

**5. Vite Error Overlay**:
- Automatic on build errors
- Shows stack traces
- Hot reloads on fix

## Code Quality

### Backend Linting

**Tools** (configured in `pyproject.toml`):
- `ruff` - Fast Python linter
- `black` - Code formatter
- `isort` - Import sorter
- `flake8` - Style checker

**Run Linters**:
```bash
cd backend

# Ruff (lint + format)
uv run ruff check .
uv run ruff format .

# Black
uv run black .

# isort
uv run isort .

# flake8
uv run flake8 .
```

**Auto-fix**:
```bash
uv run ruff check --fix .
```

**Pre-commit Hook** (`.git/hooks/pre-commit`):
```bash
#!/bin/bash
cd backend
uv run ruff check . || exit 1
uv run black --check . || exit 1
```

### Frontend Linting

**Tool**: ESLint (configured in `eslintrc.cjs`)

**Run Linter**:
```bash
cd frontend

# Check
npm run lint

# Auto-fix
npm run lint -- --fix
```

**VS Code Integration**:
Install ESLint extension for real-time feedback.

### Type Checking

**Backend** (Pyright):
```bash
cd backend
uv run pyright .
```

**Configuration** (`pyrightconfig.json`):
```json
{
  "typeCheckingMode": "basic",
  "reportMissingImports": true
}
```

## Hot Reload

### Backend Hot Reload

**Uvicorn** with `--reload` flag watches for file changes:

```bash
uv run uvicorn main:app --reload
```

**What triggers reload**:
- `.py` file changes in backend/
- New files added
- File deletions

**What doesn't trigger reload**:
- `config.yml` changes (requires manual restart)
- Dependency changes (requires `uv sync`)

**Docker Volume Mount**:
```yaml
# docker-compose.yml
services:
  backend:
    volumes:
      - ./backend:/app  # Local changes reflect in container
```

### Frontend Hot Reload

**Vite HMR** (Hot Module Replacement):

```bash
npm run dev
```

**Features**:
- Instant updates (< 100ms)
- Preserves component state
- Updates CSS without full reload
- Shows build errors in overlay

**What triggers HMR**:
- `.jsx` file changes
- `.css` file changes
- Asset changes

**Docker Volume Mount**:
```yaml
# docker-compose.yml
services:
  frontend:
    volumes:
      - ./frontend:/app
      - /app/node_modules  # Exclude node_modules
```

## Common Tasks

### Add New Guardrail Agent

```bash
# 1. Create directory
mkdir backend/guardrails_config/my_agent

# 2. Create config.yml
cat > backend/guardrails_config/my_agent/config.yml << 'EOF'
models:
  - type: main
    engine: openai
    model: meta-llama/Llama-3.2-3B-Instruct
    parameters:
      base_url: "http://localhost:8000/v1"
      api_key: "EMPTY"
instructions:
  - type: general
    content: |
      You are an expert in [domain].
EOF

# 3. Create config.co
cat > backend/guardrails_config/my_agent/config.co << 'EOF'
define user express ask question
  "example question"

define bot respond helpfully
  "Example response"
EOF

# 4. Add to metadata
# Edit backend/guardrails_config/metadata.yaml

# 5. Validate
curl http://localhost:8000/api/agents/validate/my_agent

# 6. Test
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "agent_name": "my_agent"}'
```

### Update Dependencies

**Backend**:
```bash
cd backend

# Update package
uv add --upgrade package-name

# Update all
uv sync --upgrade

# Commit changes
git add pyproject.toml uv.lock
git commit -m "Update dependencies"
```

**Frontend**:
```bash
cd frontend

# Update package
npm update package-name

# Update all
npm update

# Check outdated
npm outdated

# Commit changes
git add package.json package-lock.json
git commit -m "Update dependencies"
```

### Reset Development Environment

```bash
# Stop all services
docker-compose down

# Remove volumes
docker-compose down -v

# Clean Docker
docker system prune -a

# Remove dependencies
rm -rf backend/.venv
rm -rf frontend/node_modules

# Reinstall
cd backend && uv sync
cd ../frontend && npm install

# Restart
docker-compose --profile fullstack up -d
```

### Database Operations

**Clear ChromaDB**:
```bash
# Stop services
docker-compose stop chromadb

# Remove volume
docker volume rm nemo_app_chroma_data

# Restart
docker-compose up chromadb -d
```

**Backup ChromaDB**:
```bash
docker run --rm \
  -v nemo_app_chroma_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/chroma_backup.tar.gz /data
```

**Restore ChromaDB**:
```bash
docker run --rm \
  -v nemo_app_chroma_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/chroma_backup.tar.gz -C /
```

### Performance Profiling

**Backend**:
```bash
# Install profiler
uv add --dev py-spy

# Profile running process
uv run py-spy top --pid <pid>

# Generate flame graph
uv run py-spy record -o profile.svg -- python main.py
```

**Frontend**:
```bash
# React Profiler (in code)
import { Profiler } from 'react';

<Profiler id="MyComponent" onRender={onRenderCallback}>
  <MyComponent />
</Profiler>

# Chrome DevTools Performance tab
# 1. Open DevTools
# 2. Performance tab
# 3. Record interaction
# 4. Analyze timeline
```

## Related Documentation

- [Architecture Overview](./ARCHITECTURE.md) - System design
- [API Reference](./API-REFERENCE.md) - API endpoints
- [Deployment Guide](./DEPLOYMENT.md) - Production deployment
- [Backend Guide](./BACKEND-GUIDE.md) - Backend details
- [Frontend Guide](./FRONTEND-GUIDE.md) - Frontend details
- [Troubleshooting](./TROUBLESHOOTING.md) - Common issues
