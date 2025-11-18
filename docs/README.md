# NeMo Guardrails Testing Application - Documentation

Welcome to the comprehensive documentation for the NeMo Guardrails Testing Application. This application demonstrates NVIDIA NeMo Guardrails integration with local LLM providers (Ollama) through an interactive full-stack web interface.

## Overview

The NeMo Guardrails Testing Application is a full-stack web platform designed for:
- Testing and demonstrating NeMo Guardrails with domain-specific AI assistants
- Experimenting with Retrieval Augmented Generation (RAG) using PDF documents
- Comparing responses with and without guardrails enabled
- Developing and validating custom guardrail configurations

**Primary Use Case**: Development and testing environment for NeMo Guardrails configurations with local LLM deployment.

## Quick Links

### Core Documentation
- **[Architecture Overview](./ARCHITECTURE.md)** - System design, components, and data flow
- **[API Reference](./API-REFERENCE.md)** - Complete endpoint documentation
- **[Guardrails Guide](./GUARDRAILS-GUIDE.md)** - Creating and configuring NeMo agents
- **[RAG System](./RAG-SYSTEM.md)** - Document processing and retrieval system

### Development Guides
- **[Development Guide](./DEVELOPMENT.md)** - Local setup and workflows
- **[Deployment Guide](./DEPLOYMENT.md)** - Docker deployment and configuration
- **[Frontend Guide](./FRONTEND-GUIDE.md)** - React components and architecture
- **[Backend Guide](./BACKEND-GUIDE.md)** - FastAPI services and routers

### Support
- **[Troubleshooting](./TROUBLESHOOTING.md)** - Common issues and solutions

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **NeMo Guardrails** v0.13.0 - NVIDIA's guardrails framework
- **vLLM** - High-performance LLM inference server
- **PostgreSQL with pgvector** - Vector database for RAG
- **LangChain** - LLM orchestration framework

### Frontend
- **React 19** - UI framework
- **Vite 6** - Build tool and dev server
- **react-markdown** - Markdown rendering with GFM support

### Infrastructure
- **Docker & Docker Compose** - Containerization
- **uv** - Modern Python package manager

## Key Features

### Chat Capabilities
- Streaming responses via Server-Sent Events (SSE)
- Session management with localStorage persistence
- Multi-model support (any Ollama model)
- Conversation history with download capability

### Guardrails
- **Domain-specific agents**: Aviation, Banking, Mathematics
- Input/output validation and filtering
- Custom personas and instructions
- Configurable via YAML + Colang

### RAG System
- PDF upload and processing
- Document chunking with metadata
- Vector similarity search
- Toggleable per request

### User Interface
- Dark/light theme support
- Markdown rendering
- Agent selector modal
- Document management viewer
- Auto-scroll with manual override

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Ollama running on port 11434
- Required Ollama model (e.g., `gemma3:latest`)

### Installation

1. Clone the repository and navigate to the project:
```bash
cd /home/ldg/NeMo_App
```

2. Copy environment configuration:
```bash
cp deploy/.env.example deploy/.env
```

3. Start all services using deployment script:
```bash
./deploy/start.sh
```

Or use Docker Compose directly:
```bash
docker-compose --profile fullstack up -d
```

4. Access the application:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Verification

Check service health:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "vllm": {"status": "connected"},
  "postgres": {"status": "connected"}
}
```

## Project Structure

```
/home/ldg/NeMo_App/
├── backend/                    # FastAPI backend
│   ├── routers/               # HTTP endpoint handlers
│   ├── services/              # Business logic layer
│   ├── guardrails_config/     # NeMo agent configurations
│   ├── main.py               # Application entry point
│   └── pyproject.toml        # Python dependencies
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── hooks/            # Custom React hooks
│   │   └── App.jsx          # Main app component
│   └── package.json         # Node dependencies
├── deploy/                   # Deployment scripts and configs
├── docs/                     # Documentation (you are here)
├── docker-compose.yml        # Container orchestration
└── Makefile                 # Development commands
```

## Common Workflows

### Testing Different Agents

1. Start the application
2. Click "Select Agent" button
3. Choose from Aviation, Banking, or Math Assistant
4. Chat with agent-specific responses and guardrails

### Using RAG with Documents

1. Upload a PDF via the upload button
2. Enable "Use RAG" toggle
3. Ask questions about the uploaded document
4. View retrieved context in responses

### Comparing With/Without Guardrails

Guardrails can be toggled in configuration (`backend/config.py`):
- `USE_GUARDRAILS = True` - Routes through NeMo Guardrails
- `USE_GUARDRAILS = False` - Direct Ollama responses

## Development Commands

### Using Makefile
```bash
make up-fullstack    # Start all services
make api             # Run backend locally
make down            # Stop all services
make clean           # Complete cleanup
```

### Using Deployment Scripts
```bash
./deploy/start.sh    # Start with pre-flight checks
./deploy/stop.sh     # Graceful shutdown
```

### Manual Backend Development
```bash
cd backend
uv sync
uv run uvicorn main:app --reload --port 8000
```

### Manual Frontend Development
```bash
cd frontend
npm install
npm run dev
```

## Configuration

### Environment Variables

Key configuration in `backend/config.py`:

```python
VLLM_BASE_URL = "http://localhost:8000"
VLLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "nemo_rag"
USE_GUARDRAILS = False
RAG_ENABLED = False
```

See [Deployment Guide](./DEPLOYMENT.md) for complete configuration reference.

## Available Agents

The application includes three pre-configured guardrail agents:

1. **Aviation Assistant** - Flight operations and aviation domain expert
2. **Banking Assistant** - Financial services and banking specialist
3. **Math Assistant** - Mathematical problem solving expert

Each agent has:
- Custom persona and instructions
- Domain-specific topic constraints
- Input/output validation rules
- Colang conversation flows

See [Guardrails Guide](./GUARDRAILS-GUIDE.md) for creating custom agents.

## API Overview

### Main Endpoints

- `POST /api/chat` - Streaming chat with optional guardrails
- `GET /api/agents/metadata` - List available agents
- `GET /api/models` - List Ollama models
- `POST /api/upload` - Upload PDF for RAG
- `GET /api/documents` - List documents in vector store
- `GET /health` - System health check

Full documentation: [API Reference](./API-REFERENCE.md)

## Architecture Highlights

### Layered Backend Architecture
```
Routers (HTTP) → Services (Logic) → Components (RAG/LLM)
```

### Component-Based Frontend
```
App → Sidebar + ChatInterface → ChatHistory + ChatForm
```

### Data Flow
```
User Query → Router → Service → NeMo Guardrails → Ollama → SSE Stream → UI
```

See [Architecture Overview](./ARCHITECTURE.md) for detailed diagrams.

## Support and Resources

### Documentation
- All documentation files in `/docs/` directory
- Inline API docs at http://localhost:8000/docs

### Health Checks
- Backend: `GET /health`
- PostgreSQL: Check connection via health endpoint
- vLLM: http://localhost:8002/v1/models

### Common Issues
See [Troubleshooting Guide](./TROUBLESHOOTING.md) for solutions to:
- Port conflicts
- vLLM connection issues
- PostgreSQL initialization problems
- Model availability

## Contributing

When modifying the application:

1. Follow the layered architecture pattern
2. Add type hints and validation (Pydantic)
3. Update relevant documentation
4. Test with different agents and models
5. Verify health checks pass

## License

This is a demonstration application for NeMo Guardrails testing and development.

## Next Steps

- **New to the project?** Start with [Architecture Overview](./ARCHITECTURE.md)
- **Setting up locally?** See [Development Guide](./DEVELOPMENT.md)
- **Creating agents?** Read [Guardrails Guide](./GUARDRAILS-GUIDE.md)
- **Deploying?** Check [Deployment Guide](./DEPLOYMENT.md)
- **Having issues?** Consult [Troubleshooting](./TROUBLESHOOTING.md)
