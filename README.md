# NeMo Guardrails Testing Application

A full-stack web application for testing **NVIDIA NeMo Guardrails** with **vLLM** as the LLM provider. This application provides a chat interface with configurable guardrails for different use cases including aviation assistance, banking, and mathematics.

## Table of Contents

- [Overview](#overview)
- [Features](#features)  
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Commands](#development-commands)
- [Manual Setup](#manual-setup)
- [Usage](#usage)
- [Guardrails Configuration](#guardrails-configuration)


## Overview

This application consists of:
- **Backend**: FastAPI server with NeMo Guardrails integration and RAG capabilities
- **Frontend**: React-based chat interface with agent selection
- **Vector Database**: PostgreSQL with pgvector for document storage and retrieval
- **LLM Provider**: vLLM for high-performance language model inference

## Features

- 🛡️ **Guardrails Testing**: Pre-configured guardrails for different domains
- 💬 **Interactive Chat**: Real-time chat interface with streaming responses
- 🤖 **Agent Selection**: Choose between different specialized agents
- 📄 **Document Upload**: PDF upload and processing for RAG
- 🔄 **Session Management**: Persistent chat sessions with local storage
- 🎨 **Modern UI**: Dark/light theme support with responsive design

## Prerequisites

Make sure you have the following installed:

### Required Tools
- **Python 3.9+**
- **[uv](https://docs.astral.sh/uv/)** - Modern Python package manager
- **Node.js 18+** and **npm** - For frontend development
- **Docker** and **Docker Compose** - For containerized services (includes vLLM, PostgreSQL, and ChromaDB)

### vLLM Setup
vLLM runs in a Docker container and is automatically started with Docker Compose. The default model is `meta-llama/Llama-3.2-3B-Instruct`, which will be downloaded automatically on first run.

To use a different model, modify the `MODEL_NAME` environment variable in `docker-compose.yml`.

## Quick Start

### Linux/Ubuntu (Docker Compose)

1. **Create `.env` file** (copy from example or create):
   ```bash
   cat > .env << 'EOF'
   VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
   VLLM_BASE_URL=http://localhost:8002
   POSTGRES_HOST=localhost
   RAG_ENABLED=true
   EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
   # Add HuggingFace token if using Llama models
   # HUGGING_FACE_HUB_TOKEN=hf_your_token_here
   EOF
   ```

2. **Start the full stack with GPU**:
   ```bash
   docker compose --profile fullstack --profile gpu up -d
   ```

   Or for **CPU-only** (slower):
   ```bash
   docker compose --profile fullstack --profile cpu up -d
   ```

3. **Check services are running**:
   ```bash
   docker compose ps
   ```

4. **Access the application**:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - vLLM API: http://localhost:8002

5. **View logs**:
   ```bash
   docker compose logs -f
   ```

6. **Stop everything**:
   ```bash
   docker compose --profile fullstack --profile gpu down
   ```

### Windows (PowerShell)

Start everything with one command:
```powershell
.\dev.ps1 up-fullstack
```

Or step-by-step:
1. **Start dependencies**:
   ```powershell
   .\dev.ps1 up-rag
   ```

2. **Start the API server**:
   ```powershell
   .\dev.ps1 api
   ```

3. **In another terminal, start the frontend**:
   ```powershell
   .\dev.ps1 up-ui
   ```

## Development Commands

Use the provided PowerShell script for common development tasks:

```powershell
# View all available commands
.\dev.ps1 help

# Start services
.\dev.ps1 up-fullstack    # Everything in one command
.\dev.ps1 up-rag          # ChromaDB + dependencies  
.\dev.ps1 up-ui           # Frontend only
.\dev.ps1 api             # FastAPI backend locally

# Management
.\dev.ps1 ps              # Show running services
.\dev.ps1 logs            # View logs
.\dev.ps1 down            # Stop all services
.\dev.ps1 clean           # Remove all containers and volumes

# Testing
.\dev.ps1 test            # Run backend tests
```

## Manual Setup

If you prefer to run services manually:

### Backend Setup
```powershell
cd backend
uv sync                   # Install dependencies
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup
```powershell
cd frontend
npm install               # Install dependencies
npm run dev               # Start development server
```

### Start ChromaDB
```powershell
docker run -p 8001:8000 chromadb/chroma:latest
```

### Cleanup commands
```bash
docker compose --profile fullstack --profile gpu down
```

```bash
  docker network prune -f
```

## Usage

1. **Access the application**: Open http://localhost:5173 in your browser

2. **Select an Agent**: Choose from pre-configured agents:
   - **Aviation Assistant**: Specialized for aviation-related queries
   - **Banking Assistant**: Configured for financial services
   - **Math Assistant**: Focused on mathematical problem-solving

3. **Select a Model**: Choose from your available vLLM models

4. **Start Chatting**: The guardrails will automatically filter and guide responses based on your selected agent

5. **Upload Documents**: Use the PDF upload feature to add context for RAG-based responses

## Model Configuration

This application supports multiple LLM and embedding models. Configure them via the `.env` file in the project root.

### LLM Models (vLLM)

The LLM is served by vLLM and configured via the `VLLM_MODEL` environment variable.

#### Recommended Models

| Model | Size | License | Notes |
|-------|------|---------|-------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Open | No auth required, good quality |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Open | Better quality, still fast |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Open | Excellent quality |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Gated | Requires HF token + license |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Gated | Requires HF token + license |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Gated | Requires HF token + license |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Open | No auth required |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Open | Good reasoning |

#### Switching LLM Models

1. Edit `.env` file:
   ```bash
   # For Qwen (no authentication needed)
   VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct

   # For Llama (requires HuggingFace token)
   VLLM_MODEL=meta-llama/Llama-3.2-1B-Instruct
   HUGGING_FACE_HUB_TOKEN=hf_your_token_here
   ```

2. Restart vLLM:
   ```bash
   docker compose --profile gpu rm -f vllm-gpu && docker compose --profile gpu up -d vllm-gpu
   ```

3. Verify the model is loaded:
   ```bash
   curl http://localhost:8002/v1/models
   ```

#### Using Gated Models (Llama)

For Meta Llama models, you need to:
1. Create a [HuggingFace account](https://huggingface.co/join)
2. Go to the model page (e.g., [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct))
3. Click "Agree and access repository" to accept the license
4. Create an [access token](https://huggingface.co/settings/tokens)
5. Add the token to `.env`:
   ```bash
   HUGGING_FACE_HUB_TOKEN=hf_your_token_here
   ```

### Embedding Models

Embeddings are used for RAG (document retrieval). Configure via `EMBEDDING_MODEL_NAME`.

#### Recommended Embedding Models

| Model | Dimensions | Size | Quality |
|-------|-----------|------|---------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Good (default) |
| `all-MiniLM-L12-v2` | 384 | 120MB | Better |
| `all-mpnet-base-v2` | 768 | 420MB | Best general |
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | Excellent |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Top tier |
| `intfloat/e5-small-v2` | 384 | 130MB | Very good |

#### Switching Embedding Models

1. Edit `.env` file:
   ```bash
   EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
   ```

2. Restart the backend:
   ```bash
   docker compose restart backend
   ```

> **Note:** Changing embedding models requires re-indexing your documents since vector dimensions may differ.

### Example `.env` Configuration

```bash
# LLM Configuration
VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
VLLM_BASE_URL=http://localhost:8002

# Embedding Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# Database
POSTGRES_HOST=localhost

# HuggingFace (for gated models)
HUGGING_FACE_HUB_TOKEN=hf_your_token_here

# RAG Settings
RAG_ENABLED=true
```

---

## Guardrails Configuration

The application includes pre-configured guardrails in `backend/guardrails_config/`:

- `aviation_assistant/config.yml` - Aviation-specific rules and constraints
- `bank_assistant/config.yml` - Banking and financial services guardrails  
- `math_assistant/config.yml` - Mathematical assistance guardrails

Each configuration includes:
- Input/output filtering rules
- Topic constraints
- Response formatting guidelines
- Safety restrictions
