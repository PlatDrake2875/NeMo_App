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

### Option 1: Full Stack (Recommended)
Start everything with one command:
```powershell
.\dev.ps1 up-fullstack
```
This will:
- Start ChromaDB (vector database)
- Start the frontend development server
- Start the FastAPI backend locally

### Option 2: Step-by-Step
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
