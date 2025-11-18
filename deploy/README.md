# NeMo App Deployment Scripts

This directory contains deployment scripts for running the NeMo App with Docker containers, complete with hot reload support for development.

## Quick Start

```bash
# Start all services
./deploy/start.sh

# Stop all services
./deploy/stop.sh
```

## Architecture

The deployment includes three main services:

1. **ChromaDB** - Vector database for RAG features (port 8001)
2. **Backend** - FastAPI application with hot reload (port 8000)
3. **Frontend** - React/Vite application with HMR (port 5173)

All services run in Docker containers with the `fullstack` profile enabled.

## Prerequisites

### Required
- **Docker** - Container runtime
- **Docker Compose** - Container orchestration
- **Ollama** - LLM inference engine (running on host)

### Optional
- Ollama model: `gemma3:latest` (will prompt to install if missing)

### Installation

#### Install Docker
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Or use Docker Desktop
# https://www.docker.com/products/docker-desktop
```

#### Install Ollama
```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Mac
brew install ollama

# Start Ollama
ollama serve

# Pull required model
ollama pull gemma3:latest
```

## Usage

### Starting Services

```bash
./deploy/start.sh
```

This script will:
1. ✓ Check prerequisites (Docker, Ollama, required models)
2. ✓ Verify ports are available (5173, 8000, 8001)
3. ✓ Stop any existing containers
4. ✓ Build Docker images
5. ✓ Start services in order with health checks
6. ✓ Display service status and URLs

### Stopping Services

```bash
./deploy/stop.sh
```

Options:
1. Keep data volumes (default) - Preserves ChromaDB data
2. Remove data volumes - Deletes all ChromaDB data
3. Remove volumes and images - Complete cleanup

### Viewing Logs

```bash
# View all logs
docker-compose -f docker-compose.yml --profile fullstack logs -f

# View specific service
docker-compose -f docker-compose.yml --profile fullstack logs -f backend
docker-compose -f docker-compose.yml --profile fullstack logs -f frontend
docker-compose -f docker-compose.yml --profile fullstack logs -f chromadb
```

### Restarting Services

```bash
# Quick restart
./deploy/stop.sh && ./deploy/start.sh

# Rebuild and restart
docker-compose -f docker-compose.yml --profile fullstack down
docker-compose -f docker-compose.yml --profile fullstack up --build -d
```

## Hot Reload

Hot reload is fully configured for rapid development:

### Backend (FastAPI)
- **Method**: Uvicorn `--reload` flag
- **Triggers**: Any `.py` file change in `./backend`
- **Speed**: ~1-2 seconds
- **Volume Mount**: `./backend:/app` (excludes `.venv`)

### Frontend (React/Vite)
- **Method**: Vite HMR (Hot Module Replacement)
- **Triggers**: Any `.jsx`, `.js`, `.css` file change in `./frontend`
- **Speed**: Instant
- **Volume Mount**: `./frontend:/app` (excludes `node_modules`)

### Testing Hot Reload

1. Start services: `./deploy/start.sh`
2. Edit a backend file: `backend/main.py`
3. Edit a frontend file: `frontend/src/App.jsx`
4. Check logs to see reload triggered
5. Refresh browser to see changes

## Service URLs

After starting services:

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ChromaDB**: http://localhost:8001
- **Ollama**: http://localhost:11434

## Environment Variables

Copy `.env.example` to `.env` to customize configuration:

```bash
cp deploy/.env.example .env
```

Key variables:
- `OLLAMA_BASE_URL` - Ollama service URL
- `OLLAMA_MODEL_FOR_RAG` - LLM model name
- `CHROMA_HOST` - ChromaDB hostname
- `RAG_ENABLED` - Enable/disable RAG features
- `USE_GUARDRAILS` - Enable/disable NeMo Guardrails

## Troubleshooting

### Port Conflicts

If ports are in use:
```bash
# Check what's using a port
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Ollama Not Found

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Verify model
ollama list
```

### Container Fails to Start

```bash
# Check logs
docker-compose -f docker-compose.yml --profile fullstack logs backend

# Restart with rebuild
docker-compose -f docker-compose.yml --profile fullstack up --build -d backend
```

### Hot Reload Not Working

**Backend:**
- Check volume mount: `docker-compose -f docker-compose.yml --profile fullstack exec backend ls -la /app`
- Check uvicorn logs: Look for "Uvicorn running with 'reloader'"

**Frontend:**
- Verify `CHOKIDAR_USEPOLLING=true` in docker-compose.yml
- Check Vite logs: Look for "HMR enabled"

### Permission Issues

```bash
# Fix file permissions
sudo chown -R $USER:$USER ./backend ./frontend

# Reset Docker permissions
docker-compose -f docker-compose.yml --profile fullstack down -v
```

## Development Workflow

1. **Start services**: `./deploy/start.sh`
2. **Make changes**: Edit files in `./backend` or `./frontend`
3. **See changes**: Services reload automatically
4. **View logs**: `docker-compose logs -f` to monitor
5. **Stop services**: `./deploy/stop.sh` when done

## Advanced Usage

### Shell Access

```bash
# Backend shell
docker-compose -f docker-compose.yml --profile fullstack exec backend bash

# Frontend shell
docker-compose -f docker-compose.yml --profile fullstack exec frontend sh
```

### Database Access

```bash
# ChromaDB API
curl http://localhost:8001/api/v1/heartbeat

# List collections
curl http://localhost:8001/api/v1/collections
```

### Manual Service Control

```bash
# Start specific service
docker-compose -f docker-compose.yml --profile fullstack up -d backend

# Stop specific service
docker-compose -f docker-compose.yml --profile fullstack stop frontend

# Restart service
docker-compose -f docker-compose.yml --profile fullstack restart backend
```

## Production Deployment

**Note**: These scripts are optimized for development with hot reload.

For production deployment:
1. Disable hot reload
2. Build optimized images
3. Use production environment variables
4. Configure proper networking
5. Set up reverse proxy (nginx)
6. Enable HTTPS
7. Configure logging and monitoring

## Files

- `start.sh` - Main deployment script with pre-flight checks
- `stop.sh` - Graceful shutdown script
- `.env.example` - Environment variable template
- `README.md` - This documentation

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Review health checks: `docker-compose ps`
3. Verify prerequisites: Run `./deploy/start.sh` pre-flight checks
4. Check GitHub issues: [NeMo App Repository]

## License

See main project LICENSE file.
