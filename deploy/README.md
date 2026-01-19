# NeMo App Deployment Scripts

This directory contains deployment scripts for running the NeMo App with Docker containers, complete with hot reload support for development.

## Quick Start

```bash
# Start with GPU acceleration (recommended)
./deploy/start.sh --gpu

# Start in CPU-only mode
./deploy/start.sh --cpu

# Start with Qdrant vector store instead of pgvector
./deploy/start.sh --gpu --qdrant

# Stop all services
./deploy/stop.sh

# Use the universal wrapper
./deploy/universal.sh start --gpu
./deploy/universal.sh status
./deploy/universal.sh stop
```

## Architecture

| Service       | Port | Description                               |
|---------------|------|-------------------------------------------|
| **Frontend**  | 5173 | React/Vite application with HMR           |
| **Backend**   | 8000 | FastAPI application with hot reload       |
| **vLLM**      | 8002 | LLM inference (GPU or CPU mode)           |
| **PostgreSQL**| 5432 | Database with pgvector extension          |
| **Qdrant**    | 6333 | Vector database (optional, alternative)   |

All services run in Docker containers with configurable profiles (fullstack, gpu, cpu).

## Prerequisites

### Required
- **Docker** - Container runtime (v20.10+)
- **Docker Compose** - Container orchestration (v2.x)

### For GPU Mode
- **NVIDIA GPU** - CUDA-compatible graphics card
- **NVIDIA Container Toolkit** - GPU support in containers

### Installation

#### Install Docker
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin

# Or use Docker Desktop
# https://www.docker.com/products/docker-desktop
```

#### Install NVIDIA Container Toolkit (GPU mode only)
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## GPU vs CPU Mode

### GPU Mode (`--gpu`)
- Uses NVIDIA GPU for LLM inference via vLLM
- Significantly faster inference times
- Requires NVIDIA Container Toolkit
- Default recommendation for production use

### CPU Mode (`--cpu`)
- Runs vLLM in CPU-only mode
- Works on any system without GPU
- Slower inference but functional
- Suitable for development/testing

## Vector Store Backends

### pgvector (default)
- PostgreSQL extension for vector similarity search
- Uses the existing PostgreSQL instance
- No additional service required
- `./deploy/start.sh --gpu --pgvector`

### Qdrant
- Dedicated vector database
- Web dashboard at http://localhost:6333/dashboard
- Better for large-scale deployments
- `./deploy/start.sh --gpu --qdrant`

## Hot Reload

Hot reload is fully configured for rapid development:

### Backend (FastAPI)
- **Method**: Uvicorn `--reload` flag
- **Triggers**: Any `.py` file change in `./backend`
- **Speed**: ~1-2 seconds
- **Volume Mount**: `./backend:/app`

### Frontend (React/Vite)
- **Method**: Vite HMR (Hot Module Replacement)
- **Triggers**: Any `.jsx`, `.tsx`, `.js`, `.css` file change in `./frontend`
- **Speed**: Instant
- **Volume Mount**: `./frontend:/app`

## Service URLs

After starting services:

| Service          | URL                                |
|------------------|------------------------------------|
| Frontend         | http://localhost:5173              |
| Backend API      | http://localhost:8000              |
| API Documentation| http://localhost:8000/docs         |
| vLLM API         | http://localhost:8002/v1/models    |
| Qdrant Dashboard | http://localhost:6333/dashboard    |

## Common Commands

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f vllm-gpu

# Check service status
docker compose --profile fullstack --profile gpu --profile cpu ps

# Force restart with rebuild
docker compose down && ./deploy/start.sh --gpu

# Shell access
docker compose exec backend bash
docker compose exec frontend sh
```

## Troubleshooting

### Port Conflicts
```bash
# Check what's using a port
ss -tuln | grep :8000
# or
lsof -i :8000

# Kill process if needed
kill -9 <PID>
```

### GPU Not Detected
```bash
# Verify NVIDIA driver
nvidia-smi

# Check Docker can see GPU
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Restart Docker if needed
sudo systemctl restart docker
```

### Container Fails to Start
```bash
# Check logs
docker compose logs backend

# Rebuild specific service
docker compose build backend
docker compose up -d backend
```

### vLLM Slow to Start
- First startup downloads model weights (several GB)
- Model loading can take 2-5 minutes
- Check logs: `docker compose logs -f vllm-gpu`

### Hot Reload Not Working
**Backend:**
- Check volume mount: `docker compose exec backend ls -la /app`
- Look for "Uvicorn running with 'reloader'" in logs

**Frontend:**
- Verify `CHOKIDAR_USEPOLLING=true` in docker-compose.yml
- Check for "HMR enabled" in Vite logs

## Files

| File          | Description                                    |
|---------------|------------------------------------------------|
| `start.sh`    | Smart startup with health checks and hot reload|
| `stop.sh`     | Graceful shutdown with cleanup options         |
| `universal.sh`| Unified wrapper for all operations             |
| `README.md`   | This documentation                             |

## Support

For issues or questions:
1. Check logs: `docker compose logs -f`
2. Review service health: `docker compose ps`
3. Run `./deploy/start.sh` to see pre-flight check diagnostics
