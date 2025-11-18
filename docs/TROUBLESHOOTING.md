# Troubleshooting Guide

Common issues, debugging steps, and solutions for the NeMo Guardrails Testing Application.

## Table of Contents

1. [General Debugging](#general-debugging)
2. [Deployment Issues](#deployment-issues)
3. [Ollama Issues](#ollama-issues)
4. [ChromaDB Issues](#chromadb-issues)
5. [Backend Issues](#backend-issues)
6. [Frontend Issues](#frontend-issues)
7. [NeMo Guardrails Issues](#nemo-guardrails-issues)
8. [RAG Issues](#rag-issues)
9. [Performance Issues](#performance-issues)
10. [Network Issues](#network-issues)

## General Debugging

### Health Check First

Always start with the health endpoint:

```bash
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-12-15T10:30:00Z",
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

### Check Running Services

```bash
# Docker services
docker-compose ps

# Ollama
ps aux | grep ollama

# Ports in use
netstat -tuln | grep -E '5173|8000|8001|11434'
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f chromadb

# Last 100 lines
docker-compose logs --tail 100 backend
```

## Deployment Issues

### Issue: "Port already in use"

**Symptom**:
```
Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use
```

**Solution**:

```bash
# Find process using port
lsof -i :8000
# or
netstat -tuln | grep 8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead of 8000
```

### Issue: "Docker daemon not running"

**Symptom**:
```
Cannot connect to the Docker daemon. Is the docker daemon running?
```

**Solution**:

```bash
# Start Docker
sudo systemctl start docker

# Enable on boot
sudo systemctl enable docker

# Check status
sudo systemctl status docker
```

### Issue: "docker-compose: command not found"

**Symptom**:
```
bash: docker-compose: command not found
```

**Solution**:

```bash
# Install Docker Compose
sudo apt install docker-compose-plugin

# Or use 'docker compose' (newer syntax)
docker compose up -d

# Verify
docker compose version
```

### Issue: Containers immediately exit

**Symptom**:
```
backend    | Exited (1) 2 seconds ago
```

**Debug**:

```bash
# View exit logs
docker-compose logs backend

# Run container interactively
docker-compose run backend /bin/bash

# Check for missing environment variables
docker-compose config
```

## vLLM Issues

### Issue: "Cannot connect to vLLM"

**Symptom**:
```
Failed to connect to vLLM at http://localhost:8000
ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

**Solutions**:

```bash
# 1. Check if vLLM container is running
docker ps | grep vllm

# 2. Start vLLM service
docker-compose up -d vllm

# 3. Check vLLM logs
docker-compose logs vllm

# 4. Test connection
curl http://localhost:8000/v1/models

# 5. Verify environment variable
echo $VLLM_BASE_URL  # Should be http://vllm:8000 in Docker, http://localhost:8000 locally

# 6. Check if port 8000 is available
lsof -i :8000
# If blocked, stop the conflicting service or change vLLM port in docker-compose.yml
```

### Issue: "Model not loaded" or "Model loading failed"

**Symptom**:
```
Error: Model 'meta-llama/Llama-3.2-3B-Instruct' not loaded
ValueError: Model could not be loaded
```

**Solutions**:

```bash
# 1. Check vLLM logs for model download progress
docker-compose logs -f vllm

# 2. Ensure sufficient disk space (models can be 5-20GB+)
df -h

# 3. Check if model name is correct
# View docker-compose.yml MODEL_NAME environment variable
cat docker-compose.yml | grep MODEL_NAME

# 4. Manually pull model (if using HuggingFace)
# vLLM will download on first start - this can take 10-30 minutes
# Check logs to monitor download progress

# 5. For models requiring authentication, add HF_TOKEN
# In docker-compose.yml under vllm service:
environment:
  - HF_TOKEN=your_huggingface_token
```

### Issue: Slow vLLM responses

**Symptom**: Responses take > 30 seconds or timeout

**Solutions**:

```bash
# 1. Check system resources
docker stats vllm
nvidia-smi  # If using GPU

# 2. Use a smaller model
# Edit docker-compose.yml:
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct  # Smaller, faster

# 3. Increase vLLM GPU memory fraction (if using GPU)
# In docker-compose.yml, add to vllm command:
--gpu-memory-utilization 0.9

# 4. Enable tensor parallelism for multi-GPU
# In docker-compose.yml:
--tensor-parallel-size 2  # For 2 GPUs

# 5. Check for network issues between services
docker network inspect nemo_app_default
```

### Issue: vLLM container crashes or OOM

**Symptom**:
```
vllm container exited with code 137
OutOfMemoryError: CUDA out of memory
```

**Solutions**:

```bash
# 1. Use smaller model
# Edit docker-compose.yml MODEL_NAME to:
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct  # or
MODEL_NAME=Qwen/Qwen2-1.5B-Instruct

# 2. Reduce max model length
# In docker-compose.yml, add to vllm command:
--max-model-len 2048  # Instead of default 4096

# 3. Increase Docker memory limit
# In docker-compose.yml under vllm service:
mem_limit: 16g  # Adjust based on available RAM

# 4. Use CPU-only mode (slower but uses less memory)
# Remove GPU devices from docker-compose.yml:
# Comment out:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]

# 5. Restart with clean state
docker-compose down
docker-compose up -d vllm
```

### Issue: vLLM GPU not detected

**Symptom**:
```
WARNING: No GPU found. Running in CPU mode.
```

**Solutions**:

```bash
# 1. Verify NVIDIA Docker runtime is installed
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# 2. Check docker-compose.yml GPU configuration
# Ensure deploy.resources.reservations.devices is properly set

# 3. Install nvidia-container-toolkit if missing
# Ubuntu/Debian:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 4. Verify GPU is available on host
nvidia-smi

# 5. Check Docker daemon.json
cat /etc/docker/daemon.json
# Should include:
# {
#   "runtimes": {
#     "nvidia": {
#       "path": "nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   }
# }
```

## ChromaDB Issues

### Issue: "Cannot connect to ChromaDB"

**Symptom**:
```
Connection refused to ChromaDB at http://localhost:8001
```

**Solutions**:

```bash
# 1. Check if ChromaDB is running
docker-compose ps chromadb

# 2. Start ChromaDB
docker-compose up chromadb -d

# 3. Test connection
curl http://localhost:8001/api/v1/heartbeat

# 4. Check logs
docker-compose logs chromadb

# 5. Restart if needed
docker-compose restart chromadb
```

### Issue: ChromaDB initialization fails

**Symptom**:
```
chromadb    | Error: Permission denied
chromadb    | Exited (1)
```

**Solutions**:

```bash
# 1. Fix volume permissions
docker-compose down
docker volume rm nemo_app_chroma_data
docker-compose up chromadb -d

# 2. Check Docker volume
docker volume ls
docker volume inspect nemo_app_chroma_data

# 3. Verify volume mount in docker-compose.yml
volumes:
  - chroma_data:/chroma/chroma
```

### Issue: ChromaDB data loss

**Symptom**: Uploaded documents disappear after restart

**Solutions**:

```bash
# 1. Verify persistence is enabled
# In docker-compose.yml:
environment:
  - IS_PERSISTENT=TRUE

# 2. Check volume is mounted
docker inspect chromadb | grep -A 5 Mounts

# 3. Backup before restarting
docker run --rm -v nemo_app_chroma_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/chroma_backup.tar.gz /data
```

### Issue: "Collection not found"

**Symptom**:
```
Collection 'documents' not found
```

**Solutions**:

```bash
# 1. Check existing collections
curl http://localhost:8001/api/v1/collections

# 2. Upload a document to auto-create collection
curl -X POST http://localhost:8000/api/upload \
  -F "file=@test.pdf"

# 3. Or manually initialize (in Python):
from rag_components import RAGComponents
rag = RAGComponents()
rag.initialize()
```

## Backend Issues

### Issue: "Module not found"

**Symptom**:
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solutions**:

```bash
# 1. Sync dependencies
cd backend
uv sync

# 2. Verify installation
uv pip list

# 3. Reinstall if needed
uv sync --reinstall

# 4. Check Python version
python --version  # Should be 3.9+
```

### Issue: "Pydantic validation error"

**Symptom**:
```
ValidationError: 1 validation error for ChatRequest
  query
    field required (type=value_error.missing)
```

**Solutions**:

```bash
# 1. Check request body
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'  # Missing fields?

# 2. Verify schema in schemas.py
# Required fields must have values

# 3. Check API documentation
# http://localhost:8000/docs
```

### Issue: Backend won't start

**Symptom**:
```
Error: Application failed to start
```

**Debug**:

```bash
# 1. Check logs
docker-compose logs backend

# 2. Run manually to see errors
cd backend
uv run uvicorn main:app --reload

# 3. Test imports
cd backend
uv run python -c "import fastapi; print('OK')"

# 4. Validate config
uv run python -c "from config import Config; Config.validate()"
```

### Issue: SSE streaming not working

**Symptom**: No tokens received in frontend

**Solutions**:

```bash
# 1. Test SSE endpoint directly
curl -N http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# 2. Check CORS headers
# In main.py, ensure:
app.add_middleware(CORSMiddleware, ...)

# 3. Verify Content-Type
# Should be: text/event-stream

# 4. Check nginx config (if using)
# Disable buffering for SSE:
proxy_buffering off;
```

## Frontend Issues

### Issue: "Cannot connect to backend"

**Symptom**:
```
Failed to fetch
Network error
```

**Solutions**:

```bash
# 1. Check backend is running
curl http://localhost:8000/health

# 2. Verify API URL in frontend
# In .env or vite.config.js:
VITE_API_URL=http://localhost:8000

# 3. Check CORS
# Backend should allow frontend origin

# 4. Check browser console for errors
# Open DevTools → Console
```

### Issue: "npm install fails"

**Symptom**:
```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE could not resolve
```

**Solutions**:

```bash
# 1. Clear cache
npm cache clean --force

# 2. Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# 3. Reinstall
npm install

# 4. Use --legacy-peer-deps if needed
npm install --legacy-peer-deps
```

### Issue: Blank screen / white screen

**Symptom**: Frontend loads but shows nothing

**Debug**:

```bash
# 1. Check browser console
# Open DevTools → Console
# Look for JavaScript errors

# 2. Check network tab
# Verify API requests succeed

# 3. Test production build locally
npm run build
npm run preview

# 4. Check React errors
# Look for React error overlay
```

### Issue: Hot reload not working

**Symptom**: Changes don't appear

**Solutions**:

```bash
# 1. Restart dev server
# Ctrl+C, then npm run dev

# 2. Check volume mount (Docker)
# In docker-compose.yml:
volumes:
  - ./frontend:/app
  - /app/node_modules  # Important!

# 3. Clear Vite cache
rm -rf frontend/node_modules/.vite

# 4. Hard refresh browser
# Ctrl+Shift+R (Linux/Windows)
# Cmd+Shift+R (Mac)
```

## NeMo Guardrails Issues

### Issue: "Agent configuration not found"

**Symptom**:
```
Failed to load agent configuration for 'aviation_assistant'
```

**Solutions**:

```bash
# 1. Verify files exist
ls backend/guardrails_config/aviation_assistant/
# Should show: config.yml, config.co

# 2. Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('backend/guardrails_config/aviation_assistant/config.yml'))"

# 3. Check file permissions
chmod 644 backend/guardrails_config/aviation_assistant/*

# 4. Validate agent
curl http://localhost:8000/api/agents/validate/aviation_assistant
```

### Issue: Guardrails not filtering

**Symptom**: Agent responds to off-topic queries

**Solutions**:

**1. Lower threshold**:
```yaml
# In config.yml
self_check:
  input:
    threshold: 0.6  # Was 0.8
```

**2. Add explicit patterns**:
```colang
# In config.co
define user express off topic query
  "banking"
  "stocks"
  "medical advice"
```

**3. Check USE_GUARDRAILS flag**:
```bash
# In .env or backend config
USE_GUARDRAILS=true
```

### Issue: NeMo initialization slow

**Symptom**: First request takes 30+ seconds

**Explanation**: NeMo loads model on first use (lazy loading)

**Solutions**:

```python
# Preload in startup event (main.py)
@app.on_event("startup")
async def startup():
    from services.nemo import nemo_service
    # Preload all agents
    for agent in ["aviation_assistant", "bank_assistant"]:
        nemo_service.get_or_create_rails(agent)
```

## RAG Issues

### Issue: No relevant documents retrieved

**Symptom**: RAG returns empty context

**Debug**:

```bash
# 1. Check documents exist
curl http://localhost:8000/api/documents

# 2. Test query directly
# In Python:
from rag_components import RAGComponents
rag = RAGComponents()
docs = rag.retriever.get_relevant_documents("query")
print(len(docs))

# 3. Check similarity scores
results = rag.vector_store.similarity_search_with_score("query", k=5)
for doc, score in results:
    print(f"Score: {score}")
```

**Solutions**:

```python
# 1. Lower threshold
RAG_SCORE_THRESHOLD=0.3  # Was 0.5

# 2. Increase top-k
RAG_TOP_K=5  # Was 3

# 3. Rephrase query to match document language
```

### Issue: PDF upload fails

**Symptom**:
```
Error: Failed to process PDF
```

**Solutions**:

```bash
# 1. Check file size
ls -lh document.pdf
# Ensure < 10MB (default limit)

# 2. Test upload manually
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"

# 3. Check backend logs
docker-compose logs backend | grep -i upload

# 4. Verify PDF is readable
pdfinfo document.pdf  # Install: apt install poppler-utils
```

### Issue: Embedding generation slow

**Symptom**: Upload takes minutes

**Solutions**:

```bash
# 1. Use GPU for embeddings
# In rag_components.py:
model_kwargs={'device': 'cuda'}

# 2. Reduce chunk size
CHUNK_SIZE=500  # Was 1000

# 3. Process in batches
# Already implemented, check batch size
```

### Issue: Out of memory during embedding

**Symptom**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:

```python
# 1. Use CPU instead of GPU
model_kwargs={'device': 'cpu'}

# 2. Process smaller batches
batch_size = 10  # Reduce from default

# 3. Use smaller embedding model
EMBEDDING_MODEL_NAME="sentence-transformers/paraphrase-MiniLM-L3-v2"
```

## Performance Issues

### Issue: Slow response times

**Symptom**: Requests take > 10 seconds

**Debug**:

```bash
# 1. Identify bottleneck
time curl http://localhost:8000/api/models

# 2. Check system resources
top
htop
free -h

# 3. Profile backend
# Add timing logs
import time
start = time.time()
# ... operation
print(f"Took {time.time() - start:.2f}s")
```

**Solutions**:

```python
# 1. Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=100)
def get_models():
    return expensive_operation()

# 2. Use async/await properly
# Don't block the event loop

# 3. Reduce model size
# Use gemma3:4b instead of gemma3:13b

# 4. Increase workers
# uvicorn main:app --workers 4
```

### Issue: High memory usage

**Symptom**: System uses > 8GB RAM

**Solutions**:

```bash
# 1. Check memory usage
docker stats

# 2. Limit Docker memory
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 4G

# 3. Use smaller models
# 4B models instead of 13B+

# 4. Restart services periodically
docker-compose restart backend
```

### Issue: CPU at 100%

**Symptom**: System unresponsive

**Solutions**:

```bash
# 1. Identify process
top
# Look for ollama, python

# 2. Limit concurrent requests
# In backend service:
from asyncio import Semaphore
sem = Semaphore(2)  # Max 2 concurrent

# 3. Use GPU for inference
ollama serve --gpu

# 4. Reduce worker processes
uvicorn main:app --workers 2  # Instead of 4
```

## Network Issues

### Issue: CORS errors

**Symptom**:
```
Access to fetch at 'http://localhost:8000/api/chat' from origin 'http://localhost:5173' has been blocked by CORS policy
```

**Solutions**:

```python
# In backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: "Connection refused"

**Symptom**:
```
Connection refused: http://localhost:8000
```

**Debug**:

```bash
# 1. Check service is running
curl http://localhost:8000/health

# 2. Check port is open
netstat -tuln | grep 8000

# 3. Check firewall
sudo ufw status
sudo ufw allow 8000/tcp

# 4. Check Docker networking
docker network ls
docker network inspect nemo_app_default
```

### Issue: Cannot access from other machines

**Symptom**: Works on localhost, not from network

**Solutions**:

```bash
# 1. Bind to 0.0.0.0 instead of 127.0.0.1
# In uvicorn command:
uvicorn main:app --host 0.0.0.0

# 2. Check firewall
sudo ufw allow 8000/tcp

# 3. Update CORS origins
allow_origins=["http://192.168.1.100:5173"]

# 4. Check Docker port mapping
docker-compose ps
# Verify: 0.0.0.0:8000->8000/tcp
```

## Getting Help

### Enable Debug Logging

```python
# backend/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

```bash
# In .env
LOG_LEVEL=debug
```

### Collect Diagnostic Information

```bash
#!/bin/bash
# Create diagnostic report

echo "=== System Info ===" > diagnostic.txt
uname -a >> diagnostic.txt
docker --version >> diagnostic.txt
docker-compose --version >> diagnostic.txt

echo "\n=== Running Services ===" >> diagnostic.txt
docker-compose ps >> diagnostic.txt

echo "\n=== Health Check ===" >> diagnostic.txt
curl http://localhost:8000/health >> diagnostic.txt 2>&1

echo "\n=== Backend Logs ===" >> diagnostic.txt
docker-compose logs --tail 50 backend >> diagnostic.txt

echo "\n=== ChromaDB Logs ===" >> diagnostic.txt
docker-compose logs --tail 50 chromadb >> diagnostic.txt

echo "\n=== Disk Usage ===" >> diagnostic.txt
df -h >> diagnostic.txt

echo "\n=== Memory Usage ===" >> diagnostic.txt
free -h >> diagnostic.txt

echo "Report saved to diagnostic.txt"
```

### Useful Commands Reference

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/api/v1/heartbeat
ollama list

# View logs
docker-compose logs -f backend
docker-compose logs --tail 100 chromadb

# Restart services
docker-compose restart backend
docker-compose restart chromadb

# Complete reset
docker-compose down -v
docker-compose up -d

# Check resources
docker stats
free -h
df -h
```

## Related Documentation

- [Development Guide](./DEVELOPMENT.md) - Local development setup
- [Deployment Guide](./DEPLOYMENT.md) - Deployment issues
- [Architecture Overview](./ARCHITECTURE.md) - System design
- [API Reference](./API-REFERENCE.md) - API details
- [RAG System](./RAG-SYSTEM.md) - RAG troubleshooting
- [Guardrails Guide](./GUARDRAILS-GUIDE.md) - NeMo troubleshooting
