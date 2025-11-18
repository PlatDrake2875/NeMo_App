# Deployment Guide

Complete guide for deploying the NeMo Guardrails Testing Application in various environments.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Configuration](#configuration)
4. [Docker Compose Deployment](#docker-compose-deployment)
5. [Manual Deployment](#manual-deployment)
6. [Production Considerations](#production-considerations)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Backup and Recovery](#backup-and-recovery)
9. [Scaling](#scaling)
10. [Security](#security)

## Deployment Overview

### Deployment Options

| Method | Complexity | Use Case |
|--------|-----------|----------|
| Docker Compose | Low | Development, small deployments |
| Manual | Medium | Custom environments, debugging |
| Kubernetes | High | Production, enterprise scale |
| Cloud Services | Medium | Managed infrastructure |

### Architecture

```
┌─────────────────────────────────────────┐
│         Load Balancer (Optional)        │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Frontend │ │ Frontend │ │ Frontend │
│  :5173   │ │  :5173   │ │  :5173   │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     └────────────┼────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │     Backend     │
        │     :8000       │
        └────────┬────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
┌────────────┐ ┌────────┐ ┌───────────┐
│  ChromaDB  │ │ Ollama │ │   NeMo    │
│   :8001    │ │ :11434 │ │Guardrails │
└────────────┘ └────────┘ └───────────┘
```

## Prerequisites

### Software Requirements

| Component | Version | Required |
|-----------|---------|----------|
| Docker | 20.10+ | Yes |
| Docker Compose | 2.0+ | Yes |
| Ollama | Latest | Yes |
| Linux/macOS/Windows | Any | Yes |

### Hardware Requirements

**Minimum**:
- 8GB RAM
- 4 CPU cores
- 20GB disk space
- Network connectivity

**Recommended**:
- 16GB+ RAM
- 8+ CPU cores
- 50GB+ SSD storage
- NVIDIA GPU (optional, for faster inference)

### Network Requirements

**Ports**:
- 5173: Frontend (configurable)
- 8000: Backend API (configurable)
- 8001: ChromaDB (internal, can be private)
- 11434: Ollama (host machine)

**Firewall Rules**:
```bash
# Allow frontend
sudo ufw allow 5173/tcp

# Allow backend
sudo ufw allow 8000/tcp

# Ollama (if remote)
sudo ufw allow 11434/tcp
```

## Configuration

### Environment Variables

**File**: `deploy/.env`

**Complete Configuration**:
```bash
# ============================================
# Application Configuration
# ============================================

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_FOR_RAG=gemma3:latest

# ChromaDB Settings
CHROMA_HOST=chromadb  # Use "localhost" for local dev
CHROMA_PORT=8001

# Feature Flags
USE_GUARDRAILS=true
RAG_ENABLED=true

# Embedding Model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# RAG Configuration
RAG_TOP_K=3
RAG_SCORE_THRESHOLD=0.5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Upload Settings
UPLOAD_DIR=/tmp/uploads
MAX_UPLOAD_SIZE=10485760  # 10MB

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:5173

# Backend URL
BACKEND_URL=http://localhost:8000

# ============================================
# Docker Configuration
# ============================================

# Container Names
COMPOSE_PROJECT_NAME=nemo_app

# Restart Policy
RESTART_POLICY=unless-stopped

# ============================================
# Logging
# ============================================

LOG_LEVEL=info  # debug, info, warning, error
```

### Docker Compose Configuration

**File**: `docker-compose.yml`

**Key Sections**:

```yaml
version: '3.8'

services:
  # Backend Service
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL}
      - CHROMA_HOST=chromadb
      - USE_GUARDRAILS=${USE_GUARDRAILS}
      - RAG_ENABLED=${RAG_ENABLED}
    volumes:
      - ./backend:/app  # Hot reload in dev
    depends_on:
      chromadb:
        condition: service_healthy
    restart: ${RESTART_POLICY:-unless-stopped}
    profiles:
      - fullstack

  # Frontend Service
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app  # Hot reload in dev
      - /app/node_modules  # Exclude node_modules
    depends_on:
      - backend
    restart: ${RESTART_POLICY:-unless-stopped}
    profiles:
      - fullstack

  # ChromaDB Service
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"  # ChromaDB uses port 8000 internally
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: ${RESTART_POLICY:-unless-stopped}

volumes:
  chroma_data:
    driver: local
```

## Docker Compose Deployment

### Quick Start

**1. Copy Environment File**:
```bash
cp deploy/.env.example deploy/.env
```

**2. Edit Configuration**:
```bash
nano deploy/.env
# Adjust settings as needed
```

**3. Ensure Ollama is Running**:
```bash
# Start Ollama
ollama serve

# Pull required model
ollama pull gemma3:latest

# Verify
ollama list
```

**4. Deploy with Script**:
```bash
./deploy/start.sh
```

Or manually:
```bash
docker-compose --profile fullstack up -d
```

**5. Verify Deployment**:
```bash
# Check services
docker-compose ps

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f
```

**6. Access Application**:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Deployment Script

**File**: `deploy/start.sh`

**Features**:
- Pre-flight checks (Docker, Ollama, ports)
- Service health verification
- Graceful error handling
- Color-coded output

**Usage**:
```bash
# Make executable
chmod +x deploy/start.sh

# Run
./deploy/start.sh

# Stop
./deploy/stop.sh
```

### Custom Deployment

**Development Mode**:
```bash
# With hot reload
docker-compose --profile fullstack up

# View logs in real-time
docker-compose logs -f backend frontend
```

**Production Mode**:
```bash
# Build images
docker-compose build --no-cache

# Deploy in background
docker-compose --profile fullstack up -d

# Verify
docker-compose ps
```

**Selective Services**:
```bash
# Only ChromaDB
docker-compose up chromadb -d

# Backend + ChromaDB
docker-compose up backend chromadb -d

# All services
docker-compose --profile fullstack up -d
```

## Manual Deployment

### Backend Deployment

**1. Install Dependencies**:
```bash
cd backend
uv sync --no-dev  # Production dependencies only
```

**2. Configure Environment**:
```bash
export OLLAMA_BASE_URL=http://localhost:11434
export CHROMA_HOST=localhost
export CHROMA_PORT=8001
export USE_GUARDRAILS=true
export RAG_ENABLED=true
```

**3. Start Backend**:
```bash
uv run uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

**Production with Gunicorn**:
```bash
uv run gunicorn main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Frontend Deployment

**1. Build Production Bundle**:
```bash
cd frontend
npm install
npm run build
```

**2. Serve with Nginx**:

**Install Nginx**:
```bash
sudo apt install nginx
```

**Configure** (`/etc/nginx/sites-available/nemo-app`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /var/www/nemo-app/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # API Proxy
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # SSE Support
    location /api/chat {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        chunked_transfer_encoding on;
    }
}
```

**Enable Site**:
```bash
sudo ln -s /etc/nginx/sites-available/nemo-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**3. Serve with Node.js** (alternative):
```bash
npm install -g serve
serve -s dist -l 5173
```

### ChromaDB Deployment

**Docker** (recommended):
```bash
docker run -d \
  --name chromadb \
  -p 8001:8000 \
  -v chroma_data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  chromadb/chroma:latest
```

**Manual Install**:
```bash
pip install chromadb
chroma run --host 0.0.0.0 --port 8001 --path /data/chroma
```

### Systemd Services

**Backend Service** (`/etc/systemd/system/nemo-backend.service`):
```ini
[Unit]
Description=NeMo Guardrails Backend
After=network.target chromadb.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/nemo-app/backend
Environment="PATH=/opt/nemo-app/backend/.venv/bin"
ExecStart=/opt/nemo-app/backend/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**Enable and Start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable nemo-backend
sudo systemctl start nemo-backend
sudo systemctl status nemo-backend
```

## Production Considerations

### Environment-Specific Settings

**Development**:
```bash
USE_GUARDRAILS=true
RAG_ENABLED=true
LOG_LEVEL=debug
CORS_ORIGINS=*
```

**Staging**:
```bash
USE_GUARDRAILS=true
RAG_ENABLED=true
LOG_LEVEL=info
CORS_ORIGINS=https://staging.example.com
```

**Production**:
```bash
USE_GUARDRAILS=true
RAG_ENABLED=true
LOG_LEVEL=warning
CORS_ORIGINS=https://example.com
RATE_LIMIT_ENABLED=true
API_KEY_REQUIRED=true
```

### Performance Tuning

**Backend Workers**:
```python
# Calculate optimal workers
workers = (2 * cpu_cores) + 1

# For 8 cores
workers = 17
```

**Uvicorn Configuration**:
```bash
uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --limit-concurrency 100 \
  --timeout-keep-alive 5 \
  --loop uvloop  # Faster event loop
```

**ChromaDB Optimization**:
```python
# Increase batch size for better throughput
collection.add(
    documents=docs,
    batch_size=100  # Default: 41666
)
```

### Caching

**Response Caching**:
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# Initialize Redis cache
FastAPICache.init(RedisBackend(...))

# Cache endpoint
@router.get("/models")
@cache(expire=300)  # 5 minutes
async def get_models():
    return ollama_service.list_models()
```

**Static Asset Caching** (Nginx):
```nginx
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

### HTTPS/SSL

**Let's Encrypt with Certbot**:
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (already configured)
sudo certbot renew --dry-run
```

**Nginx HTTPS Configuration**:
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # ... rest of configuration
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

## Monitoring and Logging

### Health Checks

**Kubernetes Liveness Probe**:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

**Docker Healthcheck**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Logging

**Structured Logging** (`backend/main.py`):
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        return json.dumps(log_obj)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)
```

**Log Aggregation**:
```bash
# Send to Loki (or ELK, Splunk)
docker-compose logs -f | promtail --config.file=promtail.yml
```

**Access Logs**:
```bash
# Backend access logs
tail -f /var/log/nemo-backend/access.log

# Nginx access logs
tail -f /var/log/nginx/access.log
```

### Monitoring Tools

**Prometheus Metrics**:
```python
from prometheus_client import Counter, Histogram

request_count = Counter('http_requests_total', 'Total HTTP requests')
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
```

**Grafana Dashboard**:
- Request rate
- Response times
- Error rates
- ChromaDB operations
- Ollama inference times

## Backup and Recovery

### Database Backup

**ChromaDB Backup**:
```bash
# Export ChromaDB data
docker run --rm \
  -v nemo_app_chroma_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/chroma-$(date +%Y%m%d).tar.gz /data

# Schedule with cron
0 2 * * * /path/to/backup-chromadb.sh
```

**Restore ChromaDB**:
```bash
# Stop ChromaDB
docker-compose stop chromadb

# Restore from backup
docker run --rm \
  -v nemo_app_chroma_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/chroma-20241215.tar.gz -C /

# Restart
docker-compose up chromadb -d
```

### Configuration Backup

```bash
# Backup configs
tar czf config-backup-$(date +%Y%m%d).tar.gz \
  deploy/.env \
  backend/guardrails_config/ \
  docker-compose.yml

# Store in S3/remote storage
aws s3 cp config-backup-*.tar.gz s3://backups/nemo-app/
```

### Disaster Recovery

**Recovery Steps**:

1. **Provision New Server**
2. **Install Prerequisites**
3. **Restore Configurations**
   ```bash
   tar xzf config-backup.tar.gz
   ```
4. **Restore Data**
   ```bash
   # Restore ChromaDB
   docker run --rm -v nemo_app_chroma_data:/data -v $(pwd):/backup \
     alpine tar xzf /backup/chroma-backup.tar.gz -C /
   ```
5. **Deploy Application**
   ```bash
   docker-compose --profile fullstack up -d
   ```
6. **Verify**
   ```bash
   curl http://localhost:8000/health
   ```

**RTO (Recovery Time Objective)**: < 1 hour
**RPO (Recovery Point Objective)**: Last nightly backup (24 hours)

## Scaling

### Horizontal Scaling

**Load Balancer Configuration** (Nginx):
```nginx
upstream backend {
    least_conn;  # Load balancing method
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    location /api {
        proxy_pass http://backend;
    }
}
```

**Docker Swarm**:
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml nemo-app

# Scale backend
docker service scale nemo-app_backend=3
```

**Kubernetes**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    spec:
      containers:
      - name: backend
        image: nemo-backend:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Vertical Scaling

**Increase Resources**:
```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Ollama Scaling

**Multiple Ollama Instances**:
```bash
# Instance 1
ollama serve --host 0.0.0.0:11434

# Instance 2
ollama serve --host 0.0.0.0:11435

# Load balance in backend
OLLAMA_URLS=http://ollama1:11434,http://ollama2:11435
```

## Security

### API Security

**Rate Limiting**:
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: ChatRequest):
    ...
```

**API Keys**:
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(403, "Invalid API key")
    return api_key

@app.post("/api/chat")
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    ...
```

### CORS Configuration

**Production CORS**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://example.com",
        "https://app.example.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
```

### Input Validation

**Already Implemented** via Pydantic schemas
**Additional Sanitization**:
```python
import bleach

def sanitize_input(text: str) -> str:
    return bleach.clean(text, strip=True)
```

### Secrets Management

**Environment Variables**:
```bash
# Never commit secrets to git
echo ".env" >> .gitignore

# Use secret management
export API_KEY=$(vault kv get -field=api_key secret/nemo-app)
```

**Docker Secrets**:
```yaml
services:
  backend:
    secrets:
      - api_key
    environment:
      - API_KEY_FILE=/run/secrets/api_key

secrets:
  api_key:
    external: true
```

## Related Documentation

- [Architecture Overview](./ARCHITECTURE.md) - System architecture
- [Development Guide](./DEVELOPMENT.md) - Local development
- [API Reference](./API-REFERENCE.md) - API documentation
- [Troubleshooting](./TROUBLESHOOTING.md) - Common deployment issues
