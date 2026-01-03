FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for building Python packages and health checks
# git is required by RAGatouille/ColBERT
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    libc6-dev \
    make \
    pkg-config \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy pyproject.toml first for better Docker layer caching
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv sync --no-editable

COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]