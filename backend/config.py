# backend/config.py
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file in the backend directory
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
load_dotenv(env_path)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("nlp_backend")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# --- Environment Variables ---
# PostgreSQL Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "nemo_rag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "nemo_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "nemo_password")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")  # Used as table name

# Build PostgreSQL connection strings
# SQLAlchemy-style URI for PGVector and other SQLAlchemy-based libraries
POSTGRES_CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
# libpq-style connection string for direct psycopg connections
POSTGRES_LIBPQ_CONNECTION = (
    f"host={POSTGRES_HOST} port={POSTGRES_PORT} dbname={POSTGRES_DB} "
    f"user={POSTGRES_USER} password={POSTGRES_PASSWORD}"
)
VLLM_BASE_URL = os.getenv(
    "VLLM_BASE_URL", "http://localhost:8000"
)  # vLLM OpenAI-compatible API endpoint
VLLM_MODEL = os.getenv(
    "VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct"
)  # Model served by vLLM

# HuggingFace Hub Configuration
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", None)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp_uploads")
NEMO_GUARDRAILS_SERVER_URL = os.getenv(
    "NEMO_GUARDRAILS_SERVER_URL", "http://nemo-guardrails:8001"
)
USE_GUARDRAILS = os.getenv("USE_GUARDRAILS", "false").lower() == "true"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure upload directory exists
# Backward compatibility - automation uses the same vLLM model
VLLM_MODEL_FOR_AUTOMATION = os.getenv(
    "VLLM_MODEL_FOR_AUTOMATION", VLLM_MODEL
)  # Defaults to main vLLM model

# --- RAG Configuration ---
# Global flag to enable/disable RAG functionality
# Set to True to enable RAG features, False to disable globally
RAG_ENABLED = os.getenv("RAG_ENABLED", "False").lower() == "true"

# --- Prompt Templates ---
# Template for the basic RAG chain
RAG_PROMPT_TEMPLATE_STR = """SYSTEM: You are a helpful assistant. Use the following context to answer the question. If the context doesn't contain the answer, state that you don't have enough information. Do not make up information.

Context:
{context}

USER: {question}"""

# Template for the RAG chain that provides context as a prefix
RAG_CONTEXT_PREFIX_TEMPLATE_STR = """SYSTEM: You are a helpful AI assistant. Please answer the user's question. Use the following context retrieved from relevant documents to inform your answer. If the context provides a direct answer, prioritize using it. If the context is relevant but doesn't fully answer the question, use it to supplement your knowledge. If the context seems irrelevant or insufficient, answer based on your general knowledge.

Retrieved Context:
---
{context}
---

User Question: {question}

Assistant Answer:"""


# Template for simple chat without RAG
SIMPLE_PROMPT_TEMPLATE_STR = """SYSTEM: You are a helpful AI assistant. Answer the user's question based on your general knowledge.

USER: {question}"""

# Log key configurations on startup
logger.info("Configuration loaded successfully")
