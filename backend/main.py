# backend/main.py
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database_models import init_database
from rag_components import setup_rag_components
from routers import (
    agents_router,
    automate_router,
    chat_router,
    chunking_router,
    config_router,
    dataset_router,
    document_router,
    health_router,
    model_router,
    upload_router,
)


# --- FastAPI Application Lifespan (for startup and shutdown events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    # Initialize database tables
    init_database()
    # Call setup to initialize global components in rag_components.py
    setup_rag_components()
    yield
    # Shutdown event
    # No explicit cleanup needed here if components don't hold external resources
    # that need closing (like file handles, specific network connections not handled by libraries)


# --- FastAPI App Setup ---
app = FastAPI(
    title="NLP Backend API with RAG",
    description="Backend API for NLP tasks with Retrieval Augmented Generation.",
    version="0.1.0",
    lifespan=lifespan,  # Use the lifespan context manager
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# --- Include Routers ---
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the NLP Backend API with RAG support. See /docs for API documentation."
    }


app.include_router(health_router.router)
app.include_router(agents_router.router)
app.include_router(chat_router.router, prefix="/api", tags=["Chat Endpoints"])
app.include_router(model_router.router, prefix="/api", tags=["Model Endpoints"])
app.include_router(config_router.router, prefix="/api", tags=["Config Endpoints"])
app.include_router(dataset_router.router, prefix="/api", tags=["Dataset Endpoints"])
app.include_router(chunking_router.router, prefix="/api", tags=["Chunking Endpoints"])
app.include_router(document_router.router, prefix="/api", tags=["Document Endpoints"])
app.include_router(upload_router.router, prefix="/api", tags=["Upload Endpoints"])
app.include_router(automate_router.router, prefix="/api", tags=["Automation Endpoints"])


# --- Main Method for Application Testing ---
def test_chat_router(disable_guardrails_for_testing=False):
    from fastapi.testclient import TestClient

    from routers.chat_router import router

    original_use_guardrails = None
    if disable_guardrails_for_testing:
        original_use_guardrails = os.environ.get("USE_GUARDRAILS")
        os.environ["USE_GUARDRAILS"] = "false"

    # Create a test app with the chat router
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api")

    # Test cases with both guardrails and direct modes
    test_cases = [
        {
            "name": "Simple Hello Test",
            "data": {"query": "Hello, who are you?", "history": [], "use_rag": False},
        },
        {
            "name": "Follow-up Question",
            "data": {
                "query": "What's the weather like?",
                "history": [
                    {"sender": "user", "text": "Hello, how are you?"},
                    {
                        "sender": "bot",
                        "text": "Hello! I'm doing well, thank you for asking. How can I help you today?",
                    },
                ],
                "use_rag": False,
            },
        },
        {
            "name": "RAG-enabled Query",
            "data": {
                "query": "Tell me about machine learning",
                "history": [],
                "use_rag": True,
            },
        },
    ]

    # Create test client
    with TestClient(test_app) as client:
        for test_case in test_cases:
            try:
                # Make the request
                response = client.post("/api/chat", json=test_case["data"])

                if response.status_code == 200:
                    content = response.text[:500]

                    # Check if we got a connection error (expected when testing outside Docker)
                    if "ConnectError" in content or "getaddrinfo failed" in content:
                        pass

            except Exception:
                pass

    # Restore original environment variable if it was changed
    if disable_guardrails_for_testing and original_use_guardrails is not None:
        os.environ["USE_GUARDRAILS"] = original_use_guardrails
    elif disable_guardrails_for_testing:
        # Remove the environment variable if it wasn't set originally
        os.environ.pop("USE_GUARDRAILS", None)


# --- Main execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    # # Run the test application
    # from fastapi.testclient import TestClient

    # # Test the chat router (with guardrails disabled for local testing)
    # test_chat_router(disable_guardrails_for_testing=True)

    # # Test the main application endpoints
    # with TestClient(app) as client:
    #     # Test root endpoint
    #     client.get("/")

    #     # Test health endpoint
    #     client.get("/health")

    #     # Test models endpoint
    #     client.get("/api/models")

    # Uncomment below to start the uvicorn server instead
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = False
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload_flag,
        log_level="info",
    )
