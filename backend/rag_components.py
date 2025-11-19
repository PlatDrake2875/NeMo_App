# backend/rag_components.py
from typing import Any, Optional

from fastapi import (
    HTTPException,  # Ensure Depends is imported if used directly here, though typically in routers
)
from langchain_core.documents import Document
from langchain_core.messages import (  # Removed SystemMessage as it wasn't used
    AIMessage,
    HumanMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,  # Removed MessagesPlaceholder as it wasn't used directly
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import (
    POSTGRES_LIBPQ_CONNECTION,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RAG_CONTEXT_PREFIX_TEMPLATE_STR,
    RAG_ENABLED,
    RAG_PROMPT_TEMPLATE_STR,
    SIMPLE_PROMPT_TEMPLATE_STR,
    VECTOR_STORE_BACKEND,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_MODEL_FOR_AUTOMATION,
    logger,
)
from vectorstore_factory import create_vectorstore


class RAGComponents:
    """Singleton class to manage RAG components without global variables."""

    _instance: Optional["RAGComponents"] = None
    _initialized: bool = False

    def __new__(cls) -> "RAGComponents":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.pg_connection: Optional[Any] = None
            self.embedding_function: Optional[HuggingFaceEmbeddings] = None
            self.vectorstore: Optional[Any] = None  # Can be PGVector or QdrantVectorStore
            self.retriever: Optional[Any] = None
            self.vllm_chat_for_rag: Optional[ChatOpenAI] = None
            self.vllm_chat_for_automation: Optional[ChatOpenAI] = None
            self.rag_prompt_template_obj: Optional[ChatPromptTemplate] = None
            self.simple_prompt_template_obj: Optional[ChatPromptTemplate] = None
            self.rag_context_prefix_prompt_template_obj: Optional[
                ChatPromptTemplate
            ] = None
            self.vectorstore_backend: str = VECTOR_STORE_BACKEND
            RAGComponents._initialized = True

    def setup_components(self):
        """Initializes all RAG components."""
        # 1. Embedding Function
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # 2. Vector Store (PGVector or Qdrant based on configuration)
        if self.embedding_function:
            try:
                # Use factory to create the appropriate vectorstore
                self.vectorstore = create_vectorstore(
                    embedding_function=self.embedding_function,
                    collection_name=COLLECTION_NAME,
                    backend=VECTOR_STORE_BACKEND,
                )

                logger.info(f"Initialized {VECTOR_STORE_BACKEND} vectorstore with collection: {COLLECTION_NAME}")

                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

                # Store pg_connection for health checks (only relevant for pgvector)
                if VECTOR_STORE_BACKEND == "pgvector":
                    self.pg_connection = POSTGRES_LIBPQ_CONNECTION

            except Exception as e:
                logger.critical(
                    "Failed to initialize vectorstore/retriever: %s",
                    e,
                    exc_info=True,
                )
                self.pg_connection = self.vectorstore = self.retriever = None
        else:
            self.pg_connection = self.vectorstore = self.retriever = None

        # 3. ChatOpenAI LLM (for RAG) - pointing to vLLM
        self.vllm_chat_for_rag = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_BASE_URL}/v1",
            api_key="EMPTY",  # vLLM doesn't require authentication
            temperature=0.1,
        )

        # 4. ChatOpenAI LLM (for Automation Tasks) - pointing to vLLM
        self.vllm_chat_for_automation = ChatOpenAI(
            model=VLLM_MODEL_FOR_AUTOMATION,
            base_url=f"{VLLM_BASE_URL}/v1",
            api_key="EMPTY",
            temperature=0.2,
        )

        # 5. Prompt Templates
        self.rag_prompt_template_obj = ChatPromptTemplate.from_template(
            RAG_PROMPT_TEMPLATE_STR
        )

        self.simple_prompt_template_obj = ChatPromptTemplate.from_template(
            SIMPLE_PROMPT_TEMPLATE_STR
        )

        self.rag_context_prefix_prompt_template_obj = ChatPromptTemplate.from_template(
            RAG_CONTEXT_PREFIX_TEMPLATE_STR
        )


def get_rag_components() -> RAGComponents:
    """Get the singleton RAG components instance."""
    return RAGComponents()


# --- Helper Functions (format_history_for_lc, format_docs remain the same) ---
def format_history_for_lc(history: list[dict[str, str]]) -> list:
    """Converts custom history format to LangChain Message objects."""
    lc_messages = []
    for msg in history:
        role = msg.get("sender", "user")
        content = msg.get("text", "")
        if not isinstance(content, str):
            content = str(content)

        if role.lower() == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role.lower() == "bot":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def format_docs(docs: list[Document]) -> str:
    """Helper function to format retrieved documents."""
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)


# --- Setup Function ---
def setup_rag_components():
    """Initializes all RAG components using the singleton pattern."""
    rag_components = get_rag_components()
    rag_components.setup_components()


# --- Dependency Functions (getters for initialized components) ---
def get_pg_connection() -> str:
    rag_components = get_rag_components()
    if rag_components.pg_connection is None:
        raise HTTPException(status_code=503, detail="PostgreSQL connection is not available.")
    return rag_components.pg_connection


def get_vectorstore() -> Any:
    """Get the vector store (PGVector or Qdrant based on configuration)."""
    rag_components = get_rag_components()
    if rag_components.vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store is not available.")
    return rag_components.vectorstore


def get_embedding_function() -> HuggingFaceEmbeddings:
    rag_components = get_rag_components()
    if rag_components.embedding_function is None:
        raise HTTPException(
            status_code=503, detail="Embedding function is not available."
        )
    return rag_components.embedding_function


def get_retriever() -> Any:
    rag_components = get_rag_components()
    if rag_components.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not available.")
    return rag_components.retriever


def get_vllm_chat_for_rag() -> ChatOpenAI:
    rag_components = get_rag_components()
    if rag_components.vllm_chat_for_rag is None:
        raise HTTPException(status_code=503, detail="RAG chat model is not available.")
    return rag_components.vllm_chat_for_rag


# Backward compatibility alias - DEPRECATED: Use get_vllm_chat_for_rag instead
# This will be removed in a future version
get_ollama_chat_for_rag = get_vllm_chat_for_rag


def get_llm_for_automation() -> ChatOpenAI:
    rag_components = get_rag_components()
    if rag_components.vllm_chat_for_automation is None:
        raise HTTPException(status_code=503, detail="Automation LLM is not available.")
    return rag_components.vllm_chat_for_automation


# --- Optional Dependencies (for health check) ---
def get_optional_pg_connection() -> Optional[str]:
    rag_components = get_rag_components()
    return rag_components.pg_connection


def get_optional_vllm_chat_for_rag() -> Optional[ChatOpenAI]:
    rag_components = get_rag_components()
    return rag_components.vllm_chat_for_rag


# Backward compatibility alias - DEPRECATED: Use get_optional_vllm_chat_for_rag instead
# This will be removed in a future version
get_optional_ollama_chat_for_rag = get_optional_vllm_chat_for_rag


def get_optional_llm_for_automation() -> Optional[ChatOpenAI]:
    rag_components = get_rag_components()
    return rag_components.vllm_chat_for_automation


# --- Function to Generate RAG Context Prefix ---
async def get_rag_context_prefix(query: str) -> Optional[str]:
    if not RAG_ENABLED:
        return None

    rag_components = get_rag_components()
    if (
        not rag_components.retriever
        or not rag_components.rag_context_prefix_prompt_template_obj
    ):
        return None

    try:
        retrieved_docs = await rag_components.retriever.ainvoke(query)

        if not retrieved_docs:
            return None

        formatted_context = format_docs(retrieved_docs)

        if rag_components.rag_context_prefix_prompt_template_obj:
            formatted_prompt = (
                rag_components.rag_context_prefix_prompt_template_obj.format(
                    context=formatted_context, question=query
                )
            )
            return formatted_prompt
        else:
            return None

    except Exception as e:
        logger.error("Error generating RAG context prefix: %s", e, exc_info=True)
        return None
