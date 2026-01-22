# backend/rag_components.py
"""
RAG Components - Central management of all Retrieval-Augmented Generation components.

This module provides a singleton pattern for managing:
1. PGVector/Qdrant - First-stage retrieval (fast, approximate nearest neighbor)
2. ColBERT - Second-stage reranking (accurate, late interaction)
3. Embeddings - HuggingFace sentence transformers
4. LLM - vLLM for text generation
5. Prompt templates - For formatting context and queries

The two-stage retrieval architecture:
------------------------------------
When COLBERT_RERANK_ENABLED=True:
    Query → VectorStore (retrieve 20 candidates) → ColBERT (rerank to top 5) → LLM

When COLBERT_RERANK_ENABLED=False:
    Query → VectorStore (retrieve 5 candidates) → LLM

This approach balances speed (VectorStore) with accuracy (ColBERT).
"""

from typing import Any, List, Optional, Tuple

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
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
    # ColBERT configuration
    COLBERT_RERANK_ENABLED,
    COLBERT_MODEL_NAME,
    COLBERT_FIRST_STAGE_K,
    COLBERT_FINAL_K,
    COLBERT_INDEX_ROOT,
    COLBERT_N_GPU,
    logger,
)
from vectorstore_factory import create_vectorstore

# Import ColBERT retriever (optional - graceful degradation if not available)
try:
    from services.colbert_retriever import ColBERTRetriever, is_ragatouille_available
    COLBERT_AVAILABLE = is_ragatouille_available()
except ImportError:
    COLBERT_AVAILABLE = False
    ColBERTRetriever = None

    def is_ragatouille_available():
        return False


class RAGComponents:
    """
    Singleton class to manage RAG components without global variables.

    This class centralizes the initialization and access to all components
    needed for the RAG pipeline, including the new ColBERT reranker.

    Components:
    -----------
    - pg_connection: PostgreSQL connection for health checks
    - embedding_function: HuggingFace embeddings for PGVector
    - vectorstore: PGVector or Qdrant for first-stage retrieval
    - retriever: LangChain retriever wrapper for vectorstore
    - colbert_retriever: ColBERT for second-stage reranking (optional)
    - vllm_chat_for_rag: LLM for generating responses
    - prompt templates: For formatting context and questions
    """

    _instance: Optional["RAGComponents"] = None
    _initialized: bool = False

    def __new__(cls) -> "RAGComponents":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # Existing components
            self.pg_connection: Optional[Any] = None
            self.embedding_function: Optional[HuggingFaceEmbeddings] = None
            self.vectorstore: Optional[Any] = None  # Can be PGVector or QdrantVectorStore
            self.retriever: Optional[Any] = None
            self.vllm_chat_for_rag: Optional[ChatOpenAI] = None
            self.vllm_chat_for_automation: Optional[ChatOpenAI] = None
            self.rag_prompt_template_obj: Optional[ChatPromptTemplate] = None
            self.simple_prompt_template_obj: Optional[ChatPromptTemplate] = None
            self.rag_context_prefix_prompt_template_obj: Optional[ChatPromptTemplate] = None
            self.vectorstore_backend: str = VECTOR_STORE_BACKEND

            # ColBERT components (new)
            self.colbert_retriever: Optional[ColBERTRetriever] = None
            self.colbert_rerank_enabled: bool = False

            RAGComponents._initialized = True

    def setup_components(self):
        """
        Initializes all RAG components.

        This method is called once during application startup.
        It sets up the entire RAG pipeline including optional ColBERT reranking.
        """
        # 1. Embedding Function (used by vectorstore for dense retrieval)
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info(f"Initialized embedding function: {EMBEDDING_MODEL_NAME}")

        # 2. Vector Store (PGVector or Qdrant based on configuration)
        if self.embedding_function:
            try:
                # Use factory to create the appropriate vectorstore
                # async_mode=True required for ainvoke() in retrieval
                self.vectorstore = create_vectorstore(
                    embedding_function=self.embedding_function,
                    collection_name=COLLECTION_NAME,
                    backend=VECTOR_STORE_BACKEND,
                    async_mode=True,
                )

                logger.info(f"Initialized {VECTOR_STORE_BACKEND} vectorstore with collection: {COLLECTION_NAME}")

                # Configure retriever based on whether ColBERT reranking is enabled
                # If ColBERT is enabled, retrieve more candidates for reranking
                first_stage_k = COLBERT_FIRST_STAGE_K if COLBERT_RERANK_ENABLED else COLBERT_FINAL_K
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": first_stage_k}
                )
                logger.info(f"Retriever configured with k={first_stage_k}")

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

        # 3. ColBERT Retriever (second-stage reranking - optional)
        self._setup_colbert()

        # 4. ChatOpenAI LLM (for RAG) - pointing to vLLM
        self.vllm_chat_for_rag = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_BASE_URL}/v1",
            api_key="EMPTY",  # vLLM doesn't require authentication
            temperature=0.1,
        )
        logger.info(f"Initialized vLLM chat for RAG: {VLLM_MODEL}")

        # 5. ChatOpenAI LLM (for Automation Tasks) - pointing to vLLM
        self.vllm_chat_for_automation = ChatOpenAI(
            model=VLLM_MODEL_FOR_AUTOMATION,
            base_url=f"{VLLM_BASE_URL}/v1",
            api_key="EMPTY",
            temperature=0.2,
        )

        # 6. Prompt Templates
        self.rag_prompt_template_obj = ChatPromptTemplate.from_template(
            RAG_PROMPT_TEMPLATE_STR
        )
        self.simple_prompt_template_obj = ChatPromptTemplate.from_template(
            SIMPLE_PROMPT_TEMPLATE_STR
        )
        self.rag_context_prefix_prompt_template_obj = ChatPromptTemplate.from_template(
            RAG_CONTEXT_PREFIX_TEMPLATE_STR
        )

        # Log final configuration
        self._log_configuration()

    def _setup_colbert(self):
        """
        Initialize ColBERT retriever for second-stage reranking.

        ColBERT uses late interaction to provide more accurate relevance scoring
        than dense embeddings alone. It's particularly effective for:
        - Complex queries requiring semantic understanding
        - Technical/domain-specific content
        - Cases where first-stage retrieval returns borderline candidates

        The ColBERT model is loaded lazily (on first use) to avoid slow startup.
        """
        if not COLBERT_RERANK_ENABLED:
            logger.info("ColBERT reranking is disabled (COLBERT_RERANK_ENABLED=False)")
            self.colbert_rerank_enabled = False
            return

        if not COLBERT_AVAILABLE:
            logger.warning(
                "ColBERT reranking requested but RAGatouille is not installed. "
                "Install with: pip install ragatouille"
            )
            self.colbert_rerank_enabled = False
            return

        try:
            self.colbert_retriever = ColBERTRetriever(
                model_name=COLBERT_MODEL_NAME,
                index_root=COLBERT_INDEX_ROOT,
                n_gpu=COLBERT_N_GPU
            )
            self.colbert_rerank_enabled = True
            logger.info(
                f"ColBERT reranker initialized: model={COLBERT_MODEL_NAME}, "
                f"first_stage_k={COLBERT_FIRST_STAGE_K}, final_k={COLBERT_FINAL_K}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ColBERT retriever: {e}", exc_info=True)
            self.colbert_retriever = None
            self.colbert_rerank_enabled = False

    def _log_configuration(self):
        """Log the final RAG configuration for debugging."""
        logger.info("=" * 60)
        logger.info("RAG Configuration Summary:")
        logger.info(f"  RAG Enabled: {RAG_ENABLED}")
        logger.info(f"  Vector Store Backend: {VECTOR_STORE_BACKEND}")
        logger.info(f"  VectorStore: {'OK' if self.vectorstore else 'NOT AVAILABLE'}")
        logger.info(f"  ColBERT Reranking: {'ENABLED' if self.colbert_rerank_enabled else 'DISABLED'}")
        if self.colbert_rerank_enabled:
            logger.info(f"    - First stage (VectorStore): k={COLBERT_FIRST_STAGE_K}")
            logger.info(f"    - Final (after ColBERT): k={COLBERT_FINAL_K}")
        logger.info(f"  LLM Model: {VLLM_MODEL}")
        logger.info("=" * 60)

    async def retrieve_with_rerank(
        self,
        query: str,
        first_stage_k: Optional[int] = None,
        final_k: Optional[int] = None,
        use_colbert_override: Optional[bool] = None,
    ) -> List[Document]:
        """
        Two-stage retrieval: VectorStore candidates → ColBERT reranking.

        This is the recommended way to retrieve documents when ColBERT is enabled.
        It provides better accuracy than VectorStore alone while maintaining speed.

        Args:
            query: The user's search query
            first_stage_k: Number of VectorStore candidates (default: config value)
            final_k: Number of results after reranking (default: config value)
            use_colbert_override: Optional override for ColBERT usage (None = use config)

        Returns:
            List of reranked Document objects

        Flow:
            1. VectorStore retrieves first_stage_k candidates using dense embeddings
            2. ColBERT reranks candidates using late interaction
            3. Top final_k documents are returned

        If ColBERT is not available, falls back to VectorStore-only retrieval.
        """
        first_stage_k = first_stage_k or COLBERT_FIRST_STAGE_K
        final_k = final_k or COLBERT_FINAL_K

        # Determine if ColBERT should be used for this request
        use_colbert = use_colbert_override if use_colbert_override is not None else self.colbert_rerank_enabled

        if not self.retriever:
            logger.warning("Retriever not available")
            return []

        # First stage: VectorStore retrieval
        logger.info(f"[RAG] Stage 1: PGVector retrieving {first_stage_k} candidates...")
        candidates = await self.retriever.ainvoke(query)

        if not candidates:
            logger.info("[RAG] No candidates found in first stage")
            return []

        logger.info(f"[RAG] Stage 1 complete: Got {len(candidates)} candidates from PGVector")

        # Log PGVector's top candidates (before ColBERT reranking)
        logger.info("[RAG] === PGVector Top 5 Candidates (by embedding similarity) ===")
        for i, doc in enumerate(candidates[:5]):
            source = doc.metadata.get('source', 'unknown')[:30]
            snippet = doc.page_content[:80].replace('\n', ' ')
            logger.info(f"[RAG]   PGVector #{i+1}: [{source}...] \"{snippet}...\"")

        # Second stage: ColBERT reranking (if enabled and available)
        if use_colbert and self.colbert_retriever:
            logger.info(f"[RAG] Stage 2: ColBERT reranking {len(candidates)} candidates to top {final_k}...")
            try:
                reranked_results = await self.colbert_retriever.arerank(
                    query=query,
                    documents=candidates,
                    k=final_k
                )
                # Extract just the documents (not scores) for compatibility
                reranked_docs = [doc for doc, score in reranked_results]
                top_score = reranked_results[0][1] if reranked_results else 'N/A'

                # Log ColBERT's reranked results with scores
                logger.info(f"[RAG] === ColBERT Reranked Top {final_k} (by token-level semantic matching) ===")
                for i, (doc, score) in enumerate(reranked_results):
                    source = doc.metadata.get('source', 'unknown')[:30]
                    snippet = doc.page_content[:80].replace('\n', ' ')
                    # Find original PGVector rank
                    orig_rank = next((j+1 for j, c in enumerate(candidates) if c.page_content == doc.page_content), '?')
                    logger.info(f"[RAG]   ColBERT #{i+1} (score={score:.2f}, was PGVector #{orig_rank}): \"{snippet}...\"")

                logger.info(
                    f"[RAG] Stage 2 complete: ColBERT returned {len(reranked_docs)} docs "
                    f"(top score: {top_score:.4f})"
                )
                return reranked_docs
            except Exception as e:
                logger.error(f"ColBERT reranking failed, falling back to VectorStore: {e}")
                # Fall back to VectorStore results
                return candidates[:final_k]
        else:
            # No ColBERT, use VectorStore results directly
            return candidates[:final_k]

    async def retrieve_with_scores(
        self,
        query: str,
        first_stage_k: Optional[int] = None,
        final_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Two-stage retrieval returning documents with their ColBERT scores.

        Similar to retrieve_with_rerank but includes relevance scores.
        Useful for debugging, benchmarking, and displaying confidence.

        Args:
            query: The user's search query
            first_stage_k: Number of VectorStore candidates
            final_k: Number of final results

        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        first_stage_k = first_stage_k or COLBERT_FIRST_STAGE_K
        final_k = final_k or COLBERT_FINAL_K

        if not self.retriever:
            return []

        # First stage
        candidates = await self.retriever.ainvoke(query)
        if not candidates:
            return []

        # Second stage with scores
        if self.colbert_rerank_enabled and self.colbert_retriever:
            try:
                return await self.colbert_retriever.arerank(
                    query=query,
                    documents=candidates,
                    k=final_k
                )
            except Exception as e:
                logger.error(f"ColBERT reranking failed: {e}")
                # Fall back - return without scores
                return [(doc, 0.0) for doc in candidates[:final_k]]
        else:
            return [(doc, 0.0) for doc in candidates[:final_k]]


def get_rag_components() -> RAGComponents:
    """Get the singleton RAG components instance."""
    return RAGComponents()


# --- Helper Functions ---
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


def format_docs_with_scores(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """
    Format documents with their relevance scores for debugging/display.

    Useful for understanding why certain documents were retrieved
    and their relative importance.
    """
    if not docs_with_scores:
        return "No relevant context found."

    formatted = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        formatted.append(f"[{i}] (score: {score:.3f})\n{doc.page_content}")

    return "\n\n".join(formatted)


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


def get_colbert_retriever() -> Optional[ColBERTRetriever]:
    """
    Get the ColBERT retriever for reranking.

    Returns None if ColBERT is not enabled or available.
    This allows graceful degradation in code that uses ColBERT.
    """
    rag_components = get_rag_components()
    return rag_components.colbert_retriever


def is_colbert_rerank_enabled() -> bool:
    """Check if ColBERT reranking is enabled and available."""
    rag_components = get_rag_components()
    return rag_components.colbert_rerank_enabled


def get_vllm_chat_for_rag() -> ChatOpenAI:
    rag_components = get_rag_components()
    if rag_components.vllm_chat_for_rag is None:
        raise HTTPException(status_code=503, detail="RAG chat model is not available.")
    return rag_components.vllm_chat_for_rag


# Backward compatibility alias - DEPRECATED: Use get_vllm_chat_for_rag instead
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


# Backward compatibility alias - DEPRECATED
get_optional_ollama_chat_for_rag = get_optional_vllm_chat_for_rag


def get_optional_llm_for_automation() -> Optional[ChatOpenAI]:
    rag_components = get_rag_components()
    return rag_components.vllm_chat_for_automation


def get_optional_colbert_retriever() -> Optional[ColBERTRetriever]:
    """Get ColBERT retriever or None if not available (for health checks)."""
    rag_components = get_rag_components()
    return rag_components.colbert_retriever


# --- Function to Generate RAG Context Prefix ---
async def get_rag_context_prefix(
    query: str,
    collection_name: Optional[str] = None,
    use_colbert: Optional[bool] = None,
    embedder: Optional[str] = None,
) -> Optional[str]:
    """
    Generate RAG context prefix for a query.

    This is the main entry point for the RAG pipeline. It:
    1. Retrieves relevant documents (with optional ColBERT reranking)
    2. Formats them into a context string
    3. Creates a prompt with the context and question

    The two-stage retrieval is automatically used if ColBERT is enabled.

    Args:
        query: The user's question
        collection_name: Optional collection name to use (defaults to COLLECTION_NAME)
        use_colbert: Optional override for ColBERT usage (defaults to COLBERT_RERANK_ENABLED)
        embedder: Optional embedding model name to use (defaults to EMBEDDING_MODEL_NAME)

    Returns:
        Formatted prompt with context, or None if RAG is disabled/unavailable
    """
    if not RAG_ENABLED:
        return None

    rag_components = get_rag_components()
    if not rag_components.rag_context_prefix_prompt_template_obj:
        return None

    # Determine effective collection and ColBERT settings
    effective_collection = collection_name or COLLECTION_NAME
    effective_use_colbert = use_colbert if use_colbert is not None else COLBERT_RERANK_ENABLED

    try:
        # Check if we need to use a different collection
        if effective_collection != COLLECTION_NAME:
            logger.info(f"[RAG] Using custom collection: {effective_collection}")
            retrieved_docs = await _retrieve_from_collection(
                query, effective_collection, effective_use_colbert, embedder
            )
        else:
            # Use default retriever
            if not rag_components.retriever:
                return None
            retrieved_docs = await rag_components.retrieve_with_rerank(
                query, use_colbert_override=effective_use_colbert
            )

        if not retrieved_docs:
            return None

        formatted_context = format_docs(retrieved_docs)

        formatted_prompt = (
            rag_components.rag_context_prefix_prompt_template_obj.format(
                context=formatted_context, question=query
            )
        )
        return formatted_prompt

    except Exception as e:
        logger.error("Error generating RAG context prefix: %s", e, exc_info=True)
        return None


def _get_collection_backend(collection_name: str) -> str:
    """
    Look up the vector backend for a collection from the database.

    Args:
        collection_name: Name of the collection

    Returns:
        The vector backend ('pgvector' or 'qdrant'), defaults to VECTOR_STORE_BACKEND
    """
    from sqlalchemy import create_engine, text
    from config import POSTGRES_CONNECTION_STRING

    try:
        engine = create_engine(POSTGRES_CONNECTION_STRING)
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT vector_backend FROM processed_datasets
                    WHERE collection_name = :collection_name
                    LIMIT 1
                """),
                {"collection_name": collection_name}
            )
            row = result.fetchone()
            if row and row[0]:
                logger.info(f"[RAG] Collection '{collection_name}' uses backend: {row[0]}")
                return row[0]
    except Exception as e:
        logger.warning(f"Could not look up backend for collection '{collection_name}': {e}")

    # Fall back to default
    return VECTOR_STORE_BACKEND


async def _retrieve_from_collection(
    query: str,
    collection_name: str,
    use_colbert: bool,
    embedder: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve documents from a specific collection.

    Creates a temporary vectorstore for the given collection and retrieves documents.

    Args:
        query: The user's query
        collection_name: Name of the vector store collection
        use_colbert: Whether to use ColBERT reranking
        embedder: Optional embedding model name (defaults to EMBEDDING_MODEL_NAME)

    Returns:
        List of retrieved documents
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        # Use provided embedder or fall back to config default
        model_name = embedder or EMBEDDING_MODEL_NAME

        # Handle nomic models which require trust_remote_code=True
        model_kwargs = {"trust_remote_code": True} if "nomic" in model_name else {}

        # Create embedding function (model_kwargs must be a dict, not None)
        embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )

        # Look up the actual backend for this collection
        backend = _get_collection_backend(collection_name)

        # Create vectorstore for the specified collection with correct backend
        vectorstore = create_vectorstore(
            embedding_function=embedding_function,
            collection_name=collection_name,
            backend=backend,
            async_mode=True,
        )

        # Create retriever
        first_stage_k = COLBERT_FIRST_STAGE_K if use_colbert else COLBERT_FINAL_K
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": first_stage_k},
        )

        # Retrieve candidates
        logger.info(f"[RAG] Retrieving from collection '{collection_name}' (backend={backend}) with k={first_stage_k}")
        candidates = await retriever.ainvoke(query)

        if not candidates:
            logger.info(f"[RAG] No candidates found in collection '{collection_name}'")
            return []

        # Apply ColBERT reranking if enabled and available
        if use_colbert and COLBERT_AVAILABLE:
            rag_components = get_rag_components()
            if rag_components.colbert_retriever:
                logger.info(f"[RAG] Applying ColBERT reranking on {len(candidates)} candidates")
                reranked_with_scores = rag_components.colbert_retriever.rerank(
                    query, candidates, k=COLBERT_FINAL_K
                )
                # Extract just the documents (rerank returns List[Tuple[Document, float]])
                return [doc for doc, score in reranked_with_scores]

        return candidates[:COLBERT_FINAL_K]

    except Exception as e:
        logger.error(f"Error retrieving from collection '{collection_name}': {e}", exc_info=True)
        return []


async def get_rag_context_with_scores(query: str) -> Optional[Tuple[str, List[Tuple[Document, float]]]]:
    """
    Generate RAG context along with document scores.

    Useful for debugging and displaying relevance information to users.

    Args:
        query: The user's question

    Returns:
        Tuple of (formatted_prompt, docs_with_scores), or None if unavailable
    """
    if not RAG_ENABLED:
        return None

    rag_components = get_rag_components()
    if not rag_components.retriever:
        return None

    try:
        docs_with_scores = await rag_components.retrieve_with_scores(query)

        if not docs_with_scores:
            return None

        # Extract just documents for formatting
        docs = [doc for doc, score in docs_with_scores]
        formatted_context = format_docs(docs)

        if rag_components.rag_context_prefix_prompt_template_obj:
            formatted_prompt = (
                rag_components.rag_context_prefix_prompt_template_obj.format(
                    context=formatted_context, question=query
                )
            )
            return formatted_prompt, docs_with_scores

        return None

    except Exception as e:
        logger.error("Error generating RAG context with scores: %s", e, exc_info=True)
        return None
