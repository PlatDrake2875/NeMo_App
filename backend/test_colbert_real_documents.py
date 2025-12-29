"""
Test ColBERT reranking with REAL documents from your PGVector database.

This test demonstrates the full two-stage retrieval pipeline:
1. PGVector (first stage) - retrieves candidates using dense embeddings
2. ColBERT (second stage) - reranks candidates using late interaction

This is exactly how the NeMo RAG pipeline works when COLBERT_RERANK_ENABLED=true.

Prerequisites:
- PostgreSQL with pgvector must be running
- Documents must be indexed in PGVector (upload some files via the app first)
- RAGatouille must be installed

Run from backend directory:
    docker exec -it nemo_app-backend-1 python test_colbert_real_documents.py
"""

import asyncio
import sys
from typing import List, Tuple

# ============================================================
# STEP 1: Import required modules
# ============================================================
# These are the same imports used in the actual NeMo RAG pipeline

print("=" * 70)
print("STEP 1: Importing modules...")
print("=" * 70)

try:
    # LangChain for document handling and PGVector
    from langchain_core.documents import Document
    from langchain_postgres import PGVector
    from langchain_huggingface import HuggingFaceEmbeddings
    print("  ✓ LangChain modules imported")
except ImportError as e:
    print(f"  ✗ LangChain import failed: {e}")
    sys.exit(1)

try:
    # ColBERT retriever (our implementation)
    from services.colbert_retriever import ColBERTRetriever, is_ragatouille_available
    if not is_ragatouille_available():
        print("  ✗ RAGatouille not available")
        sys.exit(1)
    print("  ✓ ColBERT retriever imported")
except ImportError as e:
    print(f"  ✗ ColBERT import failed: {e}")
    sys.exit(1)

try:
    # Configuration (database connection, model names, etc.)
    from config import (
        POSTGRES_CONNECTION_STRING,
        COLLECTION_NAME,
        EMBEDDING_MODEL_NAME,
        COLBERT_FIRST_STAGE_K,
        COLBERT_FINAL_K,
    )
    print("  ✓ Configuration loaded")
    print(f"      - Database collection: {COLLECTION_NAME}")
    print(f"      - Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"      - First stage k: {COLBERT_FIRST_STAGE_K}")
    print(f"      - Final k: {COLBERT_FINAL_K}")
except ImportError as e:
    print(f"  ✗ Config import failed: {e}")
    sys.exit(1)

print()


# ============================================================
# STEP 2: Connect to PGVector (your real database)
# ============================================================
print("=" * 70)
print("STEP 2: Connecting to PGVector database...")
print("=" * 70)

try:
    # Initialize the same embedding function used when documents were indexed
    # This MUST match what was used during document upload
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print(f"  ✓ Embedding function initialized: {EMBEDDING_MODEL_NAME}")

    # Connect to PGVector - this is your real vector database
    vectorstore = PGVector(
        connection=POSTGRES_CONNECTION_STRING,
        embeddings=embedding_function,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )
    print(f"  ✓ Connected to PGVector collection: {COLLECTION_NAME}")

    # Create a retriever that will fetch documents
    # k=20 means "get top 20 candidates" for ColBERT to rerank
    pgvector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    print("  ✓ PGVector retriever created (k=20 candidates)")

except Exception as e:
    print(f"  ✗ Database connection failed: {e}")
    print("\n  Make sure:")
    print("    1. PostgreSQL container is running")
    print("    2. You have uploaded documents via the app")
    sys.exit(1)

print()


# ============================================================
# STEP 3: Initialize ColBERT reranker
# ============================================================
print("=" * 70)
print("STEP 3: Initializing ColBERT reranker...")
print("=" * 70)

try:
    # ColBERT will be used to rerank PGVector's candidates
    # The model is loaded lazily (on first use)
    colbert = ColBERTRetriever()
    print("  ✓ ColBERT retriever initialized")
    print("      - Model: colbert-ir/colbertv2.0")
    print("      - Mode: Reranking (no index needed)")
except Exception as e:
    print(f"  ✗ ColBERT initialization failed: {e}")
    sys.exit(1)

print()


# ============================================================
# STEP 4: Define the two-stage retrieval function
# ============================================================
print("=" * 70)
print("STEP 4: Setting up two-stage retrieval pipeline...")
print("=" * 70)

async def two_stage_retrieval(
    query: str,
    first_stage_k: int = 20,
    final_k: int = 5,
    verbose: bool = True
) -> Tuple[List[Document], List[Tuple[Document, float]]]:
    """
    Two-stage retrieval: PGVector → ColBERT reranking.

    This is the exact same logic used in rag_components.py
    when COLBERT_RERANK_ENABLED=true.

    Args:
        query: User's question
        first_stage_k: How many candidates PGVector retrieves
        final_k: How many results after ColBERT reranking
        verbose: Print detailed output

    Returns:
        Tuple of (pgvector_candidates, colbert_reranked_results)
    """

    if verbose:
        print(f"\n  Query: '{query}'")
        print("-" * 60)

    # STAGE 1: PGVector retrieval (fast, approximate)
    # Uses dense embeddings (all-MiniLM-L6-v2) to find similar documents
    if verbose:
        print(f"\n  [Stage 1] PGVector retrieving top {first_stage_k} candidates...")

    pgvector_candidates = await pgvector_retriever.ainvoke(query)

    if not pgvector_candidates:
        if verbose:
            print("  ⚠ No documents found in database!")
            print("    Upload some documents via the app first.")
        return [], []

    if verbose:
        print(f"  ✓ Retrieved {len(pgvector_candidates)} candidates")
        print("\n  PGVector Results (first stage):")
        for i, doc in enumerate(pgvector_candidates[:5], 1):
            content_preview = doc.page_content[:80].replace('\n', ' ')
            print(f"    {i}. {content_preview}...")

    # STAGE 2: ColBERT reranking (accurate, late interaction)
    # Uses token-level matching for better semantic understanding
    if verbose:
        print(f"\n  [Stage 2] ColBERT reranking to top {final_k}...")

    reranked_results = await colbert.arerank(
        query=query,
        documents=pgvector_candidates,
        k=final_k
    )

    if verbose:
        print(f"  ✓ Reranked to {len(reranked_results)} results")
        print("\n  ColBERT Results (after reranking):")
        for i, (doc, score) in enumerate(reranked_results, 1):
            content_preview = doc.page_content[:80].replace('\n', ' ')
            print(f"    {i}. [Score: {score:.3f}] {content_preview}...")

    return pgvector_candidates, reranked_results

print("  ✓ Two-stage retrieval function defined")
print()


# ============================================================
# STEP 5: Run the test with real queries
# ============================================================
print("=" * 70)
print("STEP 5: Testing with real queries on YOUR documents...")
print("=" * 70)

async def run_tests():
    """Run test queries against your real document database."""

    # You can modify these queries to match your uploaded documents
    test_queries = [
        "What is this document about?",
        "Explain the main concepts",
        "Give me a summary",
    ]

    print("\n" + "=" * 70)
    print("Enter your own query (or press Enter to use defaults):")
    print("=" * 70)

    try:
        # Try to get user input (works in interactive mode)
        user_query = input("\nYour query: ").strip()
        if user_query:
            test_queries = [user_query]
    except EOFError:
        # Non-interactive mode, use defaults
        print("(Using default queries)")

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        try:
            pgvector_results, colbert_results = await two_stage_retrieval(
                query=query,
                first_stage_k=20,  # PGVector gets 20 candidates
                final_k=5,         # ColBERT picks top 5
                verbose=True
            )

            if colbert_results:
                print("\n" + "-" * 70)
                print("COMPARISON: How reranking changed the order")
                print("-" * 70)

                # Show how rankings changed
                for i, (doc, score) in enumerate(colbert_results, 1):
                    # Find original PGVector rank
                    original_rank = "?"
                    for j, orig_doc in enumerate(pgvector_results, 1):
                        if orig_doc.page_content == doc.page_content:
                            original_rank = j
                            break

                    movement = ""
                    if isinstance(original_rank, int):
                        if original_rank > i:
                            movement = f"↑ moved UP from #{original_rank}"
                        elif original_rank < i:
                            movement = f"↓ moved DOWN from #{original_rank}"
                        else:
                            movement = "= stayed same"

                    print(f"  #{i} (was #{original_rank}) {movement}")
                    print(f"      Score: {score:.3f}")
                    preview = doc.page_content[:100].replace('\n', ' ')
                    print(f"      Content: {preview}...")
                    print()

        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nThis demonstrates how ColBERT improves retrieval quality by")
    print("reranking PGVector's candidates based on deeper semantic matching.")
    print("\nTo enable this in production, set in your .env:")
    print("    COLBERT_RERANK_ENABLED=true")
    print("=" * 70)


# ============================================================
# MAIN: Run everything
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING TWO-STAGE RETRIEVAL TEST")
    print("PGVector (dense embeddings) → ColBERT (late interaction)")
    print("=" * 70 + "\n")

    asyncio.run(run_tests())
