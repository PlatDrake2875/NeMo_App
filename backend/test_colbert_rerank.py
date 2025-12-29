"""
Test script for ColBERT reranking (no indexing required).

This test demonstrates:
1. Using ColBERT as a reranker (primary use case in NeMo RAG)
2. No indexing needed - works directly on candidate documents
3. Much faster than full indexing for small document sets

This is how ColBERT is used in the NeMo RAG pipeline:
- PGVector retrieves candidates (fast, approximate)
- ColBERT reranks candidates (accurate, late interaction)

Run from backend directory:
    python test_colbert_rerank.py

Note: First run will download the ColBERT model (~500MB).
"""

from services.colbert_retriever import ColBERTRetriever, is_ragatouille_available
from langchain_core.documents import Document

if __name__ == "__main__":
    # Check if RAGatouille is available
    if not is_ragatouille_available():
        print("ERROR: RAGatouille is not installed.")
        print("Install with: pip install ragatouille")
        exit(1)

    # Simulate documents that might come from PGVector first-stage retrieval
    print("Creating sample documents (simulating PGVector candidates)...")
    docs = [
        Document(
            page_content="Paris is the capital of France and is known for the Eiffel Tower.",
            metadata={"source": "doc1", "pgvector_rank": 1}
        ),
        Document(
            page_content="France is a country in Western Europe with a rich history.",
            metadata={"source": "doc2", "pgvector_rank": 2}
        ),
        Document(
            page_content="The French Revolution began in Paris in 1789.",
            metadata={"source": "doc3", "pgvector_rank": 3}
        ),
        Document(
            page_content="London is the capital of England and home to Big Ben.",
            metadata={"source": "doc4", "pgvector_rank": 4}
        ),
        Document(
            page_content="French cuisine is famous worldwide for its sophistication.",
            metadata={"source": "doc5", "pgvector_rank": 5}
        ),
    ]
    print(f"Created {len(docs)} candidate documents\n")

    # Initialize ColBERT retriever (for reranking, no index needed)
    print("Initializing ColBERT retriever...")
    print("(First run will download the model, ~500MB)")
    retriever = ColBERTRetriever()

    # Test reranking
    query = "What is the capital of France?"
    print(f"\nQuery: '{query}'")
    print("\nReranking documents (no indexing, much faster)...")
    print("-" * 60)

    results = retriever.rerank(query, docs, k=3)

    print("\nReranked Results:")
    print("=" * 60)
    for i, (doc, score) in enumerate(results, 1):
        original_rank = doc.metadata.get('pgvector_rank', '?')
        print(f"\n{i}. ColBERT Score: {score:.3f} (was PGVector rank #{original_rank})")
        print(f"   Content: {doc.page_content}")
        print(f"   Source: {doc.metadata.get('source', 'unknown')}")

    print("\n" + "=" * 60)
    print("Analysis:")
    print("-" * 60)
    print("Notice how ColBERT reranks the documents based on semantic relevance.")
    print("The document about 'Paris is the capital of France' should rank highest")
    print("because it directly answers the query, even if other documents mention")
    print("'France' more frequently.")
    print("=" * 60)

    # Test with a different query to show reranking effect
    query2 = "Tell me about European history"
    print(f"\n\nSecond Query: '{query2}'")
    print("-" * 60)

    results2 = retriever.rerank(query2, docs, k=3)

    print("\nReranked Results:")
    print("=" * 60)
    for i, (doc, score) in enumerate(results2, 1):
        original_rank = doc.metadata.get('pgvector_rank', '?')
        print(f"\n{i}. ColBERT Score: {score:.3f} (was PGVector rank #{original_rank})")
        print(f"   Content: {doc.page_content}")

    print("\n" + "=" * 60)
    print("Rerank test completed successfully!")
    print("=" * 60)
