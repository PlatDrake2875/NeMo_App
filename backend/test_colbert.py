"""
Test script for ColBERT retriever - Full indexing and search.

This test demonstrates:
1. Creating a ColBERT index from documents
2. Searching the index using late interaction
3. Result format with scores and metadata

Run from backend directory:
    python test_colbert.py

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

    # Create sample documents about European capitals
    print("Creating sample documents...")
    docs = [
        Document(
            page_content="Paris is the capital of France and is known for the Eiffel Tower.",
            metadata={"source": "doc1", "country": "France"}
        ),
        Document(
            page_content="London is the capital of England and home to Big Ben.",
            metadata={"source": "doc2", "country": "England"}
        ),
        Document(
            page_content="Berlin is the capital of Germany and has the Brandenburg Gate.",
            metadata={"source": "doc3", "country": "Germany"}
        ),
        Document(
            page_content="Rome is the capital of Italy and features the Colosseum.",
            metadata={"source": "doc4", "country": "Italy"}
        ),
        Document(
            page_content="Madrid is the capital of Spain and is famous for the Prado Museum.",
            metadata={"source": "doc5", "country": "Spain"}
        ),
    ]

    print(f"Created {len(docs)} documents\n")

    # Initialize ColBERT retriever
    print("Initializing ColBERT retriever...")
    print("(First run will download the model, ~500MB)")
    retriever = ColBERTRetriever(index_name="test_capitals")

    # Index documents
    print("\nIndexing documents (this may take 1-2 minutes on first run)...")
    index_path = retriever.index_documents(docs)
    print(f"Documents indexed at: {index_path}\n")

    # Test search
    queries = [
        "What is the capital of France?",
        "Tell me about German landmarks",
        "Which city has the Colosseum?"
    ]

    for query in queries:
        print("=" * 60)
        print(f"Query: '{query}'")
        print("-" * 60)

        results = retriever.search(query, k=3)

        print("Results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Content: {doc.page_content}")
            print(f"   Source: {doc.metadata.get('source', 'unknown')}")
            print(f"   Country: {doc.metadata.get('country', 'unknown')}")

        print()

    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
