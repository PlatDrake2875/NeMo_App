"""Test script for ColBERT retriever."""
from colbert_retriever import ColBERTRetriever
from langchain_core.documents import Document

# Create sample documents
print("Creating sample documents...")
docs = [
    Document(
        page_content="Paris is the capital of France and is known for the Eiffel Tower.",
        metadata={"source": "doc1"}
    ),
    Document(
        page_content="London is the capital of England and home to Big Ben.",
        metadata={"source": "doc2"}
    ),
    Document(
        page_content="Berlin is the capital of Germany and has the Brandenburg Gate.",
        metadata={"source": "doc3"}
    ),
    Document(
        page_content="Rome is the capital of Italy and features the Colosseum.",
        metadata={"source": "doc4"}
    ),
]

print(f"Created {len(docs)} documents\n")

# Initialize ColBERT retriever
print("Initializing ColBERT retriever...")
retriever = ColBERTRetriever(index_name="test_capitals")

# Index documents
print("Indexing documents (this may take 1-2 minutes)...")
index_path = retriever.index_documents(docs)
print(f"✓ Documents indexed at: {index_path}\n")

# Test search
query = "What is the capital of France?"
print(f"Query: '{query}'")
print("Searching...\n")

results = retriever.search(query, k=3)

print("=" * 60)
print("RESULTS:")
print("=" * 60)
for i, (doc, score) in enumerate(results, 1):
    print(f"\n{i}. Score: {score:.3f}")
    print(f"   Content: {doc.page_content}")
    print(f"   Source: {doc.metadata.get('source', 'unknown')}")

print("\n✓ Test completed successfully!")
