"""Test ColBERT reranking (faster, no indexing needed)."""
from colbert_retriever import ColBERTRetriever
from langchain_core.documents import Document

print("Creating sample documents...")
docs = [
    Document(page_content="Paris is the capital of France.", metadata={"source": "doc1"}),
    Document(page_content="London is the capital of England.", metadata={"source": "doc2"}),
    Document(page_content="Berlin is the capital of Germany.", metadata={"source": "doc3"}),
    Document(page_content="Rome is the capital of Italy.", metadata={"source": "doc4"}),
]

print("Initializing ColBERT...")
retriever = ColBERTRetriever()

query = "What is the capital of France?"
print(f"\nQuery: '{query}'")
print("Reranking documents (no indexing, much faster)...\n")

results = retriever.rerank(query, docs, k=3)

print("=" * 60)
print("RESULTS:")
print("=" * 60)
for i, (doc, score) in enumerate(results, 1):
    print(f"\n{i}. Score: {score:.3f}")
    print(f"   Content: {doc.page_content}")

print("\n✓ Rerank test completed!")
