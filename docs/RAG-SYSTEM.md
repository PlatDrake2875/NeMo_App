# RAG System Documentation

Complete guide to the Retrieval Augmented Generation (RAG) system implementation in the NeMo Guardrails Testing Application.

## Table of Contents

1. [RAG Overview](#rag-overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Document Processing Pipeline](#document-processing-pipeline)
5. [Embedding Generation](#embedding-generation)
6. [Vector Storage](#vector-storage)
7. [Retrieval Process](#retrieval-process)
8. [Prompt Construction](#prompt-construction)
9. [Configuration](#configuration)
10. [API Integration](#api-integration)
11. [Performance Optimization](#performance-optimization)
12. [Troubleshooting](#troubleshooting)

## RAG Overview

**Retrieval Augmented Generation (RAG)** enhances LLM responses by retrieving relevant context from a knowledge base before generating answers.

### Why RAG?

| Without RAG | With RAG |
|-------------|----------|
| Relies only on training data | Accesses up-to-date documents |
| May hallucinate facts | Grounds responses in evidence |
| Limited to general knowledge | Domain-specific expertise |
| No source attribution | Can cite sources |

### How RAG Works

```
┌─────────────────┐
│  User Query     │
│  "What is V1?"  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  1. Query Embedding             │
│  Convert to vector              │
│  [0.23, -0.45, 0.67, ...]      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  2. Vector Search (ChromaDB)    │
│  Find similar document chunks   │
│  based on semantic similarity   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  3. Retrieve Top-K Chunks       │
│  - Aviation manual, page 12     │
│  - Flight ops guide, page 5     │
│  - Safety procedures, page 8    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  4. Construct Prompt            │
│  Context: [retrieved chunks]    │
│  Question: What is V1?          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  5. LLM Generation              │
│  Generate answer using context  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  6. Response                    │
│  "Based on the aviation manual, │
│   V1 is the critical engine..." │
└─────────────────────────────────┘
```

## Architecture

### Component Diagram

```
┌──────────────────────────────────────────────────┐
│           RAG Components (Singleton)             │
│  File: backend/rag_components.py:15-125          │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────────┐  ┌─────────────────┐       │
│  │ ChromaDB Client │  │ Embedding Model │       │
│  │ (HttpClient)    │  │ (HuggingFace)   │       │
│  └────────┬────────┘  └────────┬────────┘       │
│           │                    │                 │
│           └──────────┬─────────┘                 │
│                      │                           │
│           ┌──────────▼──────────┐                │
│           │   Vector Store      │                │
│           │   (Chroma)          │                │
│           └──────────┬──────────┘                │
│                      │                           │
│           ┌──────────▼──────────┐                │
│           │   Retriever         │                │
│           │   (LangChain)       │                │
│           └─────────────────────┘                │
│                                                  │
└──────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
┌────────────────┐         ┌──────────────────┐
│  ChromaDB      │         │  Ollama          │
│  Container     │         │  (Host/LLM)      │
│  Port: 8001    │         │  Port: 11434     │
└────────────────┘         └──────────────────┘
```

### Data Flow

```
Upload Flow:
PDF → PyPDF Parser → Text Chunks → Embeddings → ChromaDB

Query Flow:
Query → Embedding → ChromaDB Search → Top-K Docs → Prompt → LLM → Response
```

## Components

### 1. RAG Components Singleton

**File**: `backend/rag_components.py:15-125`

**Purpose**: Manages all RAG-related resources as a singleton to ensure single initialization and shared access.

```python
class RAGComponents:
    """Singleton manager for RAG components"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """Initialize all RAG components once"""
        if self._initialized:
            return

        # ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT
        )

        # Embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME
        )

        # Vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name="documents",
            embedding_function=self.embedding_function
        )

        # Retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        self._initialized = True
```

**Benefits**:
- Single initialization (performance)
- Shared across requests (memory efficiency)
- Connection pooling
- Lazy loading

### 2. ChromaDB Client

**Purpose**: Connect to ChromaDB vector database

**Configuration**:
```python
chroma_client = chromadb.HttpClient(
    host="localhost",  # or "chromadb" in Docker
    port=8001
)
```

**Collection**: `documents` (default)

### 3. Embedding Model

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

**Why This Model?**:
- Fast inference (suitable for real-time)
- Small size (~80MB)
- Good general-purpose quality
- 384-dimensional embeddings
- Free and open-source

**Initialization**:
```python
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use GPU if available
    encode_kwargs={'normalize_embeddings': True}
)
```

### 4. Vector Store

**Purpose**: Interface between application and ChromaDB

```python
vector_store = Chroma(
    client=chroma_client,
    collection_name="documents",
    embedding_function=embedding_function
)
```

**Operations**:
- `add_documents()` - Store new documents
- `similarity_search()` - Find similar chunks
- `similarity_search_with_score()` - With relevance scores
- `delete_collection()` - Clear all documents

### 5. Retriever

**Purpose**: Query interface for document retrieval

```python
retriever = vector_store.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={
        "k": 3,  # Number of results
        "score_threshold": 0.7  # Minimum relevance (optional)
    }
)
```

**Search Types**:
- `similarity` - Pure semantic similarity
- `mmr` - Maximum Marginal Relevance (diversity + relevance)

## Document Processing Pipeline

### Complete Upload Flow

**File**: `backend/services/upload.py:25-95`

```python
async def process_pdf(file: UploadFile) -> UploadResponse:
    """
    Complete PDF processing pipeline
    """

    # 1. Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files supported")

    # 2. Save temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 3. Extract text with PyPDF
    pdf_reader = PdfReader(temp_path)
    full_text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        full_text += f"\n\n--- Page {page_num + 1} ---\n\n"
        full_text += page.extract_text()

    # 4. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    # 5. Create documents with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": file.filename,
                "chunk_index": i,
                "page": estimate_page(chunk, full_text)
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    # 6. Generate embeddings and store
    rag = RAGComponents()
    rag.vector_store.add_documents(documents)

    # 7. Cleanup
    os.remove(temp_path)

    return UploadResponse(
        message="File processed successfully",
        filename=file.filename,
        chunks_added=len(documents)
    )
```

### Text Chunking Strategy

**Chunker**: `RecursiveCharacterTextSplitter`

**Parameters**:
```python
chunk_size = 1000      # Target characters per chunk
chunk_overlap = 200    # Overlap for context continuity
separators = [
    "\n\n",           # Paragraph breaks (preferred)
    "\n",             # Line breaks
    ". ",             # Sentence ends
    " ",              # Words
    ""                # Characters (fallback)
]
```

**Why Overlap?**
- Prevents context loss at boundaries
- Ensures important info not split awkwardly
- Improves retrieval relevance

**Example**:
```
Chunk 1: [... text ending with "The aircraft approaches V1 speed."]
         [200 chars overlap: "V1 speed. This is critical because..."]

Chunk 2: ["V1 speed. This is critical because..." continues...]
```

### Metadata Structure

Each chunk includes metadata for traceability:

```python
metadata = {
    "source": "aviation_manual.pdf",  # Original filename
    "chunk_index": 5,                  # Position in document
    "page": 12,                        # Estimated page number
    "uploaded_at": "2024-12-15T10:30:00Z"  # Timestamp
}
```

## Embedding Generation

### Process

```python
# Input text
text = "V1 is the critical engine failure recognition speed."

# Generate embedding
embedding = embedding_function.embed_query(text)

# Result: 384-dimensional vector
# [0.023, -0.145, 0.267, 0.089, ..., 0.156]  # 384 values
```

### Vector Properties

- **Dimensionality**: 384 (all-MiniLM-L6-v2)
- **Normalization**: Cosine normalized (unit vectors)
- **Distance Metric**: Cosine similarity
- **Range**: -1 (opposite) to 1 (identical)

### Similarity Calculation

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Calculate similarity between two vectors
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Example
query_embedding = [0.1, 0.2, 0.3, ...]
doc_embedding = [0.15, 0.18, 0.32, ...]

similarity = cosine_similarity(query_embedding, doc_embedding)
# Result: 0.95 (highly similar)
```

### Batch Processing

For multiple documents:

```python
# Batch embed for efficiency
texts = [chunk.page_content for chunk in documents]
embeddings = embedding_function.embed_documents(texts)

# Store with batch insert
vector_store.add_texts(
    texts=texts,
    embeddings=embeddings,
    metadatas=[chunk.metadata for chunk in documents]
)
```

## Vector Storage

### ChromaDB Collection Schema

```
Collection: "documents"
├── ID: Unique identifier (auto-generated)
├── Embedding: 384-dimensional vector
├── Document: Original text content
└── Metadata: JSON object
    ├── source: string
    ├── chunk_index: int
    ├── page: int
    └── uploaded_at: timestamp
```

### Storage Operations

#### Add Documents

```python
# Single document
vector_store.add_texts(
    texts=["content"],
    metadatas=[{"source": "file.pdf"}]
)

# Multiple documents
vector_store.add_documents([
    Document(page_content="...", metadata={...}),
    Document(page_content="...", metadata={...})
])
```

#### Query Documents

```python
# Similarity search
results = vector_store.similarity_search(
    query="What is V1 speed?",
    k=3  # Top 3 results
)

# With scores
results = vector_store.similarity_search_with_score(
    query="What is V1 speed?",
    k=3
)
# Returns: [(Document, score), (Document, score), ...]
```

#### Delete Documents

```python
# Delete by metadata filter
vector_store.delete(
    filter={"source": "old_document.pdf"}
)

# Clear entire collection
chroma_client.delete_collection("documents")
chroma_client.create_collection("documents")
```

### Persistence

ChromaDB data persists in Docker volume:

```yaml
# docker-compose.yml
volumes:
  - chroma_data:/chroma/chroma
```

**Location**: Named volume `chroma_data`

**Backup**:
```bash
# Export volume
docker run --rm -v chroma_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/chroma_backup.tar.gz /data

# Restore volume
docker run --rm -v chroma_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/chroma_backup.tar.gz -C /
```

## Retrieval Process

### Query Flow

**File**: `backend/services/chat.py:55-95`

```python
async def retrieve_context(query: str, k: int = 3) -> str:
    """
    Retrieve relevant context for query
    """

    # 1. Get RAG components
    rag = RAGComponents()

    # 2. Retrieve documents
    docs = rag.retriever.get_relevant_documents(query)

    # 3. Format context
    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(
            f"[Source: {doc.metadata['source']}, "
            f"Page: {doc.metadata.get('page', 'Unknown')}]\n"
            f"{doc.page_content}\n"
        )

    # 4. Join with separators
    context = "\n---\n".join(context_parts)

    return context
```

### Retrieval Parameters

**Configurable** in `backend/rag_components.py:85-95`:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,                    # Number of chunks to retrieve
        "score_threshold": 0.5,    # Minimum relevance score
        "filter": {"source": "..."}  # Metadata filtering (optional)
    }
)
```

### Search Strategies

#### 1. Similarity Search (Default)

Returns top-k most similar chunks:

```python
docs = retriever.get_relevant_documents("query")
# Returns most semantically similar chunks
```

#### 2. Maximum Marginal Relevance (MMR)

Balances relevance and diversity:

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,      # Fetch 10, select diverse 3
        "lambda_mult": 0.5  # 0=diversity, 1=relevance
    }
)
```

**When to Use**:
- Avoid redundant chunks
- Get broader context
- Multiple subtopics in query

#### 3. Similarity with Threshold

Only return highly relevant results:

```python
docs = vector_store.similarity_search_with_score(
    query="...",
    k=5
)

# Filter by score
filtered_docs = [
    doc for doc, score in docs
    if score >= 0.7
]
```

### Relevance Scoring

ChromaDB returns cosine similarity scores:

| Score Range | Interpretation |
|-------------|----------------|
| 0.9 - 1.0 | Highly relevant (near duplicate) |
| 0.7 - 0.9 | Very relevant |
| 0.5 - 0.7 | Moderately relevant |
| 0.3 - 0.5 | Somewhat relevant |
| 0.0 - 0.3 | Marginally relevant |

## Prompt Construction

### RAG Prompt Template

**File**: `backend/config.py:65-85`

```python
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, acknowledge this
- Cite sources when relevant (mention document name and page)
- Be concise but comprehensive
- If context is contradictory, note the discrepancies

Answer:
"""
```

### Prompt Construction Flow

```python
def build_rag_prompt(query: str, context: str) -> str:
    """
    Build final prompt with context
    """

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=query
    )

    return prompt
```

### Example

**Input**:
- Query: "What is V1 speed?"
- Context: (Retrieved from aviation manual)

**Final Prompt**:
```
You are a helpful assistant. Answer the question based on the provided context.

Context:
[Source: aviation_manual.pdf, Page: 12]
V1 is the critical engine failure recognition speed during takeoff. At speeds
below V1, the pilot should abort the takeoff if an engine fails. At or above
V1, the pilot should continue the takeoff even with an engine failure.

[Source: flight_ops_guide.pdf, Page: 5]
The calculation of V1 depends on aircraft weight, runway length, weather
conditions, and aircraft configuration. V1 must always be less than VR
(rotation speed) and is typically calculated before each flight.

Question: What is V1 speed?

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, acknowledge this
- Cite sources when relevant (mention document name and page)
- Be concise but comprehensive

Answer:
```

**LLM Response**:
```
Based on the aviation manual (page 12), V1 is the critical engine failure
recognition speed during takeoff. It's the decision speed where:
- Below V1: Abort takeoff if engine fails
- At or above V1: Continue takeoff even with engine failure

According to the flight operations guide (page 5), V1 is calculated before
each flight based on aircraft weight, runway length, weather conditions, and
configuration. It must always be less than VR (rotation speed).
```

## Configuration

### Environment Variables

**File**: `backend/config.py:15-60`

```python
# ChromaDB Connection
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

# Embedding Model
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# RAG Feature Toggle
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"

# Retrieval Settings
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Upload Settings
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
```

### Tuning Parameters

#### Chunk Size

| Chunk Size | Use Case | Pros | Cons |
|------------|----------|------|------|
| 500-750 | Short Q&A, facts | Precise retrieval | May lose context |
| 1000-1500 | General docs | Balanced | Default choice |
| 2000-3000 | Long-form content | More context | Less precise |

#### Overlap

| Overlap | Chunk Size | Ratio |
|---------|-----------|-------|
| 100 | 500 | 20% |
| 200 | 1000 | 20% (recommended) |
| 400 | 2000 | 20% |

**Rule of Thumb**: 15-25% overlap

#### Top-K

| K Value | Use Case |
|---------|----------|
| 1-2 | Specific factual queries |
| 3-5 | General questions (recommended) |
| 6-10 | Complex multi-faceted topics |

#### Score Threshold

| Threshold | Strictness |
|-----------|------------|
| 0.8-1.0 | Very strict (exact matches) |
| 0.6-0.8 | Moderate (recommended) |
| 0.4-0.6 | Lenient (broader results) |

## API Integration

### Enable RAG in Chat Request

```python
# Frontend request
{
    "query": "What is V1 speed?",
    "model": "gemma3:latest",
    "agent_name": "aviation_assistant",
    "use_rag": true,  # Enable RAG
    "history": []
}
```

### Backend Processing

**File**: `backend/services/chat.py:45-125`

```python
async def process_chat_with_rag(request: ChatRequest):
    """
    Process chat with RAG retrieval
    """

    # 1. Check if RAG enabled
    if not request.use_rag:
        # Direct LLM call without RAG
        return await process_chat_direct(request)

    # 2. Retrieve context
    rag = RAGComponents()
    docs = rag.retriever.get_relevant_documents(request.query)

    # 3. Format context
    context = "\n\n".join([
        f"[{doc.metadata['source']}]\n{doc.page_content}"
        for doc in docs
    ])

    # 4. Build RAG prompt
    prompt = Config.RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=request.query
    )

    # 5. Generate with context
    if Config.USE_GUARDRAILS and request.agent_name:
        response = await nemo_service.generate(
            agent_name=request.agent_name,
            message=prompt
        )
    else:
        response = await ollama_service.generate(
            model=request.model,
            prompt=prompt
        )

    return response
```

### Upload Document API

```python
# Upload PDF
POST /api/upload
Content-Type: multipart/form-data

{
    "file": <PDF file>
}

# Response
{
    "message": "File uploaded and processed successfully",
    "filename": "aviation_manual.pdf",
    "chunks_added": 15
}
```

### List Documents API

```python
# Get all documents
GET /api/documents

# Response
{
    "count": 2,
    "documents": [
        {
            "filename": "aviation_manual.pdf",
            "chunks": [
                {
                    "id": "...",
                    "content": "...",
                    "metadata": {...}
                }
            ]
        }
    ]
}
```

## Performance Optimization

### 1. Embedding Caching

Cache embeddings for repeated queries:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_query_embedding(query: str):
    """Cache query embeddings"""
    return embedding_function.embed_query(query)
```

### 2. Batch Processing

Process multiple documents efficiently:

```python
# Instead of loop
for doc in documents:
    vector_store.add_documents([doc])  # Slow

# Use batch
vector_store.add_documents(documents)  # Fast
```

### 3. GPU Acceleration

Use GPU for embeddings if available:

```python
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}  # Use GPU
)
```

### 4. Index Optimization

ChromaDB automatically creates indexes, but you can optimize:

```python
# Create collection with specific distance metric
collection = chroma_client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # Optimized for cosine similarity
)
```

### 5. Async Operations

Use async for I/O operations:

```python
async def process_upload_async(file: UploadFile):
    """Async processing"""
    content = await file.read()
    # Process in background
    asyncio.create_task(embed_and_store(content))
```

### Performance Metrics

| Operation | Time (CPU) | Time (GPU) |
|-----------|-----------|-----------|
| Single embedding | 50ms | 10ms |
| Batch 100 embeddings | 2s | 200ms |
| ChromaDB insert (100 docs) | 500ms | 500ms |
| Similarity search | 20ms | 20ms |
| PDF parsing (10 pages) | 1s | 1s |

## Troubleshooting

### ChromaDB Connection Failed

**Symptom**: `Cannot connect to ChromaDB`

**Solutions**:
```bash
# 1. Check ChromaDB is running
curl http://localhost:8001/api/v1/heartbeat

# 2. Check Docker container
docker ps | grep chroma

# 3. Restart ChromaDB
docker-compose restart chromadb

# 4. Check logs
docker logs chromadb
```

### Embedding Model Download Fails

**Symptom**: `Cannot download model`

**Solutions**:
```python
# Pre-download model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('/path/to/local/model')

# Use local model
embedding_function = HuggingFaceEmbeddings(
    model_name='/path/to/local/model'
)
```

### Poor Retrieval Quality

**Symptom**: Retrieved chunks not relevant

**Debug**:
```python
# Check similarity scores
results = vector_store.similarity_search_with_score(query, k=5)
for doc, score in results:
    print(f"Score: {score}, Content: {doc.page_content[:100]}")

# If all scores < 0.5, possible issues:
# 1. No relevant documents uploaded
# 2. Query phrasing mismatch
# 3. Wrong embedding model
```

**Solutions**:
1. Upload more relevant documents
2. Rephrase query to match document language
3. Increase `k` value
4. Lower `score_threshold`

### Out of Memory

**Symptom**: `OutOfMemoryError` during embedding

**Solutions**:
```python
# 1. Process in smaller batches
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vector_store.add_documents(batch)

# 2. Use smaller embedding model
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"  # Smaller
)

# 3. Reduce chunk size
chunk_size = 500  # Instead of 1000
```

### Slow Retrieval

**Symptom**: Queries take too long

**Optimizations**:
```python
# 1. Reduce k value
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}  # Instead of 5
)

# 2. Add score threshold (early termination)
search_kwargs={"k": 5, "score_threshold": 0.7}

# 3. Use GPU for embeddings
model_kwargs={'device': 'cuda'}
```

## Advanced Topics

### Custom Embedding Models

Use domain-specific models:

```python
# Legal domain
embedding_function = HuggingFaceEmbeddings(
    model_name="nlpaueb/legal-bert-base-uncased"
)

# Scientific domain
embedding_function = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased"
)
```

### Hybrid Search

Combine semantic and keyword search:

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# Semantic retriever
semantic = vector_store.as_retriever()

# Keyword retriever
keyword = BM25Retriever.from_documents(documents)

# Combine
ensemble = EnsembleRetriever(
    retrievers=[semantic, keyword],
    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
)
```

### Metadata Filtering

Filter by document properties:

```python
# Only search in specific documents
results = vector_store.similarity_search(
    query="...",
    k=3,
    filter={"source": "aviation_manual.pdf"}
)

# Filter by date range
filter={
    "uploaded_at": {"$gte": "2024-12-01", "$lt": "2024-12-31"}
}
```

## Related Documentation

- [Architecture Overview](./ARCHITECTURE.md) - RAG system architecture
- [API Reference](./API-REFERENCE.md) - Upload and document endpoints
- [Backend Guide](./BACKEND-GUIDE.md) - RAG implementation details
- [Development Guide](./DEVELOPMENT.md) - Testing RAG locally
