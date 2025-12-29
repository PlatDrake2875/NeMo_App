"""
ColBERT retriever implementation using RAGatouille for IR benchmarking.

ColBERT (Contextualized Late Interaction over BERT) is a neural retrieval model
that uses "late interaction" - it encodes queries and documents separately,
then computes fine-grained similarity at search time. This provides:

1. Better retrieval quality than dense embeddings (like all-MiniLM-L6-v2)
2. More efficient than cross-encoders (which require encoding query+doc together)
3. Ideal for reranking candidates from a first-stage retriever (PGVector)

In the NeMo RAG application, ColBERT serves two purposes:
- PRIMARY: Reranker after PGVector retrieval (two-stage retrieval)
- SECONDARY: Standalone retriever for IR benchmarking comparisons
"""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from ragatouille import RAGPretrainedModel
    RAGATOUILLE_AVAILABLE = True
except ImportError:
    RAGATOUILLE_AVAILABLE = False
    RAGPretrainedModel = None

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Thread pool for running sync ColBERT operations in async context
_executor = ThreadPoolExecutor(max_workers=2)


class ColBERTRetriever:
    """
    ColBERT-based retriever for late interaction retrieval and reranking.

    This class wraps RAGatouille's RAGPretrainedModel to provide:
    - Document indexing with ColBERT embeddings
    - Semantic search using late interaction
    - Reranking of candidate documents (primary use case in NeMo)
    - Index persistence and loading
    - Incremental index updates

    Integration with NeMo RAG:
    --------------------------
    The typical flow in our RAG pipeline is:
    1. User query comes in via /chat endpoint
    2. PGVector retrieves top-20 candidates (fast, approximate)
    3. ColBERT reranks to top-5 (accurate, late interaction)
    4. Top-5 documents are used as context for LLM

    This two-stage approach balances speed and accuracy.
    """

    def __init__(
        self,
        index_name: str = "colbert_index",
        model_name: str = "colbert-ir/colbertv2.0",
        index_root: Optional[str] = None,
        n_gpu: int = -1
    ):
        """
        Initialize ColBERT retriever.

        Args:
            index_name: Name for the ColBERT index (used for persistence)
            model_name: HuggingFace model ID for ColBERT
                       Default "colbert-ir/colbertv2.0" is the standard pretrained model
            index_root: Directory to store ColBERT indices
                       Defaults to ".ragatouille" in current directory
            n_gpu: Number of GPUs to use (-1 = all available, 0 = CPU only)

        Note:
            The model is loaded lazily on first use to avoid slow startup.
            This is important for the NeMo app where ColBERT might not be
            needed for every request.
        """
        if not RAGATOUILLE_AVAILABLE:
            raise ImportError(
                "RAGatouille not installed. Install with: pip install ragatouille"
            )

        self.index_name = index_name
        self.model_name = model_name
        self.index_root = index_root or ".ragatouille"
        self.n_gpu = n_gpu
        self.model: Optional[RAGPretrainedModel] = None
        self.index_path: Optional[str] = None
        self._indexed = False
        self._encoded_docs: Optional[Any] = None

        logger.info(f"Initializing ColBERT retriever with model: {model_name}")

    def _load_model(self) -> None:
        """
        Load the ColBERT model if not already loaded.

        This uses lazy loading - the model is only loaded when first needed.
        This is important because:
        1. ColBERT models are large (~500MB)
        2. Loading takes several seconds
        3. Not all requests need ColBERT (e.g., when RAG is disabled)
        """
        if self.model is None:
            logger.info(f"Loading ColBERT model: {self.model_name}")
            self.model = RAGPretrainedModel.from_pretrained(
                self.model_name,
                n_gpu=self.n_gpu,
                verbose=1
            )
            logger.info("ColBERT model loaded successfully")

    def load_index(self, index_path: str) -> None:
        """
        Load an existing ColBERT index from disk.

        This is crucial for production deployments where:
        1. Index is built once during document ingestion
        2. Index is loaded on server startup
        3. Multiple workers share the same index

        Args:
            index_path: Path to the ColBERT index directory

        Example:
            retriever = ColBERTRetriever()
            retriever.load_index(".ragatouille/colbert/indexes/my_index")

        In NeMo app context:
            When a processed dataset is loaded, we load its ColBERT index
            if it was built with ColBERT embeddings.
        """
        logger.info(f"Loading ColBERT index from: {index_path}")
        self.model = RAGPretrainedModel.from_index(
            index_path,
            n_gpu=self.n_gpu,
            verbose=1
        )
        self.index_path = index_path
        self._indexed = True
        logger.info("ColBERT index loaded successfully")

    def index_documents(
        self,
        documents: List[Document],
        max_document_length: int = 256,
        split_documents: bool = True,
        overwrite: bool = True
    ) -> str:
        """
        Index documents using ColBERT for later retrieval.

        This creates a ColBERT index that enables fast semantic search.
        The index stores:
        - Token-level embeddings for each document
        - Document IDs and metadata
        - Compressed representations for efficiency

        Args:
            documents: List of LangChain Document objects to index
            max_document_length: Maximum tokens per document (default 256)
                                Longer documents are truncated
            split_documents: Whether to split long documents into chunks
                            Recommended True for better retrieval
            overwrite: Whether to overwrite existing index with same name

        Returns:
            Path to the created index

        In NeMo app context:
            This is called when processing a dataset in the RAG Benchmark Hub.
            The index is stored persistently so it can be loaded later.
        """
        self._load_model()

        # Extract text and metadata from LangChain documents
        texts = [doc.page_content for doc in documents]

        # Generate document IDs - use metadata id if available, else index
        document_ids = []
        for i, doc in enumerate(documents):
            doc_id = doc.metadata.get("id") or doc.metadata.get("doc_id") or str(i)
            document_ids.append(str(doc_id))

        document_metadatas = [doc.metadata for doc in documents]

        logger.info(f"Indexing {len(documents)} documents with ColBERT...")
        logger.info(f"Settings: max_length={max_document_length}, split={split_documents}")

        self.index_path = self.model.index(
            collection=texts,
            index_name=self.index_name,
            document_ids=document_ids,
            document_metadatas=document_metadatas,
            max_document_length=max_document_length,
            split_documents=split_documents,
            overwrite_index=overwrite
        )

        self._indexed = True
        logger.info(f"Successfully indexed documents. Index path: {self.index_path}")

        return self.index_path

    def add_documents(
        self,
        documents: List[Document],
        max_document_length: int = 256,
        split_documents: bool = True
    ) -> None:
        """
        Add documents to an existing ColBERT index.

        This enables incremental indexing - you can add new documents
        without rebuilding the entire index. Useful for:
        1. Adding new documents to a dataset
        2. Real-time document ingestion
        3. Updating the knowledge base

        Args:
            documents: New documents to add
            max_document_length: Maximum tokens per document
            split_documents: Whether to split long documents

        Raises:
            ValueError: If no index is loaded

        In NeMo app context:
            When users upload additional files to an existing dataset,
            we can add them to the ColBERT index incrementally.
        """
        if not self._indexed:
            raise ValueError(
                "No index loaded. Call index_documents() or load_index() first."
            )

        texts = [doc.page_content for doc in documents]
        document_ids = []
        for i, doc in enumerate(documents):
            doc_id = doc.metadata.get("id") or doc.metadata.get("doc_id") or f"new_{i}"
            document_ids.append(str(doc_id))

        document_metadatas = [doc.metadata for doc in documents]

        logger.info(f"Adding {len(documents)} documents to existing index...")

        self.model.add_to_index(
            new_collection=texts,
            new_document_ids=document_ids,
            new_document_metadatas=document_metadatas,
            index_name=self.index_name,
            max_document_length=max_document_length,
            split_documents=split_documents
        )

        logger.info(f"Successfully added {len(documents)} documents to index")

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Remove documents from the ColBERT index.

        This enables index maintenance - removing outdated or
        incorrect documents without full reindexing.

        Args:
            document_ids: List of document IDs to remove

        Raises:
            ValueError: If no index is loaded

        In NeMo app context:
            When users delete files from a dataset, we remove
            them from the ColBERT index as well.
        """
        if not self._indexed:
            raise ValueError(
                "No index loaded. Call index_documents() or load_index() first."
            )

        logger.info(f"Deleting {len(document_ids)} documents from index...")

        self.model.delete_from_index(
            document_ids=document_ids,
            index_name=self.index_name
        )

        logger.info(f"Successfully deleted {len(document_ids)} documents")

    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search the indexed documents using ColBERT late interaction.

        This performs semantic search by:
        1. Encoding the query into token embeddings
        2. Computing MaxSim scores against all document embeddings
        3. Returning top-k documents by score

        Args:
            query: The search query
            k: Number of results to return (default 5)

        Returns:
            List of (Document, score) tuples, sorted by relevance

        Raises:
            ValueError: If no index is loaded

        In NeMo app context:
            This is used for standalone ColBERT retrieval in benchmarking.
            For production RAG, prefer the two-stage approach with rerank().
        """
        if not self._indexed:
            raise ValueError(
                "No index loaded. Call index_documents() or load_index() first."
            )

        logger.debug(f"Searching for: '{query}' (top-{k})")

        results = self.model.search(query, k=k)

        return self._format_results(results)

    def search_batch(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[List[Tuple[Document, float]]]:
        """
        Search for multiple queries in batch.

        More efficient than multiple single searches when you have
        multiple queries to process.

        Args:
            queries: List of search queries
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        if not self._indexed:
            raise ValueError(
                "No index loaded. Call index_documents() or load_index() first."
            )

        logger.debug(f"Batch searching {len(queries)} queries (top-{k} each)")

        results = self.model.search(queries, k=k)

        # Results for multiple queries is a list of lists
        return [self._format_results(query_results) for query_results in results]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = 5,
        bsize: int = 64
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using ColBERT without requiring an index.

        THIS IS THE PRIMARY USE CASE FOR COLBERT IN NEMO RAG.

        Reranking works by:
        1. Encoding the query with ColBERT
        2. Encoding each candidate document with ColBERT
        3. Computing late interaction scores (MaxSim)
        4. Returning documents sorted by score

        This is faster than indexing because:
        - No index construction needed
        - Only encodes the candidate documents (not entire corpus)
        - Ideal for reranking 10-50 candidates from first-stage retrieval

        Args:
            query: The search query
            documents: Candidate documents to rerank (from PGVector)
            k: Number of top documents to return
            bsize: Batch size for encoding (higher = faster but more memory)

        Returns:
            Top-k documents sorted by ColBERT relevance score

        Example in NeMo RAG pipeline:
            # First stage: PGVector retrieves 20 candidates (fast)
            candidates = pgvector_retriever.invoke(query)[:20]

            # Second stage: ColBERT reranks to top 5 (accurate)
            reranked = colbert.rerank(query, candidates, k=5)

            # Use reranked docs as context for LLM
            context = format_docs(reranked)
        """
        self._load_model()

        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        # Extract text from documents
        texts = [doc.page_content for doc in documents]

        logger.debug(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")

        # RAGatouille's rerank method
        results = self.model.rerank(
            query=query,
            documents=texts,
            k=min(k, len(documents)),  # Can't return more than we have
            bsize=bsize
        )

        # Map results back to original documents with scores
        reranked = []
        for result in results:
            # Find the original document by content
            content = result["content"]
            score = result["score"]

            # Find matching document to preserve metadata
            for doc in documents:
                if doc.page_content == content:
                    # Create new document with score in metadata
                    reranked_doc = Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "colbert_score": score,
                            "colbert_rank": result.get("rank", len(reranked) + 1)
                        }
                    )
                    reranked.append((reranked_doc, score))
                    break
            else:
                # Document not found (shouldn't happen), create new one
                reranked_doc = Document(
                    page_content=content,
                    metadata={
                        "colbert_score": score,
                        "colbert_rank": result.get("rank", len(reranked) + 1)
                    }
                )
                reranked.append((reranked_doc, score))

        logger.debug(f"Reranking complete. Top score: {reranked[0][1] if reranked else 'N/A'}")

        return reranked

    async def arerank(
        self,
        query: str,
        documents: List[Document],
        k: int = 5,
        bsize: int = 64
    ) -> List[Tuple[Document, float]]:
        """
        Async version of rerank for FastAPI compatibility.

        ColBERT operations are CPU/GPU bound and synchronous.
        This wrapper runs rerank in a thread pool to avoid
        blocking the FastAPI event loop.

        Args:
            query: The search query
            documents: Candidate documents to rerank
            k: Number of top documents to return
            bsize: Batch size for encoding

        Returns:
            Top-k documents sorted by ColBERT relevance score

        In NeMo app context:
            Use this in async FastAPI endpoints:

            @router.post("/chat")
            async def chat(request: ChatRequest):
                candidates = await pgvector.ainvoke(query)
                reranked = await colbert.arerank(query, candidates, k=5)
                ...
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.rerank(query, documents, k, bsize)
        )

    async def asearch(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Async version of search for FastAPI compatibility.

        Runs the synchronous search in a thread pool.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.search(query, k)
        )

    def encode_documents(
        self,
        documents: List[Document],
        bsize: int = 32,
        max_document_length: int = 256
    ) -> None:
        """
        Pre-encode documents for fast in-memory searching.

        This is useful when you want to:
        1. Search the same document set multiple times
        2. Avoid re-encoding documents for each query
        3. Have a smaller document set that fits in memory

        Args:
            documents: Documents to encode
            bsize: Batch size for encoding
            max_document_length: Maximum tokens per document

        After calling this, use search_encoded() instead of rerank().
        """
        self._load_model()

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        logger.info(f"Encoding {len(documents)} documents for in-memory search...")

        self.model.encode(
            documents=texts,
            document_metadatas=metadatas,
            bsize=bsize,
            max_document_length=max_document_length
        )

        self._encoded_docs = documents  # Store reference for later
        logger.info("Documents encoded successfully")

    def search_encoded(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search pre-encoded documents (faster than rerank for repeated searches).

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            Top-k documents sorted by score

        Raises:
            ValueError: If no documents have been encoded
        """
        if self._encoded_docs is None:
            raise ValueError(
                "No encoded documents. Call encode_documents() first."
            )

        results = self.model.search_encoded_docs(query=query, k=k)
        return self._format_results(results)

    def clear_encoded_documents(self) -> None:
        """
        Clear pre-encoded documents from memory.

        Call this when you're done with in-memory search to free memory.
        """
        if self.model is not None:
            self.model.clear_encoded_docs(force=True)
        self._encoded_docs = None
        logger.info("Cleared encoded documents from memory")

    def _format_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Tuple[Document, float]]:
        """
        Convert RAGatouille results to LangChain Document format.

        This ensures compatibility with the rest of the NeMo RAG pipeline,
        which uses LangChain Document objects throughout.
        """
        documents_with_scores = []

        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    **result.get("document_metadata", {}),
                    "colbert_score": result["score"],
                    "colbert_rank": result["rank"],
                    "document_id": result.get("document_id")
                }
            )
            documents_with_scores.append((doc, result["score"]))

        return documents_with_scores

    def is_available(self) -> bool:
        """Check if ColBERT retriever is ready for use."""
        return RAGATOUILLE_AVAILABLE

    def has_index(self) -> bool:
        """Check if an index is loaded."""
        return self._indexed

    def get_index_path(self) -> Optional[str]:
        """Get the path to the current index."""
        return self.index_path


# Convenience function for checking RAGatouille availability
def is_ragatouille_available() -> bool:
    """Check if RAGatouille library is installed."""
    return RAGATOUILLE_AVAILABLE
