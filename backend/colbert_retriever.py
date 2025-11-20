"""ColBERT retriever implementation using RAGatouille for IR benchmarking."""
from typing import List, Tuple, Optional
from pathlib import Path
import logging

try:
    from ragatouille import RAGPretrainedModel
except ImportError:
    raise ImportError(
        "RAGatouille not installed. Install with: "
        "pip install ragatouille --break-system-packages"
    )

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ColBERTRetriever:
    """ColBERT-based retriever for late interaction retrieval."""
    
    def __init__(
        self,
        index_name: str = "colbert_index",
        model_name: str = "colbert-ir/colbertv2.0",
        index_root: Optional[str] = None
    ):
        """Initialize ColBERT retriever."""
        self.index_name = index_name
        self.model_name = model_name
        self.index_root = index_root or ".ragatouille"
        self.model: Optional[RAGPretrainedModel] = None
        self.index_path: Optional[str] = None
        self._indexed = False
        
        logger.info(f"Initializing ColBERT retriever with model: {model_name}")
    
    def _load_model(self) -> None:
        """Load the ColBERT model if not already loaded."""
        if self.model is None:
            logger.info(f"Loading ColBERT model: {self.model_name}")
            self.model = RAGPretrainedModel.from_pretrained(self.model_name)
            logger.info("ColBERT model loaded successfully")
    
    def index_documents(
        self,
        documents: List[Document],
        max_document_length: int = 512,
        split_documents: bool = True
    ) -> str:
        """Index documents using ColBERT."""
        self._load_model()
        
        texts = [doc.page_content for doc in documents]
        document_ids = [str(i) for i in range(len(documents))]
        document_metadatas = [doc.metadata for doc in documents]
        
        logger.info(f"Indexing {len(documents)} documents with ColBERT...")
        
        self.index_path = self.model.index(
            collection=texts,
            index_name=self.index_name,
            document_ids=document_ids,
            document_metadatas=document_metadatas,
            max_document_length=max_document_length,
            split_documents=split_documents
        )
        
        self._indexed = True
        logger.info(f"Successfully indexed documents. Index path: {self.index_path}")
        
        return self.index_path
    
    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Search using ColBERT late interaction."""
        if not self._indexed:
            raise ValueError(
                "No index loaded. Call index_documents() first."
            )
        
        logger.debug(f"Searching for: '{query}' (top-{k})")
        
        results = self.model.search(query, k=k)
        
        documents_with_scores = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    **result.get("document_metadata", {}),
                    "score": result["score"],
                    "rank": result["rank"],
                }
            )
            documents_with_scores.append((doc, result["score"]))
        
        return documents_with_scores
