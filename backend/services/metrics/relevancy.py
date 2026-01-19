"""
Response Relevancy Metric - Measures how relevant the answer is to the question.

Uses embedding-based semantic similarity between question and answer
to determine if the answer addresses the question.
"""

import logging
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseMetric, MetricResult
from .registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("relevancy")
class RelevancyMetric(BaseMetric):
    """
    Measures how relevant the generated answer is to the question.

    Uses cosine similarity between question and answer embeddings.
    High score means the answer directly addresses the question.
    """

    name = "relevancy"
    description = "Measures answer relevance to the question using embedding similarity"
    requires_context = False
    requires_reference = False

    def __init__(self, embedder_model: str = "all-MiniLM-L6-v2"):
        self._embedder = None
        self._embedder_model = embedder_model

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy load embedder to avoid startup delay."""
        if self._embedder is None:
            logger.info(f"Loading embedder model: {self._embedder_model}")
            self._embedder = SentenceTransformer(self._embedder_model)
        return self._embedder

    async def compute(
        self,
        question: str,
        generated_answer: str,
        reference_answer: Optional[str] = None,
        retrieved_chunks: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute response relevancy score.

        Returns score between 0 and 1 where:
        - 1.0 = answer is highly relevant to the question
        - 0.0 = answer is completely unrelated to the question
        """
        if not question or not generated_answer:
            return MetricResult(
                score=0.0,
                details={"error": "Missing question or answer"},
            )

        similarity = self._cosine_similarity(question, generated_answer)

        return MetricResult(
            score=similarity,
            details={
                "embedding_model": self._embedder_model,
                "question_length": len(question),
                "answer_length": len(generated_answer),
            },
        )

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using embeddings."""
        if not text1 or not text2:
            return 0.0

        embeddings = self.embedder.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(similarity / (norm1 * norm2))
