"""
Answer Correctness Metric - Measures how correct the generated answer is.

Based on RAGAS answer correctness metric:
- Extracts claims from both predicted and ground truth
- Calculates F1 score based on claim overlap
- Combines with semantic similarity
- Final score = weighted average
"""

import asyncio
import logging
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from services.openrouter_client import OpenRouterClient

from .base import BaseMetric, MetricResult
from .registry import register_metric

logger = logging.getLogger(__name__)


CLAIM_EXTRACTION_PROMPT = """You are an expert at extracting factual claims from text.

Given the following text, extract all atomic factual claims. Each claim should be:
- A single, verifiable statement
- Self-contained (understandable without context)
- Factual (not opinions or questions)

Text:
{text}

Return a JSON object with a "claims" array containing the extracted claims as strings.
Example: {{"claims": ["Claim 1", "Claim 2", "Claim 3"]}}

Extract the claims:"""


@register_metric("answer_correctness")
class AnswerCorrectnessMetric(BaseMetric):
    """
    Measures how correct the generated answer is compared to ground truth.

    Combines:
    - Factual F1: Claim overlap between predicted and ground truth
    - Semantic similarity: Embedding-based similarity
    """

    name = "answer_correctness"
    description = "Measures answer correctness using F1 factual + semantic similarity"
    requires_context = False
    requires_reference = True

    def __init__(
        self,
        llm_client: Optional[OpenRouterClient] = None,
        llm_model: str = "openai/gpt-4o-mini",
        embedder_model: str = "all-MiniLM-L6-v2",
        factual_weight: float = 0.5,
    ):
        self.llm = llm_client or OpenRouterClient()
        self.llm_model = llm_model
        self._embedder = None
        self._embedder_model = embedder_model
        self.factual_weight = factual_weight

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
        Compute answer correctness score.

        Returns score between 0 and 1 where:
        - 1.0 = perfect match with ground truth
        - 0.0 = no overlap with ground truth
        """
        if not reference_answer:
            return MetricResult(
                score=0.0,
                details={"error": "No reference answer provided"},
            )

        # Extract claims from both texts
        predicted_claims, ground_truth_claims = await asyncio.gather(
            self._extract_claims(generated_answer),
            self._extract_claims(reference_answer),
        )

        # Calculate factual F1 score
        if not predicted_claims and not ground_truth_claims:
            tp, fp, fn = 0, 0, 0
            factual_score = 1.0
        elif not predicted_claims or not ground_truth_claims:
            tp = 0
            fp = len(predicted_claims)
            fn = len(ground_truth_claims)
            factual_score = 0.0
        else:
            tp, fp, fn = await self._compare_claims(predicted_claims, ground_truth_claims)
            # Standard F1 formula: 2*TP / (2*TP + FP + FN)
            denominator = 2 * tp + fp + fn
            if denominator == 0:
                factual_score = 0.0
            else:
                factual_score = (2 * tp) / denominator

        # Calculate semantic similarity
        semantic_score = self._cosine_similarity(generated_answer, reference_answer)

        # Weighted average
        final_score = self.factual_weight * factual_score + (1 - self.factual_weight) * semantic_score

        return MetricResult(
            score=final_score,
            details={
                "factual_score": factual_score,
                "semantic_score": semantic_score,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "predicted_claims_count": len(predicted_claims),
                "ground_truth_claims_count": len(ground_truth_claims),
            },
        )

    async def _extract_claims(self, text: str) -> list[str]:
        """Extract atomic factual claims from text using LLM."""
        if not text or not text.strip():
            return []

        try:
            prompt = CLAIM_EXTRACTION_PROMPT.format(text=text[:3000])

            result = await self.llm.generate_json(
                prompt=prompt,
                model=self.llm_model,
                temperature=0.1,
                max_tokens=1024,
            )

            claims = result.get("claims", [])
            return [c for c in claims if isinstance(c, str) and c.strip()]
        except Exception as e:
            logger.warning(f"Failed to extract claims: {e}")
            return [s.strip() for s in text.split(".") if s.strip() and len(s.strip()) > 10]

    async def _compare_claims(
        self,
        predicted_claims: list[str],
        ground_truth_claims: list[str],
        threshold: float = 0.7,
    ) -> tuple[int, int, int]:
        """
        Compare two sets of claims to find TP, FP, FN.
        Uses semantic similarity to match claims.
        """
        all_claims = predicted_claims + ground_truth_claims
        embeddings = self.embedder.encode(all_claims)

        pred_embeddings = embeddings[: len(predicted_claims)]
        truth_embeddings = embeddings[len(predicted_claims) :]

        # Calculate similarity matrix
        similarity_matrix = np.dot(pred_embeddings, truth_embeddings.T)

        # Find matches above threshold
        matched_truth = set()
        matched_pred = set()

        flat_indices = np.argsort(similarity_matrix.flatten())[::-1]

        for flat_idx in flat_indices:
            pred_idx = flat_idx // len(ground_truth_claims)
            truth_idx = flat_idx % len(ground_truth_claims)

            if similarity_matrix[pred_idx, truth_idx] < threshold:
                break

            if pred_idx not in matched_pred and truth_idx not in matched_truth:
                matched_pred.add(pred_idx)
                matched_truth.add(truth_idx)

        tp = len(matched_pred)
        fp = len(predicted_claims) - tp
        fn = len(ground_truth_claims) - tp

        return tp, fp, fn

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
