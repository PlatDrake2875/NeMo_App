"""
Precision@K and Recall@K Metrics - Measure retrieval quality.

Precision@K = relevant_chunks_in_top_K / K
Recall@K = relevant_chunks_in_top_K / relevant_chunks_in_expanded_set

Both metrics reuse the same LLM-based relevance judgments as Context Precision,
avoiding duplicate LLM calls when all three metrics are computed together.
"""

import logging
from typing import Any, Optional

from services.openrouter_client import OpenRouterClient

from .base import BaseMetric, MetricResult
from .registry import register_metric

logger = logging.getLogger(__name__)


async def judge_chunk_relevance(
    llm: OpenRouterClient,
    llm_model: str,
    query: str,
    chunk: str,
    ground_truth: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Judge if a chunk is relevant using LLM.

    Shared utility used by Context Precision, Precision@K, and Recall@K.

    Args:
        llm: OpenRouter client instance
        llm_model: Model to use for judging
        query: The question being asked
        chunk: The text chunk to evaluate
        ground_truth: Optional ground truth answer for more accurate evaluation

    Returns:
        Tuple of (is_relevant, reason)
    """
    from .context_precision import RELEVANCE_JUDGMENT_PROMPT, GROUND_TRUTH_RELEVANCE_PROMPT

    try:
        if ground_truth:
            prompt = GROUND_TRUTH_RELEVANCE_PROMPT.format(
                question=query,
                ground_truth=ground_truth[:1000],
                chunk=chunk[:2000],
            )
        else:
            prompt = RELEVANCE_JUDGMENT_PROMPT.format(
                question=query,
                chunk=chunk[:2000],
            )

        result = await llm.generate_json(
            prompt=prompt,
            model=llm_model,
            temperature=0.1,
            max_tokens=128,
        )

        return result.get("relevant", False), result.get("reason", "")
    except Exception as e:
        logger.warning(f"Failed to judge chunk relevance: {e}")
        reference_text = ground_truth if ground_truth else query
        reference_words = set(reference_text.lower().split())
        chunk_words = set(chunk.lower().split())
        overlap = len(reference_words & chunk_words) / len(reference_words) if reference_words else 0
        return overlap > 0.3, "Fallback: keyword overlap"


@register_metric("precision_at_k")
class PrecisionAtK(BaseMetric):
    """
    Measures Precision@K: fraction of top-K retrieved chunks that are relevant.

    Precision@K = relevant_chunks_in_top_K / K
    """

    name = "precision_at_k"
    description = "Fraction of top-K retrieved chunks that are relevant"
    requires_context = True
    requires_reference = False

    def __init__(
        self,
        llm_client: Optional[OpenRouterClient] = None,
        llm_model: str = "openai/gpt-4o-mini",
    ):
        self.llm = llm_client or OpenRouterClient()
        self.llm_model = llm_model

    async def compute(
        self,
        question: str,
        generated_answer: str,
        reference_answer: Optional[str] = None,
        retrieved_chunks: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute Precision@K.

        Accepts optional pre-computed chunk_relevances to avoid duplicate LLM calls
        when Context Precision has already judged the chunks.
        """
        if not retrieved_chunks:
            return MetricResult(
                score=0.0,
                details={"error": "No chunks provided", "k": 0, "relevant_count": 0},
            )

        k = len(retrieved_chunks)

        # Use pre-computed relevances if provided (avoids duplicate LLM calls)
        relevances = kwargs.get("chunk_relevances")
        if relevances is None:
            relevances = []
            for chunk in retrieved_chunks:
                is_relevant, _ = await judge_chunk_relevance(
                    self.llm, self.llm_model, question, chunk,
                    ground_truth=reference_answer,
                )
                relevances.append(is_relevant)

        relevant_count = sum(bool(r) for r in relevances[:k])
        score = relevant_count / k if k > 0 else 0.0

        return MetricResult(
            score=score,
            details={
                "k": k,
                "relevant_count": relevant_count,
                "chunk_relevances": relevances[:k],
            },
        )


@register_metric("recall_at_k")
class RecallAtK(BaseMetric):
    """
    Measures Recall@K: fraction of all relevant chunks (in an expanded set)
    that appear in the top-K.

    Recall@K = relevant_chunks_in_top_K / relevant_chunks_in_expanded_set

    The expanded set is typically 3*K chunks retrieved with a larger K.
    If no expanded set is provided, falls back to using only the top-K chunks.
    """

    name = "recall_at_k"
    description = "Fraction of relevant chunks captured in top-K vs expanded set"
    requires_context = True
    requires_reference = False

    def __init__(
        self,
        llm_client: Optional[OpenRouterClient] = None,
        llm_model: str = "openai/gpt-4o-mini",
    ):
        self.llm = llm_client or OpenRouterClient()
        self.llm_model = llm_model

    async def compute(
        self,
        question: str,
        generated_answer: str,
        reference_answer: Optional[str] = None,
        retrieved_chunks: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute Recall@K.

        Args (via kwargs):
            expanded_chunks: List of 3*K chunks for computing the denominator.
                If not provided, uses retrieved_chunks as both top-K and expanded set.
            chunk_relevances: Pre-computed relevances for top-K chunks.
            expanded_relevances: Pre-computed relevances for the full expanded set.
        """
        expanded_chunks = kwargs.get("expanded_chunks", retrieved_chunks)

        if not retrieved_chunks or not expanded_chunks:
            return MetricResult(
                score=0.0,
                details={"error": "No chunks provided", "k": 0},
            )

        k = len(retrieved_chunks)

        # Use pre-computed relevances if provided
        top_k_relevances = kwargs.get("chunk_relevances")
        expanded_relevances = kwargs.get("expanded_relevances")

        if top_k_relevances is None:
            top_k_relevances = []
            for chunk in retrieved_chunks:
                is_relevant, _ = await judge_chunk_relevance(
                    self.llm, self.llm_model, question, chunk,
                    ground_truth=reference_answer,
                )
                top_k_relevances.append(is_relevant)

        if expanded_relevances is None:
            # Start with top-K relevances, then judge remaining expanded chunks
            expanded_relevances = list(top_k_relevances)
            for chunk in expanded_chunks[k:]:
                is_relevant, _ = await judge_chunk_relevance(
                    self.llm, self.llm_model, question, chunk,
                    ground_truth=reference_answer,
                )
                expanded_relevances.append(is_relevant)

        relevant_in_top_k = sum(bool(r) for r in top_k_relevances[:k])
        relevant_in_expanded = sum(bool(r) for r in expanded_relevances)

        score = relevant_in_top_k / relevant_in_expanded if relevant_in_expanded > 0 else 0.0

        return MetricResult(
            score=score,
            details={
                "k": k,
                "expanded_size": len(expanded_chunks),
                "relevant_in_top_k": relevant_in_top_k,
                "relevant_in_expanded": relevant_in_expanded,
            },
        )
