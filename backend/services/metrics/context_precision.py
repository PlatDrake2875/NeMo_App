"""
Context Precision Metric - Measures retrieval quality.

Based on RAGAS context precision metric:
- Judges relevance of each retrieved chunk to the query
- Calculates mean precision@k weighted by position
- Earlier irrelevant chunks hurt the score more
"""

import logging
from typing import Any, Optional

from services.openrouter_client import OpenRouterClient

from .base import BaseMetric, MetricResult
from .registry import register_metric

logger = logging.getLogger(__name__)


RELEVANCE_JUDGMENT_PROMPT = """You are an expert at judging the relevance of text chunks for answering questions.

Given a question and a text chunk, determine if the chunk contains information relevant to answering the question.

Question: {question}

Chunk:
{chunk}

Return a JSON object with:
- "relevant": true/false
- "reason": brief explanation of why it is or isn't relevant

Example: {{"relevant": true, "reason": "Contains definition of the key concept"}}

Judge relevance:"""

GROUND_TRUTH_RELEVANCE_PROMPT = """You are an expert at judging whether text chunks support answering questions correctly.

Given a question, a reference answer (ground truth), and a text chunk, determine if the chunk contains information that supports or leads to the ground truth answer.

A chunk is relevant if it contains information that would help derive or verify the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Chunk:
{chunk}

Return a JSON object with:
- "relevant": true/false - whether the chunk supports the ground truth answer
- "reason": brief explanation of why it does or doesn't support the answer

Example: {{"relevant": true, "reason": "Contains the specific fact mentioned in the ground truth"}}

Judge relevance to ground truth:"""


@register_metric("context_precision")
class ContextPrecisionMetric(BaseMetric):
    """
    Measures the precision of retrieved context.

    Evaluates whether retrieved chunks are relevant and well-ranked.
    Uses LLM to judge relevance of each chunk.

    When ground truth is available, checks if chunks support the correct answer.
    Without ground truth, falls back to query relevance only.
    """

    name = "context_precision"
    description = "Measures retrieval quality using mean precision@k"
    requires_context = True
    requires_reference = False  # Ground truth is optional but recommended

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
        Compute context precision score.

        Returns score between 0 and 1 where:
        - 1.0 = all chunks are relevant and perfectly ranked
        - 0.0 = no relevant chunks retrieved
        """
        if not retrieved_chunks:
            return MetricResult(
                score=0.0,
                details={
                    "error": "No retrieved chunks provided",
                    "chunk_count": 0,
                    "relevant_count": 0,
                    "used_ground_truth": False,
                },
            )

        # Judge relevance of each chunk
        # Use ground truth if available for more accurate evaluation
        relevances = []
        reasons = []
        use_ground_truth = reference_answer is not None and reference_answer.strip()
        for chunk in retrieved_chunks:
            is_relevant, reason = await self._judge_chunk_relevance(
                question, chunk, ground_truth=reference_answer if use_ground_truth else None
            )
            relevances.append(is_relevant)
            reasons.append(reason)

        # Calculate mean precision@k
        total_relevant = sum(relevances)
        if total_relevant == 0:
            return MetricResult(
                score=0.0,
                details={
                    "chunk_count": len(retrieved_chunks),
                    "relevant_count": 0,
                    "chunk_relevances": relevances,
                    "chunk_reasons": reasons,
                    "used_ground_truth": use_ground_truth,
                },
            )

        precision_sum = 0.0
        relevant_so_far = 0

        for k, is_relevant in enumerate(relevances, start=1):
            if is_relevant:
                relevant_so_far += 1
                precision_at_k = relevant_so_far / k
                precision_sum += precision_at_k

        score = precision_sum / total_relevant

        return MetricResult(
            score=score,
            details={
                "chunk_count": len(retrieved_chunks),
                "relevant_count": total_relevant,
                "chunk_relevances": relevances,
                "chunk_reasons": reasons,
                "used_ground_truth": use_ground_truth,
            },
        )

    async def _judge_chunk_relevance(
        self, query: str, chunk: str, ground_truth: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Judge if a chunk is relevant using LLM.

        Args:
            query: The question being asked
            chunk: The text chunk to evaluate
            ground_truth: Optional ground truth answer. If provided, checks if chunk
                         supports the ground truth answer (more accurate). If not provided,
                         falls back to query relevance only.

        Returns:
            Tuple of (is_relevant, reason)
        """
        try:
            if ground_truth:
                # Use ground truth-aware prompt for more accurate evaluation
                prompt = GROUND_TRUTH_RELEVANCE_PROMPT.format(
                    question=query,
                    ground_truth=ground_truth[:1000],
                    chunk=chunk[:2000],
                )
            else:
                # Fall back to query-only relevance
                prompt = RELEVANCE_JUDGMENT_PROMPT.format(
                    question=query,
                    chunk=chunk[:2000],
                )

            result = await self.llm.generate_json(
                prompt=prompt,
                model=self.llm_model,
                temperature=0.1,
                max_tokens=128,
            )

            return result.get("relevant", False), result.get("reason", "")
        except Exception as e:
            logger.warning(f"Failed to judge chunk relevance: {e}")
            # Fallback: simple keyword check
            # Use ground truth words if available for more accurate fallback
            reference_text = ground_truth if ground_truth else query
            reference_words = set(reference_text.lower().split())
            chunk_words = set(chunk.lower().split())
            overlap = len(reference_words & chunk_words) / len(reference_words) if reference_words else 0
            return overlap > 0.3, "Fallback: keyword overlap"
