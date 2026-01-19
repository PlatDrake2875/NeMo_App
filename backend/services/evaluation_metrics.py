"""
RAGAS-style Evaluation Metrics for RAG Pipeline.

DEPRECATED: This module is deprecated. Use `services.metrics` instead.

This module is kept for backward compatibility. All functionality
delegates to the new plugin-based metrics in `services.metrics`.

New code should use:
    from services.metrics import get_metric, compute_metric

    metric = get_metric("answer_correctness")
    result = await metric.compute(...)

Implements proper evaluation metrics based on the RAGAS methodology:
- Answer Correctness: F1 factual similarity + semantic similarity
- Faithfulness: LLM-based claim verification against context
- Response Relevancy: Embedding-based semantic similarity
- Context Precision: Mean precision@k for retrieval ranking
"""

import asyncio
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from services.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

# Emit deprecation warning on module import
warnings.warn(
    "The 'evaluation_metrics' module is deprecated. "
    "Use 'services.metrics' instead for the plugin-based metrics architecture.",
    DeprecationWarning,
    stacklevel=2,
)


# --- LLM Prompts ---

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

CLAIM_VERIFICATION_PROMPT = """You are an expert at verifying factual claims against source text.

Given a claim and a context, determine if the claim is supported by the context.
A claim is supported if the context contains information that directly supports or implies the claim.

Claim: {claim}

Context:
{context}

Return a JSON object with:
- "supported": true/false
- "evidence": the specific text from context that supports the claim (or null if not supported)

Example: {{"supported": true, "evidence": "The relevant text from context..."}}

Verify the claim:"""

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


# --- Data Classes ---

@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""
    claim: str
    supported: bool
    evidence: Optional[str] = None


@dataclass
class FaithfulnessResult:
    """Detailed faithfulness evaluation result."""
    score: float
    total_claims: int
    supported_claims: int
    claim_details: list[ClaimVerification]


@dataclass
class AnswerCorrectnessResult:
    """Detailed answer correctness evaluation result."""
    score: float
    factual_score: float
    semantic_score: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class ContextPrecisionResult:
    """Detailed context precision evaluation result."""
    score: float
    chunk_relevances: list[bool]


# --- Main Metrics Class ---

class RAGEvaluationMetrics:
    """
    RAGAS-style evaluation metrics for RAG pipelines.

    DEPRECATED: This class is deprecated. Use the plugin-based metrics instead:

        from services.metrics import get_metric, compute_metric

        # Get a specific metric
        metric = get_metric("answer_correctness")
        result = await metric.compute(question, answer, reference, chunks)

        # Or use the convenience function
        result = await compute_metric("faithfulness", ...)

    This class is kept for backward compatibility and will be removed in a future version.

    Uses a combination of LLM-based evaluation and embedding similarity
    to provide accurate, meaningful metrics.
    """

    def __init__(
        self,
        llm_client: Optional[OpenRouterClient] = None,
        embedder_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-4o-mini",
    ):
        """
        Initialize the metrics evaluator.

        Args:
            llm_client: OpenRouter client for LLM calls (created if not provided)
            embedder_model: Sentence transformer model for embeddings
            llm_model: Model to use for LLM-based evaluation
        """
        warnings.warn(
            "RAGEvaluationMetrics is deprecated. "
            "Use 'services.metrics' module instead (get_metric, compute_metric).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.llm = llm_client or OpenRouterClient()
        self.llm_model = llm_model
        self._embedder = None
        self._embedder_model = embedder_model

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy load embedder to avoid startup delay."""
        if self._embedder is None:
            logger.info(f"Loading embedder model: {self._embedder_model}")
            self._embedder = SentenceTransformer(self._embedder_model)
        return self._embedder

    # --- Core Metrics ---

    async def answer_correctness(
        self,
        predicted: str,
        ground_truth: str,
        factual_weight: float = 0.5,
    ) -> AnswerCorrectnessResult:
        """
        Calculate answer correctness using F1 factual similarity + semantic similarity.

        Based on RAGAS Answer Correctness metric:
        - Extracts claims from both predicted and ground truth
        - Calculates F1 score based on claim overlap (TP, FP, FN)
        - Calculates semantic similarity using embeddings
        - Returns weighted average

        Args:
            predicted: The generated/predicted answer
            ground_truth: The expected/reference answer
            factual_weight: Weight for factual score (semantic = 1 - factual_weight)

        Returns:
            AnswerCorrectnessResult with score breakdown
        """
        # Extract claims from both texts
        predicted_claims, ground_truth_claims = await asyncio.gather(
            self._extract_claims(predicted),
            self._extract_claims(ground_truth),
        )

        logger.debug(f"Predicted claims: {len(predicted_claims)}, Ground truth claims: {len(ground_truth_claims)}")

        # Calculate factual F1 score
        if not predicted_claims and not ground_truth_claims:
            # Both empty - perfect match
            tp, fp, fn = 0, 0, 0
            factual_score = 1.0
        elif not predicted_claims or not ground_truth_claims:
            # One empty - no overlap possible
            tp = 0
            fp = len(predicted_claims)
            fn = len(ground_truth_claims)
            factual_score = 0.0
        else:
            # Compare claims using semantic similarity
            tp, fp, fn = await self._compare_claims(predicted_claims, ground_truth_claims)

            # Standard F1 score formula: 2*TP / (2*TP + FP + FN)
            denominator = 2 * tp + fp + fn
            if denominator == 0:
                factual_score = 0.0
            else:
                factual_score = (2 * tp) / denominator

        # Calculate semantic similarity
        semantic_score = self._cosine_similarity(predicted, ground_truth)

        # Weighted average
        final_score = factual_weight * factual_score + (1 - factual_weight) * semantic_score

        return AnswerCorrectnessResult(
            score=final_score,
            factual_score=factual_score,
            semantic_score=semantic_score,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    async def faithfulness(
        self,
        answer: str,
        chunks: list[str],
    ) -> FaithfulnessResult:
        """
        Calculate faithfulness score - are claims in the answer supported by context?

        Based on RAGAS Faithfulness metric:
        - Extracts claims from the generated answer
        - Verifies each claim against the retrieved context
        - Score = supported_claims / total_claims

        Args:
            answer: The generated answer to evaluate
            chunks: List of retrieved context chunks

        Returns:
            FaithfulnessResult with score and claim details
        """
        # Extract claims from answer
        claims = await self._extract_claims(answer)

        if not claims:
            # No claims to verify - consider faithful
            return FaithfulnessResult(
                score=1.0,
                total_claims=0,
                supported_claims=0,
                claim_details=[],
            )

        # Combine chunks into context
        context = "\n\n".join(chunks)

        # Verify each claim against context
        verifications = await asyncio.gather(
            *[self._verify_claim(claim, context) for claim in claims]
        )

        supported_count = sum(1 for v in verifications if v.supported)

        return FaithfulnessResult(
            score=supported_count / len(claims),
            total_claims=len(claims),
            supported_claims=supported_count,
            claim_details=verifications,
        )

    def response_relevancy(
        self,
        query: str,
        answer: str,
    ) -> float:
        """
        Calculate response relevancy using embedding similarity.

        Measures how well the answer addresses the question semantically.
        Uses cosine similarity between query and answer embeddings.

        Args:
            query: The original question
            answer: The generated answer

        Returns:
            Relevancy score between 0 and 1
        """
        return self._cosine_similarity(query, answer)

    async def context_precision(
        self,
        query: str,
        chunks: list[str],
        ground_truth: Optional[str] = None,
    ) -> ContextPrecisionResult:
        """
        Calculate context precision - are retrieved chunks relevant and well-ranked?

        Based on RAGAS Context Precision metric:
        - Judges relevance of each chunk to the query (and ground truth if available)
        - Calculates mean precision@k, weighted by position
        - Earlier irrelevant chunks hurt the score more

        Args:
            query: The original question
            chunks: List of retrieved chunks in order
            ground_truth: Optional ground truth answer. If provided, checks if chunks
                         support the ground truth answer (more accurate evaluation).

        Returns:
            ContextPrecisionResult with score and chunk relevances
        """
        if not chunks:
            return ContextPrecisionResult(score=0.0, chunk_relevances=[])

        # Judge relevance of each chunk
        # Use ground truth if available for more accurate evaluation
        relevances = await asyncio.gather(
            *[self._judge_chunk_relevance(query, chunk, ground_truth) for chunk in chunks]
        )

        # Calculate mean precision@k
        # precision@k = relevant_in_top_k / k
        # Final score = sum(precision@k * is_relevant_k) / total_relevant

        total_relevant = sum(relevances)
        if total_relevant == 0:
            return ContextPrecisionResult(score=0.0, chunk_relevances=relevances)

        precision_sum = 0.0
        relevant_so_far = 0

        for k, is_relevant in enumerate(relevances, start=1):
            if is_relevant:
                relevant_so_far += 1
                precision_at_k = relevant_so_far / k
                precision_sum += precision_at_k

        score = precision_sum / total_relevant

        return ContextPrecisionResult(score=score, chunk_relevances=relevances)

    # --- Helper Methods ---

    async def _extract_claims(self, text: str) -> list[str]:
        """Extract atomic factual claims from text using LLM."""
        if not text or not text.strip():
            return []

        try:
            prompt = CLAIM_EXTRACTION_PROMPT.format(text=text[:3000])  # Truncate long text

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
            # Fallback: split by sentences
            return [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 10]

    async def _verify_claim(self, claim: str, context: str) -> ClaimVerification:
        """Verify if a claim is supported by the context using LLM."""
        try:
            prompt = CLAIM_VERIFICATION_PROMPT.format(
                claim=claim,
                context=context[:4000],  # Truncate long context
            )

            result = await self.llm.generate_json(
                prompt=prompt,
                model=self.llm_model,
                temperature=0.1,
                max_tokens=256,
            )

            return ClaimVerification(
                claim=claim,
                supported=result.get("supported", False),
                evidence=result.get("evidence"),
            )
        except Exception as e:
            logger.warning(f"Failed to verify claim: {e}")
            # Fallback: check simple word overlap
            claim_words = set(claim.lower().split())
            context_words = set(context.lower().split())
            overlap = len(claim_words & context_words) / len(claim_words) if claim_words else 0
            return ClaimVerification(
                claim=claim,
                supported=overlap > 0.5,
                evidence=None,
            )

    async def _compare_claims(
        self,
        predicted_claims: list[str],
        ground_truth_claims: list[str],
        threshold: float = 0.7,
    ) -> tuple[int, int, int]:
        """
        Compare two sets of claims to find TP, FP, FN.
        Uses semantic similarity to match claims.

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        # Get embeddings for all claims
        all_claims = predicted_claims + ground_truth_claims
        embeddings = self.embedder.encode(all_claims)

        pred_embeddings = embeddings[:len(predicted_claims)]
        truth_embeddings = embeddings[len(predicted_claims):]

        # Calculate similarity matrix
        similarity_matrix = np.dot(pred_embeddings, truth_embeddings.T)

        # Find matches above threshold
        matched_truth = set()
        matched_pred = set()

        # Greedy matching: match highest similarities first
        flat_indices = np.argsort(similarity_matrix.flatten())[::-1]

        for flat_idx in flat_indices:
            pred_idx = flat_idx // len(ground_truth_claims)
            truth_idx = flat_idx % len(ground_truth_claims)

            if similarity_matrix[pred_idx, truth_idx] < threshold:
                break

            if pred_idx not in matched_pred and truth_idx not in matched_truth:
                matched_pred.add(pred_idx)
                matched_truth.add(truth_idx)

        tp = len(matched_pred)  # Claims in both
        fp = len(predicted_claims) - tp  # Claims only in predicted
        fn = len(ground_truth_claims) - tp  # Claims only in ground truth

        return tp, fp, fn

    async def _judge_chunk_relevance(
        self, query: str, chunk: str, ground_truth: Optional[str] = None
    ) -> bool:
        """
        Judge if a chunk is relevant using LLM.

        Args:
            query: The question being asked
            chunk: The text chunk to evaluate
            ground_truth: Optional ground truth answer. If provided, checks if chunk
                         supports the ground truth answer (more accurate).

        Returns:
            True if the chunk is relevant, False otherwise
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

            return result.get("relevant", False)
        except Exception as e:
            logger.warning(f"Failed to judge chunk relevance: {e}")
            # Fallback: use embedding similarity
            # Use ground truth for fallback if available
            reference_text = ground_truth if ground_truth else query
            similarity = self._cosine_similarity(reference_text, chunk)
            return similarity > 0.3

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using embeddings."""
        if not text1 or not text2:
            return 0.0

        embeddings = self.embedder.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        # Normalize (sentence-transformers usually returns normalized embeddings)
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(similarity / (norm1 * norm2))


# --- Convenience Functions ---

async def evaluate_single_response(
    query: str,
    predicted_answer: str,
    ground_truth: str,
    chunks: list[str],
    metrics: Optional[RAGEvaluationMetrics] = None,
) -> dict[str, Any]:
    """
    Evaluate a single RAG response with all metrics.

    Returns dict with all metric scores and details.
    """
    if metrics is None:
        metrics = RAGEvaluationMetrics()

    # Run metrics in parallel where possible
    correctness_task = metrics.answer_correctness(predicted_answer, ground_truth)
    faithfulness_task = metrics.faithfulness(predicted_answer, chunks)
    precision_task = metrics.context_precision(query, chunks, ground_truth)

    correctness, faithfulness, precision = await asyncio.gather(
        correctness_task,
        faithfulness_task,
        precision_task,
    )

    relevancy = metrics.response_relevancy(query, predicted_answer)

    return {
        "answer_correctness": correctness.score,
        "answer_correctness_detail": {
            "factual_score": correctness.factual_score,
            "semantic_score": correctness.semantic_score,
            "true_positives": correctness.true_positives,
            "false_positives": correctness.false_positives,
            "false_negatives": correctness.false_negatives,
        },
        "faithfulness": faithfulness.score,
        "faithfulness_detail": {
            "total_claims": faithfulness.total_claims,
            "supported_claims": faithfulness.supported_claims,
            "claims": [
                {
                    "claim": c.claim,
                    "supported": c.supported,
                    "evidence": c.evidence,
                }
                for c in faithfulness.claim_details
            ],
        },
        "response_relevancy": relevancy,
        "context_precision": precision.score,
        "context_precision_detail": {
            "chunk_relevances": precision.chunk_relevances,
        },
    }
