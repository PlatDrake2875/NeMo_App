"""
Faithfulness Metric - Measures how well the answer is grounded in the retrieved context.

Based on RAGAS faithfulness metric:
- Extracts claims from the generated answer
- Verifies each claim against the retrieved context
- Score = supported_claims / total_claims
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

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


@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""

    claim: str
    supported: bool
    evidence: Optional[str] = None


@register_metric("faithfulness")
class FaithfulnessMetric(BaseMetric):
    """
    Measures how faithful the generated answer is to the retrieved context.

    A high faithfulness score means the answer only contains claims that
    are supported by the retrieved documents (no hallucinations).
    """

    name = "faithfulness"
    description = "Measures if claims in the answer are supported by retrieved context"
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
        Compute faithfulness score.

        Returns score between 0 and 1 where:
        - 1.0 = all claims in answer are supported by context
        - 0.0 = no claims in answer are supported by context
        """
        if not retrieved_chunks:
            return MetricResult(
                score=0.0,
                details={
                    "error": "No retrieved chunks provided",
                    "total_claims": 0,
                    "supported_claims": 0,
                },
            )

        # Extract claims from answer
        claims = await self._extract_claims(generated_answer)

        if not claims:
            # No claims to verify - consider fully faithful
            return MetricResult(
                score=1.0,
                details={
                    "total_claims": 0,
                    "supported_claims": 0,
                    "claims": [],
                },
            )

        # Combine chunks into context
        context = "\n\n".join(retrieved_chunks)

        # Verify each claim against context
        verifications = []
        for claim in claims:
            verification = await self._verify_claim(claim, context)
            verifications.append(verification)

        supported_count = sum(1 for v in verifications if v.supported)
        score = supported_count / len(claims)

        return MetricResult(
            score=score,
            details={
                "total_claims": len(claims),
                "supported_claims": supported_count,
                "claims": [
                    {
                        "claim": v.claim,
                        "supported": v.supported,
                        "evidence": v.evidence,
                    }
                    for v in verifications
                ],
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
            # Fallback: split by sentences
            return [s.strip() for s in text.split(".") if s.strip() and len(s.strip()) > 10]

    async def _verify_claim(self, claim: str, context: str) -> ClaimVerification:
        """Verify if a claim is supported by the context using LLM."""
        try:
            prompt = CLAIM_VERIFICATION_PROMPT.format(
                claim=claim,
                context=context[:4000],
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
