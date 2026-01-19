"""
Contamination Checker - Detect data leakage between training and evaluation sets.

For publication-grade ML research, it's critical to ensure evaluation data
hasn't leaked into training data. This module provides methods to:
- Detect exact duplicates via content hashing
- Find near-duplicates via n-gram overlap
- Identify semantic duplicates via embedding similarity
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContaminationHit:
    """A single contamination detection."""

    eval_pair_index: int
    eval_question: str
    match_type: str  # "exact", "ngram", "semantic"
    similarity_score: float
    matched_content: Optional[str] = None
    matched_source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_pair_index": self.eval_pair_index,
            "eval_question": self.eval_question[:100] + "..." if len(self.eval_question) > 100 else self.eval_question,
            "match_type": self.match_type,
            "similarity_score": self.similarity_score,
            "matched_content": self.matched_content[:200] + "..." if self.matched_content and len(self.matched_content) > 200 else self.matched_content,
            "matched_source": self.matched_source,
        }


@dataclass
class ContaminationResult:
    """Result of contamination checking."""

    total_eval_pairs: int
    total_training_items: int
    contaminated_pairs: int
    contamination_rate: float
    exact_matches: int
    ngram_matches: int
    semantic_matches: int
    hits: list[ContaminationHit] = field(default_factory=list)
    is_clean: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_eval_pairs": self.total_eval_pairs,
            "total_training_items": self.total_training_items,
            "contaminated_pairs": self.contaminated_pairs,
            "contamination_rate": self.contamination_rate,
            "exact_matches": self.exact_matches,
            "ngram_matches": self.ngram_matches,
            "semantic_matches": self.semantic_matches,
            "is_clean": self.is_clean,
            "hits": [h.to_dict() for h in self.hits[:50]],  # Limit output size
        }


class ContaminationChecker:
    """
    Check for contamination between evaluation data and training corpus.

    Supports multiple detection methods:
    - Exact hash matching (fastest, most precise)
    - N-gram overlap (catches paraphrases)
    - Semantic similarity (catches semantic equivalents)
    """

    def __init__(
        self,
        ngram_size: int = 5,
        ngram_threshold: float = 0.3,
        semantic_threshold: float = 0.85,
        embedder_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the contamination checker.

        Args:
            ngram_size: Size of n-grams for overlap detection
            ngram_threshold: Minimum n-gram overlap ratio to flag
            semantic_threshold: Minimum cosine similarity to flag
            embedder_model: Model for semantic similarity
        """
        self.ngram_size = ngram_size
        self.ngram_threshold = ngram_threshold
        self.semantic_threshold = semantic_threshold
        self._embedder = None
        self._embedder_model = embedder_model

    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._embedder_model)
            except ImportError:
                logger.warning("sentence-transformers not available for semantic checking")
        return self._embedder

    def check_contamination(
        self,
        eval_pairs: list[dict[str, Any]],
        training_corpus: list[str],
        check_exact: bool = True,
        check_ngram: bool = True,
        check_semantic: bool = False,  # Slower, optional
    ) -> ContaminationResult:
        """
        Check for contamination between evaluation pairs and training corpus.

        Args:
            eval_pairs: List of evaluation Q&A pairs
            training_corpus: List of training text (documents, chunks, etc.)
            check_exact: Check for exact hash matches
            check_ngram: Check for n-gram overlap
            check_semantic: Check for semantic similarity (slower)

        Returns:
            ContaminationResult with detailed findings
        """
        hits: list[ContaminationHit] = []
        exact_matches = 0
        ngram_matches = 0
        semantic_matches = 0

        # Build training corpus index
        if check_exact:
            training_hashes = self._build_hash_index(training_corpus)

        if check_ngram:
            training_ngrams = [self._get_ngrams(text) for text in training_corpus]

        if check_semantic and self.embedder:
            training_embeddings = self.embedder.encode(
                training_corpus,
                show_progress_bar=False,
                batch_size=32,
            )

        # Check each evaluation pair
        contaminated_indices = set()

        for i, pair in enumerate(eval_pairs):
            question = pair.get("question", pair.get("query", ""))
            answer = pair.get("expected_answer", pair.get("ground_truth", ""))
            combined = f"{question} {answer}"

            # Exact hash check
            if check_exact:
                q_hash = self._compute_hash(question)
                a_hash = self._compute_hash(answer)
                c_hash = self._compute_hash(combined)

                if q_hash in training_hashes or a_hash in training_hashes or c_hash in training_hashes:
                    hits.append(ContaminationHit(
                        eval_pair_index=i,
                        eval_question=question,
                        match_type="exact",
                        similarity_score=1.0,
                        matched_source="hash_match",
                    ))
                    exact_matches += 1
                    contaminated_indices.add(i)
                    continue  # Skip other checks if exact match found

            # N-gram overlap check
            if check_ngram:
                question_ngrams = self._get_ngrams(question)
                if question_ngrams:
                    for j, corpus_ngrams in enumerate(training_ngrams):
                        if corpus_ngrams:
                            overlap = self._ngram_overlap(question_ngrams, corpus_ngrams)
                            if overlap >= self.ngram_threshold:
                                hits.append(ContaminationHit(
                                    eval_pair_index=i,
                                    eval_question=question,
                                    match_type="ngram",
                                    similarity_score=overlap,
                                    matched_content=training_corpus[j],
                                ))
                                ngram_matches += 1
                                contaminated_indices.add(i)
                                break  # One match is enough

            # Semantic similarity check
            if check_semantic and self.embedder and i not in contaminated_indices:
                question_embedding = self.embedder.encode([question])[0]
                similarities = np.dot(training_embeddings, question_embedding)
                max_sim = float(np.max(similarities))
                max_idx = int(np.argmax(similarities))

                if max_sim >= self.semantic_threshold:
                    hits.append(ContaminationHit(
                        eval_pair_index=i,
                        eval_question=question,
                        match_type="semantic",
                        similarity_score=max_sim,
                        matched_content=training_corpus[max_idx],
                    ))
                    semantic_matches += 1
                    contaminated_indices.add(i)

        contaminated_count = len(contaminated_indices)
        contamination_rate = contaminated_count / len(eval_pairs) if eval_pairs else 0.0

        return ContaminationResult(
            total_eval_pairs=len(eval_pairs),
            total_training_items=len(training_corpus),
            contaminated_pairs=contaminated_count,
            contamination_rate=contamination_rate,
            exact_matches=exact_matches,
            ngram_matches=ngram_matches,
            semantic_matches=semantic_matches,
            hits=hits,
            is_clean=contaminated_count == 0,
        )

    def check_internal_contamination(
        self,
        pairs: list[dict[str, Any]],
    ) -> ContaminationResult:
        """
        Check for contamination within a single dataset (duplicates).

        Args:
            pairs: List of Q&A pairs to check

        Returns:
            ContaminationResult showing internal duplicates
        """
        hits: list[ContaminationHit] = []
        seen_hashes: dict[str, int] = {}  # hash -> first index

        for i, pair in enumerate(pairs):
            question = pair.get("question", pair.get("query", ""))
            answer = pair.get("expected_answer", pair.get("ground_truth", ""))

            # Normalize and hash
            q_hash = self._compute_hash(question)
            qa_hash = self._compute_hash(f"{question} {answer}")

            # Check for duplicates
            if q_hash in seen_hashes:
                hits.append(ContaminationHit(
                    eval_pair_index=i,
                    eval_question=question,
                    match_type="exact",
                    similarity_score=1.0,
                    matched_source=f"duplicate_of_index_{seen_hashes[q_hash]}",
                ))
            else:
                seen_hashes[q_hash] = i

        return ContaminationResult(
            total_eval_pairs=len(pairs),
            total_training_items=len(pairs),
            contaminated_pairs=len(hits),
            contamination_rate=len(hits) / len(pairs) if pairs else 0.0,
            exact_matches=len(hits),
            ngram_matches=0,
            semantic_matches=0,
            hits=hits,
            is_clean=len(hits) == 0,
        )

    def _compute_hash(self, text: str) -> str:
        """Compute normalized hash of text."""
        normalized = self._normalize_text(text)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase, remove extra whitespace
        return " ".join(text.lower().split())

    def _build_hash_index(self, texts: list[str]) -> set[str]:
        """Build hash index for fast lookup."""
        return {self._compute_hash(text) for text in texts}

    def _get_ngrams(self, text: str) -> set[tuple]:
        """Extract character n-grams from text."""
        normalized = self._normalize_text(text)
        if len(normalized) < self.ngram_size:
            return set()
        return {
            tuple(normalized[i:i + self.ngram_size])
            for i in range(len(normalized) - self.ngram_size + 1)
        }

    def _ngram_overlap(self, ngrams_a: set, ngrams_b: set) -> float:
        """Calculate Jaccard-like overlap between n-gram sets."""
        if not ngrams_a or not ngrams_b:
            return 0.0
        intersection = len(ngrams_a & ngrams_b)
        # Use the smaller set as denominator for asymmetric matching
        return intersection / min(len(ngrams_a), len(ngrams_b))


def compute_content_hash(question: str, answer: str) -> str:
    """
    Compute content hash for a Q&A pair.

    Use this for deduplication and content-based splitting.
    """
    normalized = f"{question.lower().strip()}|||{answer.lower().strip()}"
    return hashlib.sha256(normalized.encode()).hexdigest()
