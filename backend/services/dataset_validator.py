"""
Dataset Validator - Validates evaluation datasets before use.

Provides validation for:
- Q&A pair completeness
- Duplicate detection (exact and semantic)
- Source chunk verification
- Format consistency
- Content hashing for reproducibility
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: str  # "error" or "warning"
    message: str
    pair_index: Optional[int] = None
    field: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "pair_index": self.pair_index,
            "field": self.field,
        }


@dataclass
class ValidationResult:
    """Result of validating a dataset."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    pair_count: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [i.to_dict() for i in self.issues],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "pair_count": self.pair_count,
            "stats": self.stats,
        }


class DatasetValidator:
    """
    Validates evaluation datasets for completeness and consistency.

    Checks:
    - No empty questions or answers
    - No duplicate questions
    - Source chunk IDs exist (if provided)
    - Answer type consistency
    - Difficulty distribution
    """

    def __init__(
        self,
        min_question_length: int = 5,
        min_answer_length: int = 1,
        max_duplicate_ratio: float = 0.1,
    ):
        self.min_question_length = min_question_length
        self.min_answer_length = min_answer_length
        self.max_duplicate_ratio = max_duplicate_ratio

    def validate(
        self,
        pairs: list[dict[str, Any]],
        available_chunk_ids: Optional[set[str]] = None,
    ) -> ValidationResult:
        """
        Validate a list of Q&A pairs.

        Args:
            pairs: List of Q&A pair dictionaries
            available_chunk_ids: Optional set of valid chunk IDs

        Returns:
            ValidationResult with issues and stats
        """
        issues = []
        seen_questions = set()
        duplicates = 0

        answer_types = {}
        difficulties = {}
        answerable_count = 0

        for i, pair in enumerate(pairs):
            # Check required fields
            question = pair.get("question", pair.get("query", "")).strip()
            answer = pair.get("expected_answer", pair.get("ground_truth", "")).strip()

            # Empty question check
            if not question:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message="Empty question",
                        pair_index=i,
                        field="question",
                    )
                )
            elif len(question) < self.min_question_length:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"Question too short ({len(question)} chars)",
                        pair_index=i,
                        field="question",
                    )
                )

            # Empty answer check
            if not answer:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message="Empty answer",
                        pair_index=i,
                        field="expected_answer",
                    )
                )
            elif len(answer) < self.min_answer_length:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"Answer too short ({len(answer)} chars)",
                        pair_index=i,
                        field="expected_answer",
                    )
                )

            # Duplicate question check
            q_normalized = question.lower().strip()
            if q_normalized in seen_questions:
                duplicates += 1
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message="Duplicate question",
                        pair_index=i,
                        field="question",
                    )
                )
            seen_questions.add(q_normalized)

            # Source chunk ID check
            chunk_ids = pair.get("source_chunk_ids", [])
            if chunk_ids and available_chunk_ids:
                for chunk_id in chunk_ids:
                    if chunk_id not in available_chunk_ids:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                message=f"Unknown chunk ID: {chunk_id}",
                                pair_index=i,
                                field="source_chunk_ids",
                            )
                        )

            # Track answer type distribution
            answer_type = pair.get("answer_type", "abstractive")
            answer_types[answer_type] = answer_types.get(answer_type, 0) + 1

            # Track difficulty distribution
            difficulty = pair.get("difficulty", "medium")
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

            # Track answerable count
            if pair.get("is_answerable", True):
                answerable_count += 1

        # Check duplicate ratio
        if pairs and duplicates / len(pairs) > self.max_duplicate_ratio:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Too many duplicates: {duplicates}/{len(pairs)} ({duplicates/len(pairs)*100:.1f}%)",
                )
            )

        # Compile stats
        stats = {
            "answer_type_distribution": answer_types,
            "difficulty_distribution": difficulties,
            "answerable_count": answerable_count,
            "unanswerable_count": len(pairs) - answerable_count,
            "duplicate_count": duplicates,
        }

        # Determine overall validity
        has_errors = any(i.severity == "error" for i in issues)

        return ValidationResult(
            valid=not has_errors,
            issues=issues,
            pair_count=len(pairs),
            stats=stats,
        )

    def validate_import_result(
        self,
        import_result: "ImportResult",
        available_chunk_ids: Optional[set[str]] = None,
    ) -> ValidationResult:
        """
        Validate an import result.

        Args:
            import_result: Result from a dataset importer
            available_chunk_ids: Optional set of valid chunk IDs

        Returns:
            ValidationResult with issues and stats
        """
        # Convert ImportedQAPair objects to dicts
        pairs = [p.to_dict() for p in import_result.pairs]
        return self.validate(pairs, available_chunk_ids)


def validate_qa_pairs(
    pairs: list[dict[str, Any]],
    available_chunk_ids: Optional[set[str]] = None,
) -> ValidationResult:
    """
    Convenience function to validate Q&A pairs.

    Args:
        pairs: List of Q&A pair dictionaries
        available_chunk_ids: Optional set of valid chunk IDs

    Returns:
        ValidationResult with issues and stats
    """
    validator = DatasetValidator()
    return validator.validate(pairs, available_chunk_ids)


def compute_content_hash(question: str, answer: str) -> str:
    """
    Compute content hash for a Q&A pair.

    Use this for deduplication and content-based operations.

    Args:
        question: The question text
        answer: The answer text

    Returns:
        SHA-256 hash of normalized content
    """
    normalized = f"{question.lower().strip()}|||{answer.lower().strip()}"
    return hashlib.sha256(normalized.encode()).hexdigest()


def find_semantic_duplicates(
    pairs: list[dict[str, Any]],
    similarity_threshold: float = 0.85,
    embedder_model: str = "all-MiniLM-L6-v2",
) -> list[tuple[int, int, float]]:
    """
    Find semantically similar pairs using embedding similarity.

    Args:
        pairs: List of Q&A pairs
        similarity_threshold: Minimum cosine similarity to consider duplicate
        embedder_model: Sentence transformer model to use

    Returns:
        List of (index_a, index_b, similarity) tuples for duplicates
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not available, skipping semantic dedup")
        return []

    if len(pairs) < 2:
        return []

    # Extract questions
    questions = []
    for pair in pairs:
        q = pair.get("question", pair.get("query", ""))
        questions.append(q)

    # Compute embeddings
    embedder = SentenceTransformer(embedder_model)
    embeddings = embedder.encode(questions, show_progress_bar=False)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Find similar pairs (upper triangle only to avoid duplicates)
    duplicates = []
    n = len(pairs)

    for i in range(n):
        for j in range(i + 1, n):
            similarity = float(np.dot(normalized[i], normalized[j]))
            if similarity >= similarity_threshold:
                duplicates.append((i, j, similarity))

    return duplicates


def deduplicate_pairs(
    pairs: list[dict[str, Any]],
    method: str = "exact",
    similarity_threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Remove duplicate pairs from a dataset.

    Args:
        pairs: List of Q&A pairs
        method: "exact" (hash-based) or "semantic" (embedding-based)
        similarity_threshold: For semantic method, similarity threshold

    Returns:
        Tuple of (deduplicated_pairs, removed_pairs)
    """
    if method == "exact":
        # Hash-based deduplication
        seen_hashes = {}
        deduplicated = []
        removed = []

        for pair in pairs:
            question = pair.get("question", pair.get("query", ""))
            answer = pair.get("expected_answer", pair.get("ground_truth", ""))
            content_hash = compute_content_hash(question, answer)

            if content_hash not in seen_hashes:
                seen_hashes[content_hash] = len(deduplicated)
                deduplicated.append(pair)
            else:
                pair_with_reason = pair.copy()
                pair_with_reason["_duplicate_of"] = seen_hashes[content_hash]
                removed.append(pair_with_reason)

        return deduplicated, removed

    elif method == "semantic":
        # Semantic deduplication
        duplicates = find_semantic_duplicates(pairs, similarity_threshold)

        # Build set of indices to remove (keep first occurrence)
        to_remove = set()
        for i, j, _ in duplicates:
            to_remove.add(j)  # Remove the later occurrence

        deduplicated = []
        removed = []
        for i, pair in enumerate(pairs):
            if i in to_remove:
                removed.append(pair)
            else:
                deduplicated.append(pair)

        return deduplicated, removed

    else:
        raise ValueError(f"Unknown deduplication method: {method}")


def add_content_hashes(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add content_hash field to all pairs.

    Args:
        pairs: List of Q&A pairs

    Returns:
        Pairs with content_hash field added
    """
    result = []
    for pair in pairs:
        pair_copy = pair.copy()
        question = pair.get("question", pair.get("query", ""))
        answer = pair.get("expected_answer", pair.get("ground_truth", ""))
        pair_copy["content_hash"] = compute_content_hash(question, answer)
        result.append(pair_copy)
    return result


def compute_dataset_hash(pairs: list[dict[str, Any]]) -> str:
    """
    Compute overall hash for a dataset for integrity checking.

    Args:
        pairs: List of Q&A pairs

    Returns:
        SHA-256 hash of the entire dataset
    """
    import json
    # Sort keys for deterministic hashing
    pairs_json = json.dumps(pairs, sort_keys=True)
    return hashlib.sha256(pairs_json.encode()).hexdigest()
