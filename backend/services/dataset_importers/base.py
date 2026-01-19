"""
Base classes for dataset importers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, Optional, Union


# Valid answer types
AnswerType = Literal["extractive", "abstractive", "yes_no", "multi_hop"]


@dataclass
class ImportConfig:
    """Configuration for dataset import."""

    max_pairs: Optional[int] = None
    answer_type_override: Optional[AnswerType] = None  # Override detected answer type
    difficulty_override: Optional[str] = None  # Override detected difficulty
    include_unanswerable: bool = True  # Include unanswerable questions
    min_question_length: int = 5
    min_answer_length: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_pairs": self.max_pairs,
            "answer_type_override": self.answer_type_override,
            "difficulty_override": self.difficulty_override,
            "include_unanswerable": self.include_unanswerable,
            "min_question_length": self.min_question_length,
            "min_answer_length": self.min_answer_length,
        }


@dataclass
class ImportedQAPair:
    """A single imported Q&A pair."""

    question: str
    expected_answer: str
    alternative_answers: list[str] = field(default_factory=list)
    answer_type: str = "abstractive"  # extractive, abstractive, yes_no, multi_hop
    difficulty: str = "medium"  # easy, medium, hard
    source_chunk_ids: list[str] = field(default_factory=list)
    is_answerable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "alternative_answers": self.alternative_answers,
            "answer_type": self.answer_type,
            "difficulty": self.difficulty,
            "source_chunk_ids": self.source_chunk_ids,
            "is_answerable": self.is_answerable,
            "metadata": self.metadata,
        }

    def with_overrides(self, config: ImportConfig) -> "ImportedQAPair":
        """Apply config overrides to this pair."""
        return ImportedQAPair(
            question=self.question,
            expected_answer=self.expected_answer,
            alternative_answers=self.alternative_answers,
            answer_type=config.answer_type_override or self.answer_type,
            difficulty=config.difficulty_override or self.difficulty,
            source_chunk_ids=self.source_chunk_ids,
            is_answerable=self.is_answerable,
            metadata=self.metadata,
        )


@dataclass
class ImportResult:
    """Result of importing a dataset."""

    success: bool
    pairs: list[ImportedQAPair] = field(default_factory=list)
    total_processed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_format: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "pairs": [p.to_dict() for p in self.pairs],
            "total_processed": self.total_processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "warnings": self.warnings,
            "source_format": self.source_format,
        }


class BaseDatasetImporter(ABC):
    """
    Abstract base class for dataset importers.

    Each importer handles a specific dataset format (SQuAD, NQ, MSMARCO, etc.)
    and converts it to a standardized Q&A pair format.
    """

    format_name: str = "base"
    description: str = "Base importer"
    supported_extensions: list[str] = [".json"]

    @abstractmethod
    def import_from_file(
        self,
        file_path: Union[str, Path],
        max_pairs: Optional[int] = None,
        **kwargs: Any,
    ) -> ImportResult:
        """
        Import Q&A pairs from a file.

        Args:
            file_path: Path to the dataset file
            max_pairs: Optional limit on number of pairs to import
            **kwargs: Format-specific options

        Returns:
            ImportResult with imported pairs and status
        """
        pass

    @abstractmethod
    def import_from_stream(
        self,
        file_stream: BinaryIO,
        max_pairs: Optional[int] = None,
        **kwargs: Any,
    ) -> ImportResult:
        """
        Import Q&A pairs from a file stream.

        Args:
            file_stream: File-like object with dataset content
            max_pairs: Optional limit on number of pairs to import
            **kwargs: Format-specific options

        Returns:
            ImportResult with imported pairs and status
        """
        pass

    def validate_file(self, file_path: Union[str, Path]) -> tuple[bool, str]:
        """
        Validate that a file can be imported.

        Args:
            file_path: Path to check

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(file_path)

        if not path.exists():
            return False, f"File not found: {path}"

        if path.suffix not in self.supported_extensions:
            return False, f"Unsupported extension: {path.suffix}. Expected: {self.supported_extensions}"

        return True, ""

    def _infer_answer_type(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Infer the answer type based on question and answer characteristics.

        Args:
            question: The question text
            answer: The answer text
            context: Optional context/document text

        Returns:
            Inferred answer type: "extractive", "abstractive", "yes_no", or "multi_hop"
        """
        answer_lower = answer.lower().strip()
        question_lower = question.lower()

        # Yes/No detection
        if answer_lower in ["yes", "no", "true", "false"]:
            return "yes_no"

        # Check for multi-hop indicators in question
        multi_hop_indicators = [
            "and also",
            "in addition",
            "as well as",
            "both",
            "which of these",
            "compare",
            "what are the",
        ]
        if any(indicator in question_lower for indicator in multi_hop_indicators):
            return "multi_hop"

        # If context is provided, check if answer is extractive
        if context:
            context_lower = context.lower()
            # If the exact answer appears in context, likely extractive
            if answer_lower in context_lower:
                return "extractive"

        # Default to abstractive for longer, more complex answers
        if len(answer.split()) > 10:
            return "abstractive"

        # Short answers without context are typically extractive
        return "extractive"

    def apply_config_overrides(
        self,
        pairs: list[ImportedQAPair],
        config: Optional[ImportConfig] = None,
    ) -> list[ImportedQAPair]:
        """
        Apply configuration overrides to imported pairs.

        Args:
            pairs: List of imported pairs
            config: Import configuration with overrides

        Returns:
            Pairs with overrides applied
        """
        if config is None:
            return pairs

        result = []
        for pair in pairs:
            # Apply overrides if specified
            if config.answer_type_override or config.difficulty_override:
                pair = pair.with_overrides(config)
            result.append(pair)

        return result
