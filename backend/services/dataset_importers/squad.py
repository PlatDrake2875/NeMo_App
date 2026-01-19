"""
SQuAD Dataset Importer.

Imports Stanford Question Answering Dataset format:
https://rajpurkar.github.io/SQuAD-explorer/

SQuAD JSON structure:
{
    "data": [
        {
            "title": "Article title",
            "paragraphs": [
                {
                    "context": "The paragraph text...",
                    "qas": [
                        {
                            "question": "What is...?",
                            "id": "unique-id",
                            "answers": [
                                {"text": "answer text", "answer_start": 123}
                            ],
                            "is_impossible": false  # SQuAD 2.0
                        }
                    ]
                }
            ]
        }
    ]
}
"""

import json
import logging
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from .base import BaseDatasetImporter, ImportedQAPair, ImportResult

logger = logging.getLogger(__name__)


class SQuADImporter(BaseDatasetImporter):
    """
    Importer for SQuAD (Stanford Question Answering Dataset) format.

    Supports both SQuAD 1.1 and SQuAD 2.0 formats.
    """

    format_name = "squad"
    description = "Stanford Question Answering Dataset (SQuAD 1.1/2.0)"
    supported_extensions = [".json"]

    def import_from_file(
        self,
        file_path: Union[str, Path],
        max_pairs: Optional[int] = None,
        include_unanswerable: bool = False,
        **kwargs: Any,
    ) -> ImportResult:
        """
        Import Q&A pairs from a SQuAD JSON file.

        Args:
            file_path: Path to the SQuAD JSON file
            max_pairs: Maximum number of pairs to import
            include_unanswerable: Whether to include unanswerable questions (SQuAD 2.0)

        Returns:
            ImportResult with imported pairs
        """
        path = Path(file_path)
        is_valid, error = self.validate_file(path)
        if not is_valid:
            return ImportResult(
                success=False,
                errors=[error],
                source_format=self.format_name,
            )

        with open(path, "r", encoding="utf-8") as f:
            return self.import_from_stream(f, max_pairs, include_unanswerable, **kwargs)

    def import_from_stream(
        self,
        file_stream: BinaryIO,
        max_pairs: Optional[int] = None,
        include_unanswerable: bool = False,
        **kwargs: Any,
    ) -> ImportResult:
        """
        Import Q&A pairs from a SQuAD JSON stream.

        Args:
            file_stream: File-like object with SQuAD JSON content
            max_pairs: Maximum number of pairs to import
            include_unanswerable: Whether to include unanswerable questions

        Returns:
            ImportResult with imported pairs
        """
        pairs = []
        errors = []
        warnings = []
        total_processed = 0
        skipped = 0

        try:
            data = json.load(file_stream)
        except json.JSONDecodeError as e:
            return ImportResult(
                success=False,
                errors=[f"Invalid JSON: {e}"],
                source_format=self.format_name,
            )

        # Handle both nested and flat SQuAD formats
        squad_data = data.get("data", data if isinstance(data, list) else [])

        for article in squad_data:
            title = article.get("title", "Unknown")

            for paragraph in article.get("paragraphs", []):
                context = paragraph.get("context", "")

                for qa in paragraph.get("qas", []):
                    total_processed += 1

                    # Check if we've reached the limit
                    if max_pairs and len(pairs) >= max_pairs:
                        break

                    try:
                        question = qa.get("question", "").strip()
                        qa_id = qa.get("id", "")
                        is_impossible = qa.get("is_impossible", False)

                        if not question:
                            skipped += 1
                            continue

                        # Handle unanswerable questions (SQuAD 2.0)
                        if is_impossible:
                            if not include_unanswerable:
                                skipped += 1
                                continue
                            expected_answer = "[UNANSWERABLE]"
                            alternative_answers = []
                            is_answerable = False
                        else:
                            # Get answers
                            answers = qa.get("answers", [])
                            if not answers:
                                skipped += 1
                                continue

                            expected_answer = answers[0].get("text", "").strip()
                            if not expected_answer:
                                skipped += 1
                                continue

                            # Collect alternative answers (deduplicated)
                            alternative_answers = list(set(
                                a.get("text", "").strip()
                                for a in answers[1:]
                                if a.get("text", "").strip() and a.get("text", "").strip() != expected_answer
                            ))
                            is_answerable = True

                        pairs.append(
                            ImportedQAPair(
                                question=question,
                                expected_answer=expected_answer,
                                alternative_answers=alternative_answers,
                                answer_type="extractive",  # SQuAD is extractive QA
                                difficulty="medium",  # Could be inferred from answer length
                                is_answerable=is_answerable,
                                metadata={
                                    "source": "squad",
                                    "article_title": title,
                                    "qa_id": qa_id,
                                    "context_preview": context[:200] if context else "",
                                },
                            )
                        )

                    except Exception as e:
                        errors.append(f"Error processing QA {qa.get('id', 'unknown')}: {e}")
                        skipped += 1

                if max_pairs and len(pairs) >= max_pairs:
                    break

            if max_pairs and len(pairs) >= max_pairs:
                break

        logger.info(f"Imported {len(pairs)} pairs from SQuAD format")

        return ImportResult(
            success=len(errors) == 0 or len(pairs) > 0,
            pairs=pairs,
            total_processed=total_processed,
            skipped=skipped,
            errors=errors,
            warnings=warnings,
            source_format=self.format_name,
        )
