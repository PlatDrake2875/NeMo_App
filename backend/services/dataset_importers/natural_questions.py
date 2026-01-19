"""
Natural Questions Dataset Importer.

Imports Google's Natural Questions format:
https://ai.google.com/research/NaturalQuestions

Natural Questions JSONL structure (simplified):
{
    "question_text": "what is the capital of france",
    "annotations": [
        {
            "short_answers": [
                {"start_token": 123, "end_token": 125, "text": "Paris"}
            ],
            "long_answer": {
                "start_token": 100, "end_token": 150
            },
            "yes_no_answer": "NONE"  # or "YES" or "NO"
        }
    ],
    "document_text": "The full document..."
}
"""

import json
import logging
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from .base import BaseDatasetImporter, ImportedQAPair, ImportResult

logger = logging.getLogger(__name__)


class NaturalQuestionsImporter(BaseDatasetImporter):
    """
    Importer for Google's Natural Questions dataset format.

    Supports the simplified JSONL format commonly used for NQ.
    """

    format_name = "natural_questions"
    description = "Google Natural Questions dataset"
    supported_extensions = [".json", ".jsonl"]

    def import_from_file(
        self,
        file_path: Union[str, Path],
        max_pairs: Optional[int] = None,
        prefer_short_answers: bool = True,
        **kwargs: Any,
    ) -> ImportResult:
        """
        Import Q&A pairs from a Natural Questions file.

        Args:
            file_path: Path to the NQ JSON/JSONL file
            max_pairs: Maximum number of pairs to import
            prefer_short_answers: Prefer short answers over long answers

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
            return self.import_from_stream(f, max_pairs, prefer_short_answers, **kwargs)

    def import_from_stream(
        self,
        file_stream: BinaryIO,
        max_pairs: Optional[int] = None,
        prefer_short_answers: bool = True,
        **kwargs: Any,
    ) -> ImportResult:
        """
        Import Q&A pairs from a Natural Questions stream.

        Args:
            file_stream: File-like object with NQ content
            max_pairs: Maximum number of pairs to import
            prefer_short_answers: Prefer short answers over long answers

        Returns:
            ImportResult with imported pairs
        """
        pairs = []
        errors = []
        warnings = []
        total_processed = 0
        skipped = 0

        # Try to detect format (JSON array or JSONL)
        content = file_stream.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        # Try JSONL first (one JSON object per line)
        lines = content.strip().split("\n")
        is_jsonl = len(lines) > 1 and lines[0].strip().startswith("{")

        if is_jsonl:
            records = []
            for line in lines:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            # Try as JSON array
            try:
                data = json.loads(content)
                records = data if isinstance(data, list) else [data]
            except json.JSONDecodeError as e:
                return ImportResult(
                    success=False,
                    errors=[f"Invalid JSON: {e}"],
                    source_format=self.format_name,
                )

        for record in records:
            total_processed += 1

            if max_pairs and len(pairs) >= max_pairs:
                break

            try:
                question = record.get("question_text", record.get("question", "")).strip()
                if not question:
                    skipped += 1
                    continue

                # Initialize alternative_answers at the start of each iteration
                # This fixes the bug where the variable might not be defined
                alternative_answers = []

                # Get annotations
                annotations = record.get("annotations", [])
                if not annotations:
                    # Try alternative format
                    if "answer" in record:
                        answer = record["answer"]
                        if isinstance(answer, str):
                            expected_answer = answer
                        elif isinstance(answer, dict):
                            expected_answer = answer.get("text", answer.get("value", ""))
                        else:
                            skipped += 1
                            continue

                        pairs.append(
                            ImportedQAPair(
                                question=question,
                                expected_answer=expected_answer,
                                answer_type="abstractive",
                                metadata={"source": "natural_questions"},
                            )
                        )
                        continue
                    else:
                        skipped += 1
                        continue

                annotation = annotations[0]

                # Determine answer type and extract answer
                yes_no = annotation.get("yes_no_answer", "NONE")
                short_answers = annotation.get("short_answers", [])

                if yes_no in ["YES", "NO"]:
                    expected_answer = yes_no
                    answer_type = "yes_no"
                elif short_answers and prefer_short_answers:
                    # Get first short answer
                    first_short = short_answers[0]
                    if "text" in first_short:
                        expected_answer = first_short["text"]
                    elif "start_token" in first_short and "end_token" in first_short:
                        # Need to extract from document
                        doc_text = record.get("document_text", "")
                        tokens = doc_text.split()
                        start = first_short["start_token"]
                        end = first_short["end_token"]
                        expected_answer = " ".join(tokens[start:end])
                    else:
                        skipped += 1
                        continue
                    answer_type = "extractive"

                    # Get alternative answers
                    for sa in short_answers[1:]:
                        if "text" in sa:
                            alt = sa["text"]
                            if alt and alt != expected_answer:
                                alternative_answers.append(alt)
                else:
                    # No valid answer
                    skipped += 1
                    continue

                if not expected_answer or not expected_answer.strip():
                    skipped += 1
                    continue

                pairs.append(
                    ImportedQAPair(
                        question=question,
                        expected_answer=expected_answer.strip(),
                        alternative_answers=alternative_answers,
                        answer_type=answer_type,
                        metadata={
                            "source": "natural_questions",
                            "example_id": record.get("example_id", ""),
                        },
                    )
                )

            except Exception as e:
                errors.append(f"Error processing record: {e}")
                skipped += 1

        logger.info(f"Imported {len(pairs)} pairs from Natural Questions format")

        return ImportResult(
            success=len(errors) == 0 or len(pairs) > 0,
            pairs=pairs,
            total_processed=total_processed,
            skipped=skipped,
            errors=errors,
            warnings=warnings,
            source_format=self.format_name,
        )
