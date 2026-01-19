"""
MS MARCO Dataset Importer.

Imports Microsoft Machine Reading Comprehension format:
https://microsoft.github.io/msmarco/

MS MARCO TSV/JSON structure varies, but commonly:
{
    "query_id": 123,
    "query": "what is the speed of light",
    "passages": [
        {"is_selected": 0, "passage_text": "..."},
        {"is_selected": 1, "passage_text": "The speed of light..."}
    ],
    "answers": ["299,792,458 meters per second"]
}

Or TSV format:
query_id\tquery\tanswer\tpassages
"""

import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from .base import BaseDatasetImporter, ImportedQAPair, ImportResult

logger = logging.getLogger(__name__)


class MSMARCOImporter(BaseDatasetImporter):
    """
    Importer for Microsoft MARCO dataset format.

    Supports both JSON and TSV formats.
    """

    format_name = "msmarco"
    description = "Microsoft Machine Reading Comprehension (MS MARCO)"
    supported_extensions = [".json", ".jsonl", ".tsv"]

    def import_from_file(
        self,
        file_path: Union[str, Path],
        max_pairs: Optional[int] = None,
        **kwargs: Any,
    ) -> ImportResult:
        """
        Import Q&A pairs from an MS MARCO file.

        Args:
            file_path: Path to the MS MARCO file
            max_pairs: Maximum number of pairs to import

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
            return self.import_from_stream(f, max_pairs, file_extension=path.suffix, **kwargs)

    def import_from_stream(
        self,
        file_stream: BinaryIO,
        max_pairs: Optional[int] = None,
        file_extension: str = ".json",
        **kwargs: Any,
    ) -> ImportResult:
        """
        Import Q&A pairs from an MS MARCO stream.

        Args:
            file_stream: File-like object with MS MARCO content
            max_pairs: Maximum number of pairs to import
            file_extension: File extension to determine format

        Returns:
            ImportResult with imported pairs
        """
        content = file_stream.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        if file_extension == ".tsv":
            return self._import_tsv(content, max_pairs)
        else:
            return self._import_json(content, max_pairs)

    def _import_json(self, content: str, max_pairs: Optional[int]) -> ImportResult:
        """Import from JSON/JSONL format."""
        pairs = []
        errors = []
        warnings = []
        total_processed = 0
        skipped = 0

        # Try JSONL first
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
                query = record.get("query", record.get("question", "")).strip()
                if not query:
                    skipped += 1
                    continue

                # Get answers
                answers = record.get("answers", record.get("answer", []))
                if isinstance(answers, str):
                    answers = [answers]

                # Filter out "No Answer Present." entries
                answers = [a for a in answers if a and a.strip() and a != "No Answer Present."]

                if not answers:
                    # Check wellFormedAnswers
                    wf_answers = record.get("wellFormedAnswers", [])
                    if wf_answers and wf_answers != ["[]"]:
                        answers = [a for a in wf_answers if a and a.strip()]

                if not answers:
                    skipped += 1
                    continue

                expected_answer = answers[0].strip()
                alternative_answers = [a.strip() for a in answers[1:] if a.strip() != expected_answer]

                pairs.append(
                    ImportedQAPair(
                        question=query,
                        expected_answer=expected_answer,
                        alternative_answers=alternative_answers,
                        answer_type="abstractive",  # MS MARCO uses abstractive answers
                        metadata={
                            "source": "msmarco",
                            "query_id": record.get("query_id", ""),
                        },
                    )
                )

            except Exception as e:
                errors.append(f"Error processing record: {e}")
                skipped += 1

        logger.info(f"Imported {len(pairs)} pairs from MS MARCO JSON format")

        return ImportResult(
            success=len(errors) == 0 or len(pairs) > 0,
            pairs=pairs,
            total_processed=total_processed,
            skipped=skipped,
            errors=errors,
            warnings=warnings,
            source_format=self.format_name,
        )

    def _import_tsv(self, content: str, max_pairs: Optional[int]) -> ImportResult:
        """Import from TSV format."""
        pairs = []
        errors = []
        warnings = []
        total_processed = 0
        skipped = 0

        reader = csv.reader(StringIO(content), delimiter="\t")

        # Try to detect header
        first_row = next(reader, None)
        if not first_row:
            return ImportResult(
                success=False,
                errors=["Empty TSV file"],
                source_format=self.format_name,
            )

        # Check if first row is header
        has_header = any(col.lower() in ["query", "question", "answer"] for col in first_row)
        if has_header:
            headers = [h.lower() for h in first_row]
            query_idx = next((i for i, h in enumerate(headers) if h in ["query", "question"]), 0)
            answer_idx = next((i for i, h in enumerate(headers) if h in ["answer", "answers"]), 1)
        else:
            # Assume query, answer format
            query_idx, answer_idx = 0, 1
            # Process first row as data
            if len(first_row) > answer_idx:
                query = first_row[query_idx].strip()
                answer = first_row[answer_idx].strip()
                if query and answer:
                    pairs.append(
                        ImportedQAPair(
                            question=query,
                            expected_answer=answer,
                            answer_type="abstractive",
                            metadata={"source": "msmarco"},
                        )
                    )
                    total_processed += 1

        for row in reader:
            total_processed += 1

            if max_pairs and len(pairs) >= max_pairs:
                break

            try:
                if len(row) <= max(query_idx, answer_idx):
                    skipped += 1
                    continue

                query = row[query_idx].strip()
                answer = row[answer_idx].strip()

                if not query or not answer or answer == "No Answer Present.":
                    skipped += 1
                    continue

                pairs.append(
                    ImportedQAPair(
                        question=query,
                        expected_answer=answer,
                        answer_type="abstractive",
                        metadata={"source": "msmarco"},
                    )
                )

            except Exception as e:
                errors.append(f"Error processing row: {e}")
                skipped += 1

        logger.info(f"Imported {len(pairs)} pairs from MS MARCO TSV format")

        return ImportResult(
            success=len(errors) == 0 or len(pairs) > 0,
            pairs=pairs,
            total_processed=total_processed,
            skipped=skipped,
            errors=errors,
            warnings=warnings,
            source_format=self.format_name,
        )
