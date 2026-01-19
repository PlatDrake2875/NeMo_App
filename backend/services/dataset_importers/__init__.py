"""
Dataset Importers - Support for standard evaluation dataset formats.

Provides importers for common Q&A evaluation formats:
- SQuAD (Stanford Question Answering Dataset)
- Natural Questions
- MS MARCO
"""

from .base import BaseDatasetImporter, ImportResult
from .squad import SQuADImporter
from .natural_questions import NaturalQuestionsImporter
from .msmarco import MSMARCOImporter

__all__ = [
    "BaseDatasetImporter",
    "ImportResult",
    "SQuADImporter",
    "NaturalQuestionsImporter",
    "MSMARCOImporter",
    "get_importer",
    "list_importers",
]

# Registry of available importers
_IMPORTERS = {
    "squad": SQuADImporter,
    "natural_questions": NaturalQuestionsImporter,
    "msmarco": MSMARCOImporter,
}


def get_importer(format_name: str) -> BaseDatasetImporter:
    """
    Get an importer instance by format name.

    Args:
        format_name: The dataset format (e.g., "squad", "natural_questions")

    Returns:
        An importer instance

    Raises:
        ValueError: If format is not supported
    """
    format_name = format_name.lower()
    if format_name not in _IMPORTERS:
        available = ", ".join(_IMPORTERS.keys())
        raise ValueError(f"Format '{format_name}' not supported. Available: {available}")
    return _IMPORTERS[format_name]()


def list_importers() -> list[dict[str, str]]:
    """
    List all available importers with descriptions.

    Returns:
        List of importer info dictionaries
    """
    result = []
    for name, cls in _IMPORTERS.items():
        result.append({
            "format": name,
            "description": cls.description,
        })
    return result
