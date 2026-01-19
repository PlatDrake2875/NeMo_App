"""
Text Cleaning Utilities for RAG Preprocessing Pipeline.

Pure functions for text cleaning operations, using only stdlib where possible.
All functions are designed to be composable and side-effect free.
"""

import html
import re
import unicodedata
from typing import Dict, List, Optional, Tuple


# Regex patterns (compiled for performance)
URL_PATTERN = re.compile(
    r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\-.?=%&#+]*",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"""
    (?:
        (?:\+\d{1,3}[-.\s]?)?      # Optional country code
        (?:\(?\d{2,4}\)?[-.\s]?)?   # Optional area code
        \d{3}[-.\s]?\d{4}           # Main number
    )
    |
    (?:
        \d{3}[-.\s]\d{3}[-.\s]\d{4}  # US format: 555-555-5555
    )
    """,
    re.VERBOSE,
)

# Citation patterns: [1], [1,2,3], [1-5], (Author 2024), (Author et al., 2024)
CITATION_PATTERNS = [
    re.compile(r"\[\d+(?:[-,]\s*\d+)*\]"),  # [1], [1,2,3], [1-5]
    re.compile(r"\(\s*[A-Z][a-zA-Z]*(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-zA-Z]*))?(?:,?\s*\d{4}[a-z]?)\s*\)"),  # (Author 2024), (Smith et al., 2024)
    re.compile(r"\[[A-Z][a-zA-Z]*\d{2,4}[a-z]?\]"),  # [Smith2024], [ABC23]
]

# HTML tag pattern
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Code block markers for preservation
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`]+`", re.MULTILINE)


def remove_html_markup(text: str, preserve_links: bool = True) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Input text potentially containing HTML
        preserve_links: If True, convert <a href="url">text</a> to "text (url)"

    Returns:
        Clean text with HTML tags removed
    """
    if preserve_links:
        # Extract links before removing tags
        link_pattern = re.compile(r'<a\s+[^>]*href=["\']([^"\']*)["\'][^>]*>([^<]*)</a>', re.IGNORECASE)
        text = link_pattern.sub(r"\2 (\1)", text)

    # Decode HTML entities first
    text = html.unescape(text)

    # Remove remaining HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_urls(text: str, replacement: str = "") -> str:
    """
    Remove URLs from text.

    Args:
        text: Input text
        replacement: String to replace URLs with (default: empty string)

    Returns:
        Text with URLs removed
    """
    return URL_PATTERN.sub(replacement, text)


def extract_urls(text: str) -> Tuple[str, List[str]]:
    """
    Extract URLs from text and return both cleaned text and list of URLs.

    Args:
        text: Input text

    Returns:
        Tuple of (cleaned text, list of extracted URLs)
    """
    urls = URL_PATTERN.findall(text)
    cleaned = URL_PATTERN.sub("", text)
    return cleaned.strip(), urls


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Normalize Unicode characters.

    - Converts smart quotes to straight quotes
    - Normalizes ligatures (ﬁ → fi)
    - Standardizes dashes and spaces
    - Applies Unicode normalization (NFC by default)

    Args:
        text: Input text
        form: Unicode normalization form (NFC, NFD, NFKC, NFKD)

    Returns:
        Normalized text
    """
    # Apply Unicode normalization
    text = unicodedata.normalize(form, text)

    # Smart quotes to straight quotes
    replacements = {
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2026": "...",  # Ellipsis
        "\u00a0": " ",  # Non-breaking space
        "\u2002": " ",  # En space
        "\u2003": " ",  # Em space
        "\u2009": " ",  # Thin space
        "\ufeff": "",  # BOM
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def remove_citations(text: str, custom_patterns: Optional[List[str]] = None) -> str:
    """
    Remove academic citation markers from text.

    Handles common patterns like:
    - [1], [1,2,3], [1-5]
    - (Author 2024), (Smith et al., 2024)
    - [Smith2024], [ABC23]

    Args:
        text: Input text
        custom_patterns: Additional regex patterns to match

    Returns:
        Text with citations removed
    """
    result = text

    # Apply built-in patterns
    for pattern in CITATION_PATTERNS:
        result = pattern.sub("", result)

    # Apply custom patterns if provided
    if custom_patterns:
        for pattern_str in custom_patterns:
            try:
                pattern = re.compile(pattern_str)
                result = pattern.sub("", result)
            except re.error:
                pass  # Skip invalid patterns

    return result


def remove_emails(text: str, replacement: str = "[EMAIL]") -> str:
    """
    Remove or redact email addresses.

    Args:
        text: Input text
        replacement: String to replace emails with

    Returns:
        Text with emails removed/redacted
    """
    return EMAIL_PATTERN.sub(replacement, text)


def remove_phone_numbers(text: str, replacement: str = "[PHONE]") -> str:
    """
    Remove or redact phone numbers.

    Args:
        text: Input text
        replacement: String to replace phone numbers with

    Returns:
        Text with phone numbers removed/redacted
    """
    return PHONE_PATTERN.sub(replacement, text)


def remove_pii(
    text: str,
    remove_emails_flag: bool = True,
    remove_phones_flag: bool = True,
    redact: bool = True,
) -> str:
    """
    Remove personally identifiable information from text.

    Args:
        text: Input text
        remove_emails_flag: Whether to remove emails
        remove_phones_flag: Whether to remove phone numbers
        redact: If True, replace with placeholders; if False, remove entirely

    Returns:
        Text with PII removed
    """
    result = text

    if remove_emails_flag:
        replacement = "[EMAIL]" if redact else ""
        result = remove_emails(result, replacement)

    if remove_phones_flag:
        replacement = "[PHONE]" if redact else ""
        result = remove_phone_numbers(result, replacement)

    return result


def preserve_code_blocks(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract and preserve code blocks before cleaning.

    Args:
        text: Input text with code blocks

    Returns:
        Tuple of (text with placeholders, mapping of placeholder -> original code)
    """
    code_map: Dict[str, str] = {}
    counter = 0

    def replace_code(match: re.Match) -> str:
        nonlocal counter
        placeholder = f"__CODE_BLOCK_{counter}__"
        code_map[placeholder] = match.group(0)
        counter += 1
        return placeholder

    cleaned = CODE_BLOCK_PATTERN.sub(replace_code, text)
    return cleaned, code_map


def restore_code_blocks(text: str, code_map: Dict[str, str]) -> str:
    """
    Restore code blocks after cleaning.

    Args:
        text: Text with placeholders
        code_map: Mapping of placeholder -> original code

    Returns:
        Text with code blocks restored
    """
    result = text
    for placeholder, code in code_map.items():
        result = result.replace(placeholder, code)
    return result


def clean_text(
    text: str,
    remove_html: bool = False,
    remove_urls_flag: bool = False,
    remove_citations_flag: bool = False,
    remove_emails_flag: bool = False,
    remove_phones_flag: bool = False,
    normalize_unicode_flag: bool = False,
    preserve_code: bool = True,
    custom_citation_patterns: Optional[List[str]] = None,
) -> str:
    """
    Apply multiple cleaning operations to text.

    This is the main entry point for text cleaning. Operations are applied
    in a specific order to ensure correct results.

    Args:
        text: Input text
        remove_html: Remove HTML markup
        remove_urls_flag: Remove URLs
        remove_citations_flag: Remove citation markers
        remove_emails_flag: Remove email addresses
        remove_phones_flag: Remove phone numbers
        normalize_unicode_flag: Normalize Unicode characters
        preserve_code: Preserve code blocks during cleaning
        custom_citation_patterns: Additional citation patterns

    Returns:
        Cleaned text
    """
    if not text:
        return text

    result = text
    code_map: Dict[str, str] = {}

    # Step 1: Preserve code blocks if requested
    if preserve_code:
        result, code_map = preserve_code_blocks(result)

    # Step 2: Remove HTML (should come early)
    if remove_html:
        result = remove_html_markup(result)

    # Step 3: Normalize Unicode
    if normalize_unicode_flag:
        result = normalize_unicode(result)

    # Step 4: Remove URLs
    if remove_urls_flag:
        result = remove_urls(result)

    # Step 5: Remove citations
    if remove_citations_flag:
        result = remove_citations(result, custom_citation_patterns)

    # Step 6: Remove PII
    if remove_emails_flag or remove_phones_flag:
        result = remove_pii(result, remove_emails_flag, remove_phones_flag, redact=False)

    # Step 7: Restore code blocks
    if preserve_code and code_map:
        result = restore_code_blocks(result, code_map)

    return result
