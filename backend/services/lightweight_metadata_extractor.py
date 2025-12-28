"""
Lightweight Metadata Extraction Service.

Fast metadata extraction without requiring LLM inference.
Uses rule-based and statistical methods that run locally.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Common English stopwords for RAKE algorithm
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}

# Language detection heuristics (common words by language)
LANGUAGE_MARKERS: Dict[str, Set[str]] = {
    "en": {"the", "is", "are", "was", "were", "have", "has", "been", "will", "would"},
    "es": {"el", "la", "los", "las", "es", "son", "fue", "han", "será", "pero"},
    "fr": {"le", "la", "les", "est", "sont", "été", "avoir", "avec", "pour", "mais"},
    "de": {"der", "die", "das", "ist", "sind", "war", "haben", "werden", "mit", "aber"},
    "pt": {"o", "a", "os", "as", "é", "são", "foi", "ter", "com", "para"},
    "it": {"il", "la", "i", "le", "è", "sono", "stato", "avere", "con", "per"},
}


class LightweightMetadataExtractor:
    """
    Extract metadata from documents without LLM inference.

    Provides fast, rule-based extraction of:
    - Keywords (RAKE algorithm)
    - Document statistics (word count, readability)
    - Language detection
    - Named entities (if spaCy available)
    """

    def __init__(self):
        """Initialize the extractor."""
        self._spacy_nlp = None
        self._spacy_available = None

    def extract_all(
        self,
        text: str,
        extract_rake_keywords: bool = True,
        extract_statistics: bool = True,
        detect_language: bool = True,
        extract_spacy_entities: bool = False,
        max_keywords: int = 15,
    ) -> Dict[str, Any]:
        """
        Extract all enabled metadata types from text.

        Args:
            text: Input text
            extract_rake_keywords: Extract keywords using RAKE
            extract_statistics: Extract document statistics
            detect_language: Detect document language
            extract_spacy_entities: Extract named entities (requires spaCy)
            max_keywords: Maximum number of keywords to extract

        Returns:
            Dict with extracted metadata
        """
        result: Dict[str, Any] = {}

        if extract_rake_keywords:
            result["keywords"] = self._extract_rake_keywords(text, max_keywords)

        if extract_statistics:
            result["statistics"] = self._extract_statistics(text)

        if detect_language:
            result["language"] = self._detect_language(text)

        if extract_spacy_entities:
            result["entities"] = self._extract_spacy_entities(text)

        return result

    def _extract_rake_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """
        Extract keywords using RAKE (Rapid Automatic Keyword Extraction).

        RAKE is a domain-independent keyword extraction algorithm that:
        1. Splits text into candidate keywords by stopwords/punctuation
        2. Calculates word scores based on degree/frequency
        3. Ranks candidates by sum of word scores

        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return

        Returns:
            List of extracted keywords
        """
        if not text:
            return []

        # Normalize text
        text = text.lower()

        # Split into sentences
        sentence_delimiters = re.compile(r"[.!?\n]")
        sentences = sentence_delimiters.split(text)

        # Extract candidate keywords (phrases between stopwords)
        phrase_list: List[List[str]] = []
        word_pattern = re.compile(r"[a-zA-Z]+")

        for sentence in sentences:
            words = word_pattern.findall(sentence)
            if not words:
                continue

            # Split by stopwords to get phrases
            current_phrase: List[str] = []
            for word in words:
                if word in STOPWORDS or len(word) < 2:
                    if current_phrase:
                        phrase_list.append(current_phrase)
                        current_phrase = []
                else:
                    current_phrase.append(word)
            if current_phrase:
                phrase_list.append(current_phrase)

        # Calculate word frequency and degree (co-occurrence count)
        word_freq: Counter = Counter()
        word_degree: Counter = Counter()

        for phrase in phrase_list:
            degree = len(phrase) - 1
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += degree

        # Add frequency to degree for final degree
        for word in word_freq:
            word_degree[word] += word_freq[word]

        # Calculate word scores (degree / frequency)
        word_scores: Dict[str, float] = {}
        for word in word_freq:
            word_scores[word] = word_degree[word] / word_freq[word]

        # Score phrases by sum of word scores
        phrase_scores: Dict[str, float] = {}
        for phrase in phrase_list:
            phrase_str = " ".join(phrase)
            if phrase_str not in phrase_scores:
                score = sum(word_scores.get(word, 0) for word in phrase)
                phrase_scores[phrase_str] = score

        # Sort by score and return top keywords
        sorted_phrases = sorted(
            phrase_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [phrase for phrase, _ in sorted_phrases[:max_keywords]]

    def _extract_statistics(self, text: str) -> Dict[str, Any]:
        """
        Extract document statistics.

        Includes:
        - Word count
        - Character count
        - Sentence count
        - Average word length
        - Flesch-Kincaid readability scores

        Args:
            text: Input text

        Returns:
            Dict with statistics
        """
        if not text:
            return {
                "word_count": 0,
                "char_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0,
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
            }

        # Basic counts
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        word_count = len(words)
        char_count = len(text)
        sentences = re.split(r"[.!?]+", text)
        sentence_count = len([s for s in sentences if s.strip()])

        # Average word length
        avg_word_length = (
            sum(len(w) for w in words) / word_count if word_count > 0 else 0
        )

        # Syllable counting (approximation)
        def count_syllables(word: str) -> int:
            word = word.lower()
            if len(word) <= 3:
                return 1
            # Remove silent e at end
            if word.endswith("e"):
                word = word[:-1]
            # Count vowel groups
            vowels = "aeiouy"
            count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)

        # Flesch Reading Ease
        # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        if word_count > 0 and sentence_count > 0:
            flesch_reading_ease = (
                206.835
                - 1.015 * (word_count / sentence_count)
                - 84.6 * (total_syllables / word_count)
            )
            flesch_reading_ease = max(0, min(100, flesch_reading_ease))

            # Flesch-Kincaid Grade Level
            # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
            flesch_kincaid_grade = (
                0.39 * (word_count / sentence_count)
                + 11.8 * (total_syllables / word_count)
                - 15.59
            )
            flesch_kincaid_grade = max(0, flesch_kincaid_grade)
        else:
            flesch_reading_ease = 0
            flesch_kincaid_grade = 0

        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 2),
            "flesch_reading_ease": round(flesch_reading_ease, 2),
            "flesch_kincaid_grade": round(flesch_kincaid_grade, 2),
        }

    def _detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect document language.

        Uses word frequency heuristics. Falls back to langdetect if available.

        Args:
            text: Input text

        Returns:
            Dict with language code and confidence
        """
        if not text:
            return {"code": "unknown", "confidence": 0.0}

        # Try langdetect if available
        try:
            from langdetect import detect_langs

            results = detect_langs(text[:5000])  # Limit text for speed
            if results:
                best = results[0]
                return {"code": best.lang, "confidence": round(best.prob, 3)}
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"langdetect failed: {e}")

        # Fallback: heuristic detection based on common words
        words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
        if not words:
            return {"code": "unknown", "confidence": 0.0}

        scores: Dict[str, int] = {}
        for lang, markers in LANGUAGE_MARKERS.items():
            scores[lang] = len(words & markers)

        if not scores or max(scores.values()) == 0:
            return {"code": "unknown", "confidence": 0.0}

        best_lang = max(scores, key=lambda k: scores[k])
        confidence = scores[best_lang] / len(LANGUAGE_MARKERS[best_lang])

        return {"code": best_lang, "confidence": round(min(confidence, 1.0), 3)}

    def _extract_spacy_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities using spaCy.

        Args:
            text: Input text

        Returns:
            List of entities with text, label, and description
        """
        nlp = self._get_spacy_nlp()
        if nlp is None:
            return []

        # Process text (limit for performance)
        doc = nlp(text[:50000])

        # Extract unique entities
        seen: Set[str] = set()
        entities: List[Dict[str, str]] = []

        for ent in doc.ents:
            key = f"{ent.text.lower()}_{ent.label_}"
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "description": self._get_entity_description(ent.label_),
                })

        return entities

    def _get_spacy_nlp(self) -> Optional[Any]:
        """Get or initialize spaCy NLP model."""
        if self._spacy_available is False:
            return None

        if self._spacy_nlp is not None:
            return self._spacy_nlp

        try:
            import spacy

            # Try to load a model (prefer small English model)
            for model_name in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]:
                try:
                    self._spacy_nlp = spacy.load(model_name)
                    self._spacy_available = True
                    logger.info(f"Loaded spaCy model: {model_name}")
                    return self._spacy_nlp
                except OSError:
                    continue

            logger.warning("No spaCy model found. Install with: python -m spacy download en_core_web_sm")
            self._spacy_available = False
            return None

        except ImportError:
            logger.info("spaCy not installed. Named entity extraction disabled.")
            self._spacy_available = False
            return None

    @staticmethod
    def _get_entity_description(label: str) -> str:
        """Get human-readable description for entity label."""
        descriptions = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Geopolitical Entity",
            "LOC": "Location",
            "DATE": "Date",
            "TIME": "Time",
            "MONEY": "Monetary Value",
            "PERCENT": "Percentage",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "WORK_OF_ART": "Work of Art",
            "LAW": "Law/Legal Document",
            "LANGUAGE": "Language",
            "NORP": "Nationality/Religion/Political Group",
            "FAC": "Facility",
            "QUANTITY": "Quantity",
            "ORDINAL": "Ordinal Number",
            "CARDINAL": "Cardinal Number",
        }
        return descriptions.get(label, label)


# Singleton instance
lightweight_metadata_extractor = LightweightMetadataExtractor()
