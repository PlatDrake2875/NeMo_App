"""
LLM Metadata Extractor Service - Extract structured metadata using LLM.
Uses vLLM via OpenAI-compatible API for metadata extraction.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from config import VLLM_BASE_URL, VLLM_MODEL
from schemas import LLMMetadataConfig

logger = logging.getLogger(__name__)


class LLMMetadataExtractor:
    """Extract metadata from documents using LLM."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or VLLM_MODEL
        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy-initialize the LLM client."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key="EMPTY",
                openai_api_base=f"{VLLM_BASE_URL}/v1",
                temperature=0.1,
                max_tokens=1024,
            )
        return self._llm

    async def extract_metadata(
        self, content: str, config: LLMMetadataConfig
    ) -> Dict[str, Any]:
        """
        Extract all configured metadata fields from content.

        Args:
            content: Document text content
            config: LLM metadata extraction configuration

        Returns:
            Dictionary with extracted metadata fields
        """
        if not config.enabled:
            return {}

        result = {}

        # Truncate content if too long
        max_content_length = 4000
        truncated_content = content[:max_content_length]
        if len(content) > max_content_length:
            truncated_content += "\n\n[Content truncated...]"

        try:
            # Extract all fields in a single prompt for efficiency
            extraction_result = await self._extract_all_fields(
                truncated_content, config
            )
            result.update(extraction_result)
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}", exc_info=True)
            result["extraction_error"] = str(e)
            result["extraction_failed"] = True

        return result

    async def _extract_all_fields(
        self, content: str, config: LLMMetadataConfig
    ) -> Dict[str, Any]:
        """Extract all configured fields in a single LLM call."""
        fields_to_extract = []
        if config.extract_summary:
            fields_to_extract.append(
                f"summary: A concise summary of the document in {config.max_summary_length} characters or less"
            )
        if config.extract_keywords:
            fields_to_extract.append(
                f"keywords: Up to {config.max_keywords} relevant keywords as a JSON array of strings"
            )
        if config.extract_entities:
            fields_to_extract.append(
                "entities: Named entities (people, organizations, locations, dates) as a JSON array of objects with 'type' and 'value' keys"
            )
        if config.extract_categories:
            fields_to_extract.append(
                "categories: Document categories/topics as a JSON array of strings (e.g., 'Technology', 'Science')"
            )

        if not fields_to_extract:
            return {}

        prompt = self._build_extraction_prompt(content, fields_to_extract)

        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.content

            # Parse the structured response
            return self._parse_extraction_response(response_text, config)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise

    def _build_extraction_prompt(
        self, content: str, fields_to_extract: List[str]
    ) -> str:
        """Build the extraction prompt."""
        fields_list = "\n".join(f"- {field}" for field in fields_to_extract)

        return f"""Analyze the following document and extract the requested metadata fields.
Respond ONLY with a valid JSON object containing the extracted fields.

Fields to extract:
{fields_list}

Document:
---
{content}
---

Respond with a JSON object containing the requested fields. Example format:
{{
    "summary": "Brief summary here",
    "keywords": ["keyword1", "keyword2"],
    "entities": [{{"type": "person", "value": "Name"}}],
    "categories": ["Category1", "Category2"]
}}

JSON Response:"""

    def _parse_extraction_response(
        self, response_text: str, config: LLMMetadataConfig
    ) -> Dict[str, Any]:
        """Parse the LLM response into structured metadata."""
        result = {}

        # Try to extract JSON from the response
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())

                if config.extract_summary and "summary" in parsed:
                    summary = str(parsed["summary"])
                    # Truncate if needed
                    result["summary"] = summary[: config.max_summary_length]

                if config.extract_keywords and "keywords" in parsed:
                    keywords = parsed["keywords"]
                    if isinstance(keywords, list):
                        result["keywords"] = [
                            str(k) for k in keywords[: config.max_keywords]
                        ]

                if config.extract_entities and "entities" in parsed:
                    entities = parsed["entities"]
                    if isinstance(entities, list):
                        result["entities"] = [
                            {"type": str(e.get("type", "unknown")), "value": str(e.get("value", ""))}
                            for e in entities
                            if isinstance(e, dict)
                        ]

                if config.extract_categories and "categories" in parsed:
                    categories = parsed["categories"]
                    if isinstance(categories, list):
                        result["categories"] = [str(c) for c in categories]

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                # Try to extract fields individually
                result = self._fallback_parse(response_text, config)
                result["_used_fallback_parsing"] = True
        else:
            logger.warning("No JSON found in response, using fallback parsing")
            result = self._fallback_parse(response_text, config)
            result["_used_fallback_parsing"] = True

        return result

    def _fallback_parse(
        self, response_text: str, config: LLMMetadataConfig
    ) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails."""
        result = {}

        # Try to extract summary (first non-empty line or paragraph)
        if config.extract_summary:
            lines = [
                line.strip()
                for line in response_text.split("\n")
                if line.strip() and not line.strip().startswith("{")
            ]
            if lines:
                result["summary"] = lines[0][: config.max_summary_length]

        # Try to extract keywords (look for common patterns)
        if config.extract_keywords:
            keyword_patterns = [
                r"keywords?[:\s]+([^\n\[]+)",
                r"\[([^\]]+)\]",
            ]
            for pattern in keyword_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    keywords_text = match.group(1)
                    keywords = [
                        k.strip().strip('"\'')
                        for k in re.split(r"[,;]", keywords_text)
                    ]
                    result["keywords"] = [k for k in keywords if k][
                        : config.max_keywords
                    ]
                    break

        return result

    async def extract_summary(
        self, content: str, max_length: int = 200
    ) -> Optional[str]:
        """Extract just a summary from content."""
        config = LLMMetadataConfig(
            enabled=True,
            extract_summary=True,
            extract_keywords=False,
            extract_entities=False,
            extract_categories=False,
            max_summary_length=max_length,
        )
        result = await self.extract_metadata(content, config)
        return result.get("summary")

    async def extract_keywords(
        self, content: str, max_keywords: int = 10
    ) -> Optional[List[str]]:
        """Extract just keywords from content."""
        config = LLMMetadataConfig(
            enabled=True,
            extract_summary=False,
            extract_keywords=True,
            extract_entities=False,
            extract_categories=False,
            max_keywords=max_keywords,
        )
        result = await self.extract_metadata(content, config)
        return result.get("keywords")

    async def extract_entities(
        self, content: str
    ) -> Optional[List[Dict[str, str]]]:
        """Extract just entities from content."""
        config = LLMMetadataConfig(
            enabled=True,
            extract_summary=False,
            extract_keywords=False,
            extract_entities=True,
            extract_categories=False,
        )
        result = await self.extract_metadata(content, config)
        return result.get("entities")

    async def extract_categories(self, content: str) -> Optional[List[str]]:
        """Extract just categories from content."""
        config = LLMMetadataConfig(
            enabled=True,
            extract_summary=False,
            extract_keywords=False,
            extract_entities=False,
            extract_categories=True,
        )
        result = await self.extract_metadata(content, config)
        return result.get("categories")


# Factory function to create extractor with specific model
def create_metadata_extractor(model_name: Optional[str] = None) -> LLMMetadataExtractor:
    """Create a metadata extractor with the specified model."""
    return LLMMetadataExtractor(model_name=model_name)


# Default singleton instance
llm_metadata_extractor = LLMMetadataExtractor()
