"""
Q&A Generator Service - Generate evaluation Q&A pairs from document chunks.

Uses OpenRouter API to generate question-answer pairs suitable for
RAG evaluation from processed dataset chunks.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import POSTGRES_CONNECTION_STRING, OPENROUTER_DEFAULT_MODEL
from services.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

# Directory for storing evaluation datasets
EVAL_DATASETS_DIR = Path("data/evaluation_datasets")
EVAL_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# System prompt for Q&A generation
QA_GENERATION_SYSTEM_PROMPT = """You are an expert at creating evaluation datasets for RAG (Retrieval-Augmented Generation) systems.

Your task is to generate high-quality question-answer pairs from document chunks that can be used to evaluate how well a RAG system retrieves and uses information.

Guidelines for creating good Q&A pairs:
1. Questions should be answerable ONLY from the provided text chunk
2. Questions should test understanding of key concepts, not trivial details
3. Include a mix of question types: factual, reasoning, and comparison
4. Answers should be concise (1-3 sentences) and directly supported by the text
5. Avoid questions that require external knowledge
6. Avoid yes/no questions - prefer open-ended questions

You must respond with valid JSON only, no additional text."""


QA_GENERATION_USER_PROMPT = """Generate {pairs_count} question-answer pairs from the following text chunk from a scientific paper.

Text chunk:
---
{chunk_content}
---

Source: {source_info}

Respond with a JSON object in this exact format:
{{
  "pairs": [
    {{
      "question": "Your question here",
      "answer": "The answer based on the text",
      "difficulty": "easy|medium|hard",
      "type": "factual|reasoning|comparison"
    }}
  ]
}}"""


class QAGeneratorService:
    """Generate Q&A pairs from document chunks using OpenRouter."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or OPENROUTER_DEFAULT_MODEL
        self.client = OpenRouterClient()
        self._engine = None
        self._session_maker = None

    @property
    def engine(self):
        """Lazy-initialize database engine."""
        if self._engine is None:
            self._engine = create_engine(POSTGRES_CONNECTION_STRING)
        return self._engine

    @property
    def session_maker(self):
        """Lazy-initialize session maker."""
        if self._session_maker is None:
            self._session_maker = sessionmaker(bind=self.engine)
        return self._session_maker

    async def get_chunks_for_dataset(
        self,
        processed_dataset_id: int,
        max_chunks: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a processed dataset's vector store.

        Args:
            processed_dataset_id: ID of the processed dataset
            max_chunks: Maximum number of chunks to retrieve
            seed: Random seed for deterministic ordering. If None, uses random order.

        Returns:
            List of chunk dicts with content and metadata
        """
        with self.session_maker() as session:
            # Get the processed dataset info
            result = session.execute(
                text("""
                    SELECT pd.collection_name, pd.name, rd.name as raw_dataset_name
                    FROM processed_datasets pd
                    LEFT JOIN raw_datasets rd ON pd.raw_dataset_id = rd.id
                    WHERE pd.id = :dataset_id
                """),
                {"dataset_id": processed_dataset_id}
            )
            row = result.fetchone()

            if not row:
                raise ValueError(f"Processed dataset {processed_dataset_id} not found")

            collection_name = row[0]
            dataset_name = row[1]

            # Query chunks from langchain_pg_embedding table
            # Use deterministic ordering if seed is provided
            limit_clause = f"LIMIT {max_chunks}" if max_chunks else ""
            if seed is not None:
                # Deterministic ordering using hash of uuid combined with seed
                # This ensures same seed always produces same ordering
                order_clause = f"ORDER BY md5(uuid::text || '{seed}'::text)"
            else:
                order_clause = "ORDER BY RANDOM()"

            chunks_result = session.execute(
                text(f"""
                    SELECT document, cmetadata, uuid
                    FROM langchain_pg_embedding
                    WHERE collection_id = (
                        SELECT uuid FROM langchain_pg_collection
                        WHERE name = :collection_name
                    )
                    {order_clause}
                    {limit_clause}
                """),
                {"collection_name": collection_name}
            )

            chunks = []
            for chunk_row in chunks_result:
                content = chunk_row[0]
                metadata = chunk_row[1] if chunk_row[1] else {}
                chunk_uuid = str(chunk_row[2]) if len(chunk_row) > 2 else None

                # Extract source info from metadata
                source_info = metadata.get("source", "Unknown source")
                if "page" in metadata:
                    source_info += f", Page {metadata['page']}"

                chunks.append({
                    "content": content,
                    "metadata": metadata,
                    "source_info": source_info,
                    "dataset_name": dataset_name,
                    "chunk_id": chunk_uuid,
                })

            return chunks

    async def generate_pairs_for_chunk(
        self,
        chunk_content: str,
        source_info: str,
        pairs_count: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Generate Q&A pairs for a single chunk.

        Args:
            chunk_content: The text content of the chunk
            source_info: Source information (filename, page, etc.)
            pairs_count: Number of pairs to generate

        Returns:
            List of Q&A pair dicts
        """
        # Truncate very long chunks
        max_chunk_length = 3000
        if len(chunk_content) > max_chunk_length:
            chunk_content = chunk_content[:max_chunk_length] + "\n[Content truncated...]"

        prompt = QA_GENERATION_USER_PROMPT.format(
            pairs_count=pairs_count,
            chunk_content=chunk_content,
            source_info=source_info,
        )

        try:
            response = await self.client.generate_json(
                prompt=prompt,
                system_prompt=QA_GENERATION_SYSTEM_PROMPT,
                model=self.model,
                temperature=0.3,
                max_tokens=1024,
            )

            pairs = response.get("pairs", [])

            # Validate and clean pairs
            valid_pairs = []
            for pair in pairs:
                if "question" in pair and "answer" in pair:
                    valid_pairs.append({
                        "query": pair["question"],
                        "ground_truth": pair["answer"],
                        "difficulty": pair.get("difficulty", "medium"),
                        "type": pair.get("type", "factual"),
                        "source": source_info,
                    })

            return valid_pairs

        except Exception as e:
            logger.error(f"Error generating pairs for chunk: {e}")
            return []

    async def generate_pairs_for_dataset(
        self,
        processed_dataset_id: int,
        name: str,
        pairs_per_chunk: int = 2,
        max_chunks: Optional[int] = 50,
        batch_size: int = 5,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate Q&A pairs from a processed dataset's chunks.

        Args:
            processed_dataset_id: ID of the processed dataset
            name: Name for the evaluation dataset
            pairs_per_chunk: Number of Q&A pairs per chunk
            max_chunks: Maximum chunks to process (None for all)
            batch_size: Number of chunks to process concurrently
            seed: Random seed for deterministic chunk selection

        Returns:
            Dict with dataset info and generated pairs
        """
        logger.info(
            f"Starting Q&A generation for dataset {processed_dataset_id}, "
            f"name='{name}', pairs_per_chunk={pairs_per_chunk}, max_chunks={max_chunks}, "
            f"seed={seed}"
        )

        # Get chunks from the dataset
        chunks = await self.get_chunks_for_dataset(
            processed_dataset_id,
            max_chunks=max_chunks,
            seed=seed,
        )

        if not chunks:
            raise ValueError(f"No chunks found for dataset {processed_dataset_id}")

        logger.info(f"Retrieved {len(chunks)} chunks for processing")

        # Process chunks in batches
        all_pairs = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches}")

            # Process batch concurrently
            tasks = [
                self.generate_pairs_for_chunk(
                    chunk["content"],
                    chunk["source_info"],
                    pairs_per_chunk,
                )
                for chunk in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch item failed: {result}")
                elif isinstance(result, list):
                    all_pairs.extend(result)

            # Small delay between batches to avoid rate limits
            if i + batch_size < total_chunks:
                await asyncio.sleep(0.5)

        logger.info(f"Generated {len(all_pairs)} Q&A pairs total")

        # Create and save the evaluation dataset
        dataset_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        # Compute content hash for reproducibility tracking
        import hashlib
        pairs_json = json.dumps(all_pairs, sort_keys=True)
        content_hash = hashlib.sha256(pairs_json.encode()).hexdigest()

        dataset = {
            "id": dataset_id,
            "name": name,
            "pairs": all_pairs,
            "created_at": created_at,
            "source_dataset_id": processed_dataset_id,
            "content_hash": content_hash,
            "generation_config": {
                "model": self.model,
                "pairs_per_chunk": pairs_per_chunk,
                "max_chunks": max_chunks,
                "chunks_processed": len(chunks),
                "seed": seed,
            },
        }

        # Save to file
        dataset_path = EVAL_DATASETS_DIR / f"{dataset_id}.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Saved evaluation dataset '{name}' with ID {dataset_id}")

        return {
            "id": dataset_id,
            "name": name,
            "pair_count": len(all_pairs),
            "chunks_processed": len(chunks),
            "created_at": created_at,
        }

    async def generate_sample_pairs(
        self,
        processed_dataset_id: int,
        sample_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate a small sample of Q&A pairs for preview.

        Args:
            processed_dataset_id: ID of the processed dataset
            sample_size: Number of chunks to sample

        Returns:
            List of generated Q&A pairs
        """
        chunks = await self.get_chunks_for_dataset(
            processed_dataset_id,
            max_chunks=sample_size,
        )

        all_pairs = []
        for chunk in chunks:
            pairs = await self.generate_pairs_for_chunk(
                chunk["content"],
                chunk["source_info"],
                pairs_count=2,
            )
            all_pairs.extend(pairs)

        return all_pairs
