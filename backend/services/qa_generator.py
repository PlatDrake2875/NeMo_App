"""
Q&A Generator Service - Generate evaluation Q&A pairs from document chunks.

Uses OpenRouter API or local vLLM to generate question-answer pairs suitable for
RAG evaluation from processed dataset chunks.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import (
    POSTGRES_CONNECTION_STRING,
    OPENROUTER_DEFAULT_MODEL,
    OPENROUTER_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
)
from services.openrouter_client import OpenRouterClient
from services.chunking import ChunkingService

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
    """Generate Q&A pairs from document chunks using OpenRouter or vLLM."""

    def __init__(
        self,
        model: Optional[str] = None,
        use_vllm: Optional[bool] = None,
        temperature: float = 0.3,
    ):
        # Determine whether to use vLLM or OpenRouter
        # Use vLLM if explicitly requested OR if OpenRouter API key is not set
        if use_vllm is None:
            use_vllm = not OPENROUTER_API_KEY

        self.use_vllm = use_vllm
        self.temperature = temperature

        if self.use_vllm:
            self.model = model or VLLM_MODEL
            self.client = None  # Will use direct HTTP calls to vLLM
            logger.info(f"QAGenerator using local vLLM with model: {self.model}, temperature: {temperature}")
        else:
            self.model = model or OPENROUTER_DEFAULT_MODEL
            self.client = OpenRouterClient()
            logger.info(f"QAGenerator using OpenRouter with model: {self.model}, temperature: {temperature}")

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
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks from a processed dataset.

        Supports two flows:
        1. New flow: Read from PreprocessedDocument table and chunk on-the-fly
        2. Legacy flow: Read from vector store (pgvector/qdrant)

        Args:
            processed_dataset_id: ID of the processed dataset
            max_chunks: Maximum number of chunks to retrieve
            seed: Random seed for deterministic ordering. If None, uses random order.

        Returns:
            Tuple of (collection_name, list of chunk dicts with content and metadata)
        """
        with self.session_maker() as session:
            # First, check if this dataset uses the new flow (has PreprocessedDocument entries)
            preprocessed_check = session.execute(
                text("""
                    SELECT COUNT(*) FROM preprocessed_documents
                    WHERE processed_dataset_id = :dataset_id
                """),
                {"dataset_id": processed_dataset_id}
            )
            preprocessed_count = preprocessed_check.scalar()

            if preprocessed_count > 0:
                # New flow: Get chunks from PreprocessedDocument table
                logger.info(f"Using new flow: found {preprocessed_count} preprocessed documents")
                chunks = await self._get_chunks_from_preprocessed_docs(
                    session, processed_dataset_id, max_chunks, seed
                )
                # Use dataset ID as collection name placeholder since there's no real collection yet
                return f"preprocessed_{processed_dataset_id}", chunks

            # Legacy flow: Get the processed dataset info including vector_backend
            result = session.execute(
                text("""
                    SELECT pd.collection_name, pd.name, rd.name as raw_dataset_name, pd.vector_backend
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
            vector_backend = row[3] if len(row) > 3 else "pgvector"

            # Route to appropriate backend
            if vector_backend == "qdrant":
                chunks = await self._get_chunks_from_qdrant(
                    collection_name, dataset_name, max_chunks, seed
                )
            else:
                chunks = await self._get_chunks_from_pgvector(
                    session, collection_name, dataset_name, max_chunks, seed
                )

            return collection_name, chunks

    async def _get_chunks_from_preprocessed_docs(
        self,
        session,
        processed_dataset_id: int,
        max_chunks: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get chunks from PreprocessedDocument table (new flow).

        Chunks the full documents on-the-fly using default chunking config.
        """
        from langchain_core.documents import Document

        # Get preprocessed documents
        result = session.execute(
            text("""
                SELECT pd.id, pd.content, pd.original_filename, pd.metadata_json,
                       pds.name as dataset_name
                FROM preprocessed_documents pd
                JOIN processed_datasets pds ON pd.processed_dataset_id = pds.id
                WHERE pd.processed_dataset_id = :dataset_id
            """),
            {"dataset_id": processed_dataset_id}
        )
        rows = result.fetchall()

        if not rows:
            return []

        # Convert to LangChain documents
        documents = []
        for row in rows:
            doc_id, content, filename, metadata_json, dataset_name = row
            metadata = metadata_json if metadata_json else {}
            metadata["original_filename"] = filename
            metadata["preprocessed_doc_id"] = doc_id
            documents.append(Document(page_content=content, metadata=metadata))

        # Chunk documents using default config (for Q&A generation)
        chunking_service = ChunkingService()
        chunked_docs = chunking_service.chunk_documents(
            documents=documents,
            method="recursive",
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Apply ordering
        if seed is not None:
            import hashlib
            chunked_docs.sort(
                key=lambda d: hashlib.md5(
                    f"{d.page_content[:100]}{seed}".encode()
                ).hexdigest()
            )
        else:
            import random
            random.shuffle(chunked_docs)

        # Apply limit
        if max_chunks and len(chunked_docs) > max_chunks:
            chunked_docs = chunked_docs[:max_chunks]

        # Convert to expected format
        chunks = []
        for doc in chunked_docs:
            source_info = doc.metadata.get("original_filename", "Unknown source")
            chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source_info": source_info,
                "dataset_name": dataset_name if rows else "Unknown",
                "chunk_id": None,
            })

        logger.info(f"Generated {len(chunks)} chunks from preprocessed documents")
        return chunks

    async def _get_chunks_from_pgvector(
        self,
        session,
        collection_name: str,
        dataset_name: str,
        max_chunks: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks from PostgreSQL/pgvector backend."""
        # Query chunks from langchain_pg_embedding table
        # Use deterministic ordering if seed is provided
        limit_clause = f"LIMIT {max_chunks}" if max_chunks else ""
        if seed is not None:
            # Deterministic ordering using hash of id combined with seed
            # This ensures same seed always produces same ordering
            order_clause = f"ORDER BY md5(id::text || '{seed}'::text)"
        else:
            order_clause = "ORDER BY RANDOM()"

        chunks_result = session.execute(
            text(f"""
                SELECT document, cmetadata, id
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

    async def _get_chunks_from_qdrant(
        self,
        collection_name: str,
        dataset_name: str,
        max_chunks: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks from Qdrant backend."""
        from services.qdrant_vectorstore import get_qdrant_service

        qdrant_service = get_qdrant_service()

        # Get all documents from the collection
        limit = max_chunks if max_chunks else 10000
        documents = qdrant_service.get_all_documents(
            collection_name=collection_name,
            limit=limit,
        )

        if not documents:
            logger.warning(f"No documents found in Qdrant collection '{collection_name}'")
            return []

        chunks = []
        for doc in documents:
            content = doc.content
            metadata = doc.metadata or {}

            # Extract source info from metadata
            source_info = metadata.get("source", metadata.get("original_filename", "Unknown source"))
            if "page" in metadata:
                source_info += f", Page {metadata['page']}"

            chunks.append({
                "content": content,
                "metadata": metadata,
                "source_info": source_info,
                "dataset_name": dataset_name,
                "chunk_id": doc.id,
            })

        # Apply deterministic ordering if seed is provided
        if seed is not None:
            import hashlib
            chunks.sort(key=lambda c: hashlib.md5(f"{c['chunk_id']}{seed}".encode()).hexdigest())
        else:
            import random
            random.shuffle(chunks)

        # Apply limit after shuffling/sorting
        if max_chunks and len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]

        logger.info(f"Retrieved {len(chunks)} chunks from Qdrant collection '{collection_name}'")
        return chunks

    async def _call_vllm(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Call local vLLM API for JSON generation."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # vLLM uses port 8000 inside docker network, but we connect via VLLM_BASE_URL
        # The backend connects to vLLM via the docker network alias "vllm"
        vllm_url = VLLM_BASE_URL.rstrip("/")
        if "localhost" in vllm_url or "127.0.0.1" in vllm_url:
            # Running locally, use the configured URL
            pass
        else:
            # Inside docker, use the service name
            vllm_url = "http://vllm:8000"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{vllm_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                raise Exception(f"vLLM API error: {response.status_code} - {response.text}")

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError(f"Could not parse JSON from vLLM response: {content[:500]}")

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
            if self.use_vllm:
                # Use local vLLM
                response = await self._call_vllm(
                    prompt=prompt,
                    system_prompt=QA_GENERATION_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=1024,
                )
            else:
                # Use OpenRouter
                response = await self.client.generate_json(
                    prompt=prompt,
                    system_prompt=QA_GENERATION_SYSTEM_PROMPT,
                    model=self.model,
                    temperature=self.temperature,
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
        collection_name, chunks = await self.get_chunks_for_dataset(
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
            "source_collection": collection_name,  # Track which collection this was generated from
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
        _, chunks = await self.get_chunks_for_dataset(
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
