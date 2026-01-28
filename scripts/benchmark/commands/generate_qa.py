"""Generate Q&A command - create evaluation dataset from processed dataset."""

import logging

from scripts.benchmark.client import BenchmarkAPIClient
from scripts.benchmark.config import GenerateQAResult

logger = logging.getLogger(__name__)


async def generate_qa(
    client: BenchmarkAPIClient,
    processed_dataset_id: int,
    name: str,
    use_vllm: bool = True,
    model: str | None = None,
    pairs_per_chunk: int = 2,
    max_chunks: int | None = 50,
    temperature: float = 0.3,
    seed: int | None = None,
    system_prompt: str | None = None,
) -> GenerateQAResult:
    """
    Generate Q&A evaluation dataset from a processed dataset.

    This is Stage 2 of the benchmark pipeline. It generates Q&A pairs
    from document chunks for use in RAG evaluation.

    Args:
        client: API client instance
        processed_dataset_id: Source processed dataset ID
        name: Name for the Q&A dataset
        use_vllm: Whether to use vLLM for generation
        model: Optional model override
        pairs_per_chunk: Number of Q&A pairs per chunk (1-5)
        max_chunks: Maximum chunks to process (None for all)
        temperature: Generation temperature (0-1)
        seed: Random seed for reproducibility
        system_prompt: Optional custom system prompt

    Returns:
        GenerateQAResult with eval_dataset_id and stats
    """
    logger.info(
        f"Generating Q&A dataset '{name}' from processed dataset {processed_dataset_id}"
    )
    logger.info(
        f"Config: pairs_per_chunk={pairs_per_chunk}, max_chunks={max_chunks}, "
        f"use_vllm={use_vllm}, seed={seed}"
    )

    result = await client.generate_qa_pairs(
        processed_dataset_id=processed_dataset_id,
        name=name,
        use_vllm=use_vllm,
        model=model,
        pairs_per_chunk=pairs_per_chunk,
        max_chunks=max_chunks,
        temperature=temperature,
        seed=seed,
        system_prompt=system_prompt,
    )

    logger.info(
        f"Generated Q&A dataset: ID={result['id']}, "
        f"pairs={result['pair_count']}, chunks_processed={result['chunks_processed']}"
    )

    return GenerateQAResult(
        eval_dataset_id=result["id"],
        name=result["name"],
        pair_count=result["pair_count"],
        chunks_processed=result["chunks_processed"],
    )
