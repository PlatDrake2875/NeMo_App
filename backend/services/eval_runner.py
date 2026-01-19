"""
Evaluation Runner - Handles RAG inference execution.

Separates the inference phase from scoring, allowing:
- Re-scoring historical runs with new metrics
- Batch inference execution
- Dry-run validation
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from config import VLLM_BASE_URL, VLLM_MODEL
from rag_components import get_rag_context_prefix

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_strategy: str = "none"  # "none" or "colbert"
    top_k: int = 5
    temperature: float = 0.1
    dataset_id: Optional[str] = None
    collection_name: str = "rag_documents"
    use_rag: bool = True
    seed: Optional[int] = None  # Random seed for reproducibility

    def config_hash(self, dataset_content_hash: Optional[str] = None) -> str:
        """
        Generate SHA-256 hash for deduplication/reproducibility.

        Args:
            dataset_content_hash: Optional SHA-256 of dataset content for
                complete reproducibility (beyond just dataset ID).
        """
        config_dict = {
            "embedder_model": self.embedder_model,
            "reranker_strategy": self.reranker_strategy,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "dataset_id": self.dataset_id,
            "collection_name": self.collection_name,
            "use_rag": self.use_rag,
            "seed": self.seed,
        }
        # Include dataset content hash if available for stricter matching
        if dataset_content_hash:
            config_dict["dataset_content_hash"] = dataset_content_hash
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "embedder_model": self.embedder_model,
            "reranker_strategy": self.reranker_strategy,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "dataset_id": self.dataset_id,
            "collection_name": self.collection_name,
            "use_rag": self.use_rag,
            "seed": self.seed,
            "config_hash": self.config_hash(),
        }

    def get_determinism_warnings(self) -> list[str]:
        """Get warnings about non-deterministic settings."""
        warnings = []
        if self.temperature > 0:
            warnings.append(
                f"temperature={self.temperature} > 0 may produce non-deterministic results"
            )
        if self.seed is None:
            warnings.append("No seed set - results may not be reproducible")
        return warnings

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalConfig":
        """Create from dictionary."""
        return cls(
            embedder_model=data.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2"),
            reranker_strategy=data.get("reranker_strategy", "none"),
            top_k=data.get("top_k", 5),
            temperature=data.get("temperature", 0.1),
            dataset_id=data.get("dataset_id"),
            collection_name=data.get("collection_name", "rag_documents"),
            use_rag=data.get("use_rag", True),
            seed=data.get("seed"),
        )


@dataclass
class InferenceResult:
    """Result of running inference on a single Q&A pair."""

    query: str
    ground_truth: str
    predicted_answer: str
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    latency: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "predicted_answer": self.predicted_answer,
            "retrieved_chunks": self.retrieved_chunks,
            "latency": self.latency,
            "error": self.error,
        }


@dataclass
class EvalRunResult:
    """Complete result of an evaluation run."""

    run_id: str
    config: EvalConfig
    results: list[InferenceResult]
    created_at: str
    dataset_name: str = "Unknown"
    total_latency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "created_at": self.created_at,
            "dataset_name": self.dataset_name,
            "total_latency": self.total_latency,
        }


class EvalRunner:
    """
    Handles RAG inference execution for evaluation.

    Responsibilities:
    - Load evaluation datasets
    - Execute RAG pipeline on each query
    - Record predictions and retrieved chunks
    - Track latency and errors
    """

    def __init__(
        self,
        datasets_dir: Path = Path("data/evaluation_datasets"),
        runs_dir: Path = Path("data/evaluation_runs"),
    ):
        self.datasets_dir = datasets_dir
        self.runs_dir = runs_dir
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Load an evaluation dataset by ID."""
        path = self.datasets_dir / f"{dataset_id}.json"
        if not path.exists():
            raise ValueError(f"Dataset {dataset_id} not found")
        with open(path) as f:
            return json.load(f)

    async def run_inference(
        self,
        config: EvalConfig,
        pairs: list[dict[str, str]],
        progress_callback: Optional[callable] = None,
    ) -> list[InferenceResult]:
        """
        Run RAG inference on a list of Q&A pairs.

        Args:
            config: Evaluation configuration
            pairs: List of {"query": ..., "ground_truth": ...} dicts
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of InferenceResult objects
        """
        results = []
        total = len(pairs)

        for i, pair in enumerate(pairs):
            query = pair["query"]
            ground_truth = pair["ground_truth"]

            start_time = time.time()

            try:
                # Get RAG context and generate answer
                if config.use_rag:
                    rag_prefix = await get_rag_context_prefix(
                        query,
                        collection_name=config.collection_name,
                        use_colbert=config.reranker_strategy == "colbert",
                        embedder=config.embedder_model,
                    )
                    prompt_content = rag_prefix if rag_prefix else query
                else:
                    prompt_content = query

                # Generate answer
                predicted_answer, retrieved_chunks = await self._generate_answer(
                    prompt_content,
                    query,
                    config.temperature,
                    config.use_rag,
                )

                latency = time.time() - start_time

                results.append(
                    InferenceResult(
                        query=query,
                        ground_truth=ground_truth,
                        predicted_answer=predicted_answer,
                        retrieved_chunks=retrieved_chunks,
                        latency=latency,
                    )
                )

            except Exception as e:
                logger.error(f"Error on pair {i + 1}: {e}", exc_info=True)
                results.append(
                    InferenceResult(
                        query=query,
                        ground_truth=ground_truth,
                        predicted_answer=f"Error: {str(e)}",
                        latency=time.time() - start_time,
                        error=str(e),
                    )
                )

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    async def run_evaluation(
        self,
        config: EvalConfig,
        dataset_id: Optional[str] = None,
        test_query: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> EvalRunResult:
        """
        Run a complete evaluation.

        Args:
            config: Evaluation configuration
            dataset_id: Optional dataset ID to use
            test_query: Optional single test query (if no dataset)
            progress_callback: Optional progress callback

        Returns:
            EvalRunResult with all inference results
        """
        # Load pairs
        if dataset_id:
            dataset = self.load_dataset(dataset_id)
            pairs = dataset["pairs"]
            dataset_name = dataset["name"]
            config.dataset_id = dataset_id
        else:
            query = test_query or "What information is available in the documents?"
            pairs = [{"query": query, "ground_truth": "Based on retrieved documents."}]
            dataset_name = "Quick Test"

        logger.info(
            f"Starting evaluation: dataset='{dataset_name}', "
            f"pairs={len(pairs)}, config_hash={config.config_hash()}"
        )

        # Run inference
        start_time = time.time()
        results = await self.run_inference(config, pairs, progress_callback)
        total_latency = time.time() - start_time

        # Create run result
        import uuid

        run_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        run_result = EvalRunResult(
            run_id=run_id,
            config=config,
            results=results,
            created_at=created_at,
            dataset_name=dataset_name,
            total_latency=total_latency,
        )

        return run_result

    async def _generate_answer(
        self,
        prompt_content: str,
        original_query: str,
        temperature: float,
        use_rag: bool,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate an answer using vLLM."""
        messages = [{"role": "user", "content": prompt_content}]

        payload = {
            "model": VLLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json=payload,
            )

            if response.status_code != 200:
                raise Exception(f"vLLM error: {response.status_code}")

            data = response.json()
            answer = data["choices"][0]["message"]["content"]

            # Extract retrieved chunks from the prompt if RAG was used
            retrieved_chunks = []
            if use_rag and "Context:" in prompt_content:
                try:
                    context_start = prompt_content.find("Context:") + len("Context:")
                    question_start = prompt_content.find("Question:")
                    if question_start > context_start:
                        context_text = prompt_content[context_start:question_start].strip()
                        chunks = [c.strip() for c in context_text.split("\n\n") if c.strip()]
                        retrieved_chunks = [
                            {"content": chunk[:500], "source": "retrieved"}
                            for chunk in chunks[:5]
                        ]
                except Exception:
                    pass

            return answer, retrieved_chunks

    def dry_run(self, config: EvalConfig, dataset_id: Optional[str] = None) -> dict[str, Any]:
        """
        Validate configuration without running inference.

        Returns:
            Dict with config validation results
        """
        result = {
            "config_hash": config.config_hash(),
            "config": config.to_dict(),
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Validate dataset exists
        if dataset_id:
            try:
                dataset = self.load_dataset(dataset_id)
                result["dataset_name"] = dataset["name"]
                result["pair_count"] = len(dataset["pairs"])
            except ValueError as e:
                result["valid"] = False
                result["errors"].append(str(e))
        else:
            result["dataset_name"] = "Quick Test"
            result["pair_count"] = 1

        # Check for existing run with same config
        existing_run = self._find_existing_run(config.config_hash())
        if existing_run:
            result["existing_run_id"] = existing_run
            result["warnings"].append(f"Run with same config exists: {existing_run}")

        return result

    def _find_existing_run(self, config_hash: str) -> Optional[str]:
        """Check if a run with the same config hash already exists."""
        for path in self.runs_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    if data.get("config", {}).get("config_hash") == config_hash:
                        return data.get("id") or data.get("run_id")
            except Exception:
                pass
        return None


class BatchEvalRunner:
    """
    Handles batch evaluation of multiple configurations.
    """

    def __init__(self, runner: Optional[EvalRunner] = None):
        self.runner = runner or EvalRunner()

    async def run_batch(
        self,
        configs: list[EvalConfig],
        dataset_id: Optional[str] = None,
        parallel: bool = True,
    ) -> list[EvalRunResult]:
        """
        Run evaluations for multiple configurations.

        Args:
            configs: List of EvalConfig objects
            dataset_id: Optional dataset ID to use for all configs
            parallel: Whether to run configs in parallel

        Returns:
            List of EvalRunResult objects
        """
        if parallel:
            tasks = [
                self.runner.run_evaluation(config, dataset_id)
                for config in configs
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for config in configs:
                result = await self.runner.run_evaluation(config, dataset_id)
                results.append(result)
            return results
