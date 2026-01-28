"""Async HTTP client for benchmark API calls."""

import logging
from pathlib import Path
from typing import Any

import httpx

from scripts.benchmark.config import (
    ChunkingConfig,
    EmbedderConfig,
    EvaluationConfig,
    EvalMetrics,
    PreprocessingConfig,
    QAGenerationConfig,
)
from scripts.benchmark.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300.0  # 5 minutes for long operations


class BenchmarkAPIClient:
    """Async HTTP client for interacting with the benchmark API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API server
            timeout: Default request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "BenchmarkAPIClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        return self._client

    # =========================================================================
    # RAW DATASETS
    # =========================================================================

    async def create_raw_dataset(
        self,
        name: str,
        description: str | None = None,
    ) -> dict:
        """
        Create a new raw dataset.

        Args:
            name: Dataset name (1-100 chars)
            description: Optional description

        Returns:
            Created dataset info
        """
        payload = {"name": name, "source_type": "upload"}
        if description:
            payload["description"] = description

        response = await retry_with_backoff(
            self.client.post,
            "/api/raw-datasets",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def upload_file(
        self,
        dataset_id: int,
        file_path: Path,
    ) -> dict:
        """
        Upload a file to a raw dataset.

        Args:
            dataset_id: Raw dataset ID
            file_path: Path to file to upload

        Returns:
            Uploaded file info
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            response = await retry_with_backoff(
                self.client.post,
                f"/api/raw-datasets/{dataset_id}/files",
                files=files,
            )
        response.raise_for_status()
        return response.json()

    async def get_raw_dataset(self, dataset_id: int) -> dict:
        """Get raw dataset details."""
        response = await self.client.get(f"/api/raw-datasets/{dataset_id}")
        response.raise_for_status()
        return response.json()

    async def list_raw_datasets(self) -> list[dict]:
        """List all raw datasets."""
        response = await self.client.get("/api/raw-datasets")
        response.raise_for_status()
        data = response.json()
        return data.get("datasets", data) if isinstance(data, dict) else data

    # =========================================================================
    # PROCESSED DATASETS
    # =========================================================================

    async def create_processed_dataset(
        self,
        raw_dataset_id: int,
        name: str,
        description: str | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        embedder_config: EmbedderConfig | None = None,
        vector_backend: str = "pgvector",
    ) -> dict:
        """
        Create a processed dataset from a raw dataset.

        Args:
            raw_dataset_id: Source raw dataset ID
            name: Dataset name
            description: Optional description
            preprocessing_config: Preprocessing configuration
            embedder_config: Embedder configuration
            vector_backend: Vector store backend (pgvector|qdrant)

        Returns:
            Created processed dataset info
        """
        payload: dict[str, Any] = {
            "name": name,
            "raw_dataset_id": raw_dataset_id,
            "vector_backend": vector_backend,
        }

        if description:
            payload["description"] = description

        if preprocessing_config:
            payload["preprocessing_config"] = preprocessing_config.model_dump(mode="json")

        if embedder_config:
            payload["embedder_config"] = embedder_config.model_dump(mode="json")

        response = await retry_with_backoff(
            self.client.post,
            "/api/processed-datasets",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def start_processing(self, dataset_id: int) -> dict:
        """
        Start processing a dataset.

        Args:
            dataset_id: Processed dataset ID

        Returns:
            Processing status
        """
        response = await retry_with_backoff(
            self.client.post,
            f"/api/processed-datasets/{dataset_id}/process",
        )
        response.raise_for_status()
        return response.json()

    async def get_processing_status(self, dataset_id: int) -> dict:
        """
        Get processing status for a dataset.

        Args:
            dataset_id: Processed dataset ID

        Returns:
            Status dict with 'processing_status' field
        """
        response = await self.client.get(f"/api/processed-datasets/{dataset_id}/status")
        response.raise_for_status()
        return response.json()

    async def get_processed_dataset(self, dataset_id: int) -> dict:
        """Get processed dataset details."""
        response = await self.client.get(f"/api/processed-datasets/{dataset_id}")
        response.raise_for_status()
        return response.json()

    async def list_processed_datasets(self) -> list[dict]:
        """List all processed datasets."""
        response = await self.client.get("/api/processed-datasets")
        response.raise_for_status()
        data = response.json()
        return data.get("datasets", data) if isinstance(data, dict) else data

    # =========================================================================
    # Q&A GENERATION
    # =========================================================================

    async def generate_qa_pairs(
        self,
        processed_dataset_id: int,
        name: str,
        use_vllm: bool = True,
        model: str | None = None,
        pairs_per_chunk: int = 2,
        max_chunks: int | None = 50,
        temperature: float = 0.3,
        seed: int | None = None,
        system_prompt: str | None = None,
    ) -> dict:
        """
        Generate Q&A pairs from a processed dataset.

        Args:
            processed_dataset_id: Source processed dataset ID
            name: Name for the Q&A dataset
            use_vllm: Whether to use vLLM for generation
            model: Optional model override
            pairs_per_chunk: Number of Q&A pairs per chunk (1-5)
            max_chunks: Maximum chunks to process
            temperature: Generation temperature (0-1)
            seed: Random seed for reproducibility
            system_prompt: Optional custom system prompt

        Returns:
            Generated Q&A dataset info
        """
        payload: dict[str, Any] = {
            "processed_dataset_id": processed_dataset_id,
            "name": name,
            "use_vllm": use_vllm,
            "pairs_per_chunk": pairs_per_chunk,
            "temperature": temperature,
        }

        if model:
            payload["model"] = model
        if max_chunks is not None:
            payload["max_chunks"] = max_chunks
        if seed is not None:
            payload["seed"] = seed
        if system_prompt:
            payload["system_prompt"] = system_prompt

        response = await retry_with_backoff(
            self.client.post,
            "/api/evaluation/datasets/generate",
            json=payload,
            timeout=httpx.Timeout(1800.0),  # 30 min for generation
        )
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # EVALUATION DATASETS
    # =========================================================================

    async def list_eval_datasets(self) -> list[dict]:
        """List all evaluation datasets."""
        response = await self.client.get("/api/evaluation/datasets")
        response.raise_for_status()
        return response.json()

    async def get_eval_dataset(self, dataset_id: str) -> dict:
        """Get evaluation dataset details."""
        response = await self.client.get(f"/api/evaluation/datasets/{dataset_id}")
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # EVALUATION TASKS & RUNS
    # =========================================================================

    async def start_eval_task(
        self,
        eval_dataset_id: str,
        preprocessed_dataset_id: int,
        preset: str | None = None,
        chunking: ChunkingConfig | None = None,
        embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_rag: bool = True,
        use_colbert: bool = False,
        top_k: int = 5,
        temperature: float = 0.1,
        experiment_name: str | None = None,
    ) -> str:
        """
        Start an evaluation task.

        Args:
            eval_dataset_id: Evaluation dataset ID
            preprocessed_dataset_id: Processed dataset ID
            preset: Evaluation preset (quick|balanced|high_quality)
            chunking: Custom chunking config (overrides preset)
            embedder: Embedder model name
            use_rag: Whether to use RAG
            use_colbert: Whether to use ColBERT reranking
            top_k: Number of chunks to retrieve
            temperature: Generation temperature
            experiment_name: Optional experiment name

        Returns:
            Task ID for polling
        """
        payload: dict[str, Any] = {
            "eval_dataset_id": eval_dataset_id,
            "preprocessed_dataset_id": preprocessed_dataset_id,
            "use_rag": use_rag,
            "use_colbert": use_colbert,
            "top_k": top_k,
            "temperature": temperature,
            "embedder": embedder,
        }

        if preset:
            payload["preset"] = preset
        if chunking:
            payload["chunking"] = chunking.model_dump(mode="json")
        if experiment_name:
            payload["experiment_name"] = experiment_name

        response = await retry_with_backoff(
            self.client.post,
            "/api/evaluation/tasks/start",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["task_id"]

    async def get_eval_task(self, task_id: str) -> dict:
        """
        Get evaluation task status.

        Args:
            task_id: Task ID

        Returns:
            Task status dict
        """
        response = await self.client.get(f"/api/evaluation/tasks/{task_id}")
        response.raise_for_status()
        return response.json()

    async def get_eval_run(self, run_id: str) -> dict:
        """
        Get evaluation run details.

        Args:
            run_id: Run ID

        Returns:
            Run details including results and metrics
        """
        response = await self.client.get(f"/api/evaluation/runs/{run_id}")
        response.raise_for_status()
        return response.json()

    async def list_eval_runs(self) -> list[dict]:
        """List all evaluation runs."""
        response = await self.client.get("/api/evaluation/runs")
        response.raise_for_status()
        return response.json()

    async def export_run_csv(self, run_id: str, output_path: Path) -> Path:
        """
        Export evaluation run results to CSV.

        Args:
            run_id: Run ID
            output_path: Output file path

        Returns:
            Path to created CSV file
        """
        response = await self.client.get(f"/api/evaluation/runs/{run_id}/csv")
        response.raise_for_status()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Exported CSV to {output_path}")
        return output_path

    # =========================================================================
    # PRESETS
    # =========================================================================

    async def list_presets(self) -> list[dict]:
        """List available evaluation presets."""
        response = await self.client.get("/api/evaluation/presets")
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    async def health_check(self) -> bool:
        """
        Check if the API server is healthy.

        Returns:
            True if server is responding
        """
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
