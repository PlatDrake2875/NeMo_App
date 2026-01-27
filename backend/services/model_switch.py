"""
Model Switch Service - Handles vLLM model switching via Docker control.

Key responsibilities:
1. Track model switch tasks in database
2. Execute Docker container restart with new model
3. Monitor vLLM health during switch
4. Provide progress updates

Safeguards:
- Only allows switching vllm-gpu or vllm-cpu containers
- Singleton lock prevents concurrent switches
- Rate limiting prevents rapid switching
- Blocks during active evaluations
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import POSTGRES_CONNECTION_STRING, VLLM_BASE_URL
from database_models import ModelSwitchTask
from enums import ModelSwitchStatus

logger = logging.getLogger(__name__)


class ModelSwitchService:
    """
    Service for managing model switch operations.

    Uses Docker to restart the vLLM container with a different model.
    Tracks progress in database for recovery and UI updates.
    """

    # Safeguard: only these containers can be controlled
    ALLOWED_CONTAINERS = ["vllm-gpu", "vllm-cpu"]

    # Rate limit: minimum seconds between switches
    MIN_SWITCH_INTERVAL = 60

    # Timeout waiting for vLLM to be ready
    VLLM_READY_TIMEOUT = 300  # 5 minutes

    _instance: Optional["ModelSwitchService"] = None
    _running_task: Optional[asyncio.Task] = None
    _current_task_id: Optional[str] = None
    _last_switch_time: Optional[datetime] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._engine = create_engine(POSTGRES_CONNECTION_STRING)
            self._session_maker = sessionmaker(bind=self._engine)
            self._running_task = None
            self._current_task_id = None
            self._last_switch_time = None
            self._docker_client = None
            self._initialized = True

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self._session_maker()

    def _get_docker_client(self):
        """Lazy-load Docker client."""
        if self._docker_client is None:
            try:
                import docker
                self._docker_client = docker.from_env()
            except Exception as e:
                logger.error(f"Failed to connect to Docker: {e}")
                raise RuntimeError(
                    "Cannot connect to Docker. Make sure docker.sock is mounted."
                )
        return self._docker_client

    def is_switch_in_progress(self) -> bool:
        """Check if a model switch is currently in progress."""
        return self._current_task_id is not None

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """Get the currently running switch task if any."""
        if self._current_task_id:
            return self.get_task(self._current_task_id)
        return None

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by ID."""
        with self._get_session() as session:
            task = session.query(ModelSwitchTask).filter_by(id=task_id).first()
            if task:
                return task.to_dict()
        return None

    def _check_rate_limit(self) -> bool:
        """Check if enough time has passed since last switch."""
        if self._last_switch_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_switch_time).total_seconds()
        return elapsed >= self.MIN_SWITCH_INTERVAL

    async def _check_active_operations(self) -> bool:
        """Check if there are active evaluations that would be disrupted."""
        try:
            from services.evaluation_task_service import get_evaluation_task_service
            eval_service = get_evaluation_task_service()
            running_tasks = eval_service.list_tasks(status="running")
            return len(running_tasks) == 0
        except Exception as e:
            logger.warning(f"Could not check active operations: {e}")
            return True  # Allow if can't check

    async def start_switch(
        self,
        target_model: str,
        current_model: Optional[str] = None,
    ) -> str:
        """
        Start a model switch operation.

        Args:
            target_model: HuggingFace model ID to switch to
            current_model: Currently loaded model (for display)

        Returns:
            Task ID for tracking progress

        Raises:
            ValueError: If a switch is already in progress or rate limited
            RuntimeError: If evaluations are running
        """
        # Check if switch already in progress
        if self.is_switch_in_progress():
            raise ValueError("A model switch is already in progress")

        # Check rate limit
        if not self._check_rate_limit():
            raise ValueError(
                f"Please wait {self.MIN_SWITCH_INTERVAL} seconds between model switches"
            )

        # Check for active operations
        if not await self._check_active_operations():
            raise RuntimeError(
                "Cannot switch models while evaluations are running. "
                "Please wait for them to complete or cancel them."
            )

        task_id = str(uuid.uuid4())

        # Estimate time based on whether model is cached
        is_cached = await self._check_model_cached(target_model)
        estimated_time = 90 if is_cached else 300  # 1.5 min cached, 5 min download

        with self._get_session() as session:
            task = ModelSwitchTask(
                id=task_id,
                from_model=current_model,
                to_model=target_model,
                status=ModelSwitchStatus.PENDING.value,
                progress=0,
                current_step="Initializing model switch...",
                estimated_time=estimated_time,
            )
            session.add(task)
            session.commit()

        # Start background task
        self._current_task_id = task_id
        self._running_task = asyncio.create_task(
            self._execute_switch(task_id, target_model)
        )

        logger.info(f"Started model switch task {task_id}: {current_model} -> {target_model}")
        return task_id

    async def _check_model_cached(self, model_id: str) -> bool:
        """Check if model exists in HuggingFace cache."""
        cache_dir = os.getenv("HF_HOME", "/root/.cache/huggingface")
        model_cache_name = model_id.replace("/", "--")
        model_path = Path(cache_dir) / "hub" / f"models--{model_cache_name}"
        return model_path.exists()

    async def _execute_switch(self, task_id: str, target_model: str):
        """Execute the model switch in background."""
        try:
            # Phase 1: Check model availability
            self._update_progress(
                task_id, ModelSwitchStatus.CHECKING, 5,
                "Checking if model is cached..."
            )

            is_cached = await self._check_model_cached(target_model)

            if not is_cached:
                # Phase 2: Download model first (would need HF download service)
                self._update_progress(
                    task_id, ModelSwitchStatus.DOWNLOADING, 10,
                    f"Model not cached. Please download {target_model} first."
                )
                raise ValueError(
                    f"Model {target_model} is not cached. "
                    "Please download it first via the Models page."
                )

            # Phase 3: Stop current vLLM container
            self._update_progress(
                task_id, ModelSwitchStatus.STOPPING, 30,
                "Stopping current vLLM service..."
            )
            stopped_container = await self._stop_vllm_container()

            if stopped_container:
                logger.info(f"Stopped container: {stopped_container}")

            await asyncio.sleep(3)  # Brief pause for clean shutdown

            # Phase 4: Start with new model
            self._update_progress(
                task_id, ModelSwitchStatus.STARTING, 50,
                f"Starting vLLM with {target_model}..."
            )
            await self._start_vllm_container(target_model)

            # Phase 5: Wait for model to load
            self._update_progress(
                task_id, ModelSwitchStatus.LOADING, 60,
                "Loading model into GPU memory..."
            )
            await self._wait_for_vllm_ready(task_id)

            # Success!
            self._update_progress(
                task_id, ModelSwitchStatus.READY, 100,
                f"Model {target_model} is ready!"
            )

            with self._get_session() as session:
                task = session.query(ModelSwitchTask).filter_by(id=task_id).first()
                if task:
                    task.status = ModelSwitchStatus.READY.value
                    task.completed_at = datetime.now(timezone.utc)
                    session.commit()

            self._last_switch_time = datetime.now(timezone.utc)
            logger.info(f"Model switch {task_id} completed successfully")

        except asyncio.CancelledError:
            self._update_progress(
                task_id, ModelSwitchStatus.CANCELLED, 0,
                "Switch cancelled by user"
            )
            logger.info(f"Model switch {task_id} was cancelled")

        except Exception as e:
            logger.error(f"Model switch {task_id} failed: {e}", exc_info=True)
            self._update_progress(
                task_id, ModelSwitchStatus.FAILED, 0,
                f"Failed: {str(e)}",
                error=str(e)
            )

        finally:
            self._current_task_id = None
            self._running_task = None

    async def _stop_vllm_container(self) -> Optional[str]:
        """Stop and remove the vLLM container via Docker API."""
        try:
            client = self._get_docker_client()

            for container in client.containers.list(all=True):  # Include stopped containers
                container_name = container.name
                # Check if this is a vLLM container
                if any(allowed in container_name for allowed in self.ALLOWED_CONTAINERS):
                    logger.info(f"Stopping and removing container: {container_name}")
                    container.stop(timeout=30)
                    container.remove(force=True)
                    return container_name

            logger.warning("No vLLM container found")
            return None

        except Exception as e:
            logger.error(f"Error stopping vLLM container: {e}")
            raise RuntimeError(f"Failed to stop vLLM container: {e}")

    async def _start_vllm_container(self, model_id: str):
        """Start vLLM container with the new model using Docker SDK."""
        try:
            client = self._get_docker_client()

            # Container configuration for vllm-gpu
            container_name = "nemo_app-vllm-gpu-1"  # Docker compose naming convention
            image = "vllm/vllm-openai:latest"

            # Get HuggingFace token from environment
            hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", "")

            # vLLM command arguments
            command = [
                f"--model={model_id}",
                f"--served-model-name={model_id}",
                "--host=0.0.0.0",
                "--port=8000",
                "--trust-remote-code",
                "--max-model-len=8192",
                "--max-num-seqs=32",
                "--gpu-memory-utilization=0.7",  # Leave ~30% GPU memory for ColBERT reranking
            ]

            # Environment variables
            environment = {
                "HUGGING_FACE_HUB_TOKEN": hf_token,
                "HF_HOME": "/root/.cache/huggingface",
            }

            # Get the data directory for volume mount
            data_dir = os.getenv("DATA_DIR", "/host_project/data")

            # Volume mounts - use the same cache as docker-compose
            volumes = {
                f"{data_dir}/vllm_cache": {
                    "bind": "/root/.cache/huggingface",
                    "mode": "rw"
                }
            }

            # Remove existing container if it exists (stopped state)
            try:
                old_container = client.containers.get(container_name)
                old_container.remove(force=True)
                logger.info(f"Removed old container: {container_name}")
            except Exception:
                pass  # Container doesn't exist, which is fine

            # GPU device request
            device_requests = [
                {
                    "Driver": "nvidia",
                    "Count": -1,  # All GPUs
                    "Capabilities": [["gpu"]],
                }
            ]

            # Network name
            network_name = "nemo_app_default"

            # Create container (not started yet)
            container = client.containers.create(
                image=image,
                name=container_name,
                command=command,
                environment=environment,
                volumes=volumes,
                ports={"8000/tcp": 8002},
                detach=True,
                device_requests=device_requests,
                shm_size="1g",
            )

            # Connect to network with alias "vllm" so backend can reach it
            network = client.networks.get(network_name)
            network.connect(container, aliases=["vllm", "vllm-gpu"])

            # Start the container
            container.start()

            logger.info(f"Started vLLM container {container.id[:12]} with model: {model_id}")

        except Exception as e:
            logger.error(f"Error starting vLLM container: {e}")
            raise RuntimeError(f"Failed to start vLLM container: {e}")

    async def _wait_for_vllm_ready(self, task_id: str):
        """Wait for vLLM to be ready, updating progress."""
        start_time = asyncio.get_event_loop().time()
        check_interval = 5  # seconds

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > self.VLLM_READY_TIMEOUT:
                raise TimeoutError(
                    f"vLLM failed to start within {self.VLLM_READY_TIMEOUT} seconds"
                )

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{VLLM_BASE_URL}/v1/models")
                    if response.status_code == 200:
                        logger.info("vLLM is ready!")
                        return
            except Exception:
                pass

            # Update progress (60-95% range during loading)
            progress = min(95, 60 + int((elapsed / self.VLLM_READY_TIMEOUT) * 35))
            self._update_progress(
                task_id, ModelSwitchStatus.LOADING, progress,
                f"Waiting for model to load... ({int(elapsed)}s)"
            )

            await asyncio.sleep(check_interval)

    def _update_progress(
        self,
        task_id: str,
        status: ModelSwitchStatus,
        progress: int,
        step: str,
        error: Optional[str] = None
    ):
        """Update task progress in database."""
        with self._get_session() as session:
            task = session.query(ModelSwitchTask).filter_by(id=task_id).first()
            if task:
                task.status = status.value
                task.progress = progress
                task.current_step = step
                if error:
                    task.error_message = error
                if status == ModelSwitchStatus.STOPPING and not task.started_at:
                    task.started_at = datetime.now(timezone.utc)
                session.commit()

    async def cancel_switch(self, task_id: str) -> bool:
        """Cancel an in-progress switch."""
        if self._current_task_id == task_id and self._running_task:
            self._running_task.cancel()
            return True
        return False

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """List models that are cached in the HuggingFace cache."""
        cache_dir = os.getenv("HF_HOME", "/root/.cache/huggingface")
        hub_dir = Path(cache_dir) / "hub"

        cached_models = []

        if not hub_dir.exists():
            return cached_models

        for model_dir in hub_dir.glob("models--*"):
            # Parse model ID from directory name
            # Format: models--org--model-name
            model_id = model_dir.name.replace("models--", "").replace("--", "/", 1)

            # Calculate size
            try:
                size_bytes = sum(
                    f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                )
                size_gb = round(size_bytes / (1024**3), 2)
            except Exception:
                size_gb = 0

            # Get modification time
            try:
                cached_at = datetime.fromtimestamp(
                    model_dir.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except Exception:
                cached_at = None

            cached_models.append({
                "model_id": model_id,
                "size_gb": size_gb,
                "cached_at": cached_at,
            })

        return cached_models

    def list_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent model switch tasks."""
        with self._get_session() as session:
            tasks = (
                session.query(ModelSwitchTask)
                .order_by(ModelSwitchTask.created_at.desc())
                .limit(limit)
                .all()
            )
            return [task.to_dict() for task in tasks]


# Singleton instance
_service: Optional[ModelSwitchService] = None


def get_model_switch_service() -> ModelSwitchService:
    """Get the singleton model switch service instance."""
    global _service
    if _service is None:
        _service = ModelSwitchService()
    return _service
