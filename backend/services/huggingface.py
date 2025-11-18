"""HuggingFace Hub integration service for model validation and download."""

import asyncio
import json
from typing import AsyncGenerator, Optional

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    model_info,
    repo_info,
    snapshot_download,
)
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from config import HUGGING_FACE_HUB_TOKEN


class ModelMetadata:
    """Model metadata from HuggingFace."""

    def __init__(
        self,
        model_id: str,
        is_gated: bool,
        size_bytes: int,
        downloads: int,
        pipeline_tag: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ):
        self.model_id = model_id
        self.is_gated = is_gated
        self.size_bytes = size_bytes
        self.downloads = downloads
        self.pipeline_tag = pipeline_tag
        self.tags = tags or []

    @property
    def size_gb(self) -> float:
        """Return size in GB."""
        return self.size_bytes / (1024**3)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "is_gated": self.is_gated,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_gb, 2),
            "downloads": self.downloads,
            "pipeline_tag": self.pipeline_tag,
            "tags": self.tags,
        }


class HuggingFaceService:
    """Service for HuggingFace Hub operations."""

    def __init__(self, token: Optional[str] = None):
        """Initialize HuggingFace service.

        Args:
            token: HuggingFace API token for accessing gated models
        """
        self.token = token or HUGGING_FACE_HUB_TOKEN
        self.api = HfApi(token=self.token)

    async def validate_model(self, model_id: str) -> tuple[bool, Optional[str]]:
        """Validate if a model exists on HuggingFace.

        Args:
            model_id: HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B')

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Run blocking API call in executor to avoid blocking event loop
            await asyncio.to_thread(model_info, model_id, token=self.token)
            return True, None
        except RepositoryNotFoundError:
            return False, f"Model '{model_id}' not found on HuggingFace Hub"
        except GatedRepoError:
            return (
                False,
                f"Model '{model_id}' is gated and requires authentication. Please provide a HuggingFace token.",
            )
        except Exception as e:
            return False, f"Error validating model: {str(e)}"

    async def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Get model metadata from HuggingFace.

        Args:
            model_id: HuggingFace model ID

        Returns:
            ModelMetadata object

        Raises:
            ValueError: If model validation fails
        """
        is_valid, error = await self.validate_model(model_id)
        if not is_valid:
            raise ValueError(error)

        try:
            # Get model info
            info = await asyncio.to_thread(model_info, model_id, token=self.token)

            # Calculate total size from safetensors files
            size_bytes = 0
            if hasattr(info, "siblings") and info.siblings:
                for file in info.siblings:
                    if hasattr(file, "size") and file.size:
                        # Focus on model weight files
                        if any(
                            file.rfilename.endswith(ext)
                            for ext in [
                                ".safetensors",
                                ".bin",
                                ".pt",
                                ".pth",
                                ".gguf",
                            ]
                        ):
                            size_bytes += file.size

            # Check if gated
            is_gated = getattr(info, "gated", False)

            # Get download count
            downloads = getattr(info, "downloads", 0)

            # Get pipeline tag and tags
            pipeline_tag = getattr(info, "pipeline_tag", None)
            tags = getattr(info, "tags", [])

            return ModelMetadata(
                model_id=model_id,
                is_gated=is_gated,
                size_bytes=size_bytes,
                downloads=downloads,
                pipeline_tag=pipeline_tag,
                tags=tags,
            )
        except Exception as e:
            raise ValueError(f"Error fetching model metadata: {str(e)}")

    async def download_model_with_progress(
        self, model_id: str, token: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """Download a model from HuggingFace with progress updates.

        Args:
            model_id: HuggingFace model ID
            token: Optional HuggingFace token for gated models

        Yields:
            Progress updates as dictionaries with structure:
            {
                "stage": "validating" | "downloading" | "complete" | "error",
                "progress": 0-100,
                "message": "Status message",
                "error": "Error message" (only if stage == "error")
            }
        """
        use_token = token or self.token

        try:
            # Stage 1: Validation
            yield {
                "stage": "validating",
                "progress": 0,
                "message": f"Validating model '{model_id}'...",
            }

            metadata = await self.get_model_metadata(model_id)

            yield {
                "stage": "validating",
                "progress": 10,
                "message": f"Model found: {metadata.size_gb:.2f} GB",
            }

            # Stage 2: Downloading
            yield {
                "stage": "downloading",
                "progress": 15,
                "message": f"Starting download of {model_id}...",
            }

            # Download model using snapshot_download for full model
            # This runs in a thread to avoid blocking
            def download_task():
                return snapshot_download(
                    repo_id=model_id,
                    token=use_token,
                    # Cache directory will be default HF cache
                    local_dir=None,
                    # Download all files
                    ignore_patterns=None,
                )

            # Simulate progress updates during download
            # In a production environment, you'd want to track actual download progress
            # by monitoring the cache directory or using custom callbacks
            download_task_obj = asyncio.create_task(
                asyncio.to_thread(download_task)
            )

            # Provide progress updates while downloading
            progress_steps = [20, 30, 40, 50, 60, 70, 80, 90]
            for i, progress in enumerate(progress_steps):
                if download_task_obj.done():
                    break
                yield {
                    "stage": "downloading",
                    "progress": progress,
                    "message": f"Downloading model files... ({progress}%)",
                }
                await asyncio.sleep(2)  # Update every 2 seconds

            # Wait for download to complete
            local_path = await download_task_obj

            yield {
                "stage": "downloading",
                "progress": 95,
                "message": "Download complete, finalizing...",
            }

            # Stage 3: Complete
            yield {
                "stage": "complete",
                "progress": 100,
                "message": f"Model '{model_id}' downloaded successfully to {local_path}",
                "local_path": local_path,
            }

        except GatedRepoError:
            yield {
                "stage": "error",
                "progress": 0,
                "message": "Authentication required",
                "error": f"Model '{model_id}' is gated. Please provide a valid HuggingFace token.",
            }
        except RepositoryNotFoundError:
            yield {
                "stage": "error",
                "progress": 0,
                "message": "Model not found",
                "error": f"Model '{model_id}' does not exist on HuggingFace Hub.",
            }
        except Exception as e:
            yield {
                "stage": "error",
                "progress": 0,
                "message": "Download failed",
                "error": f"Failed to download model: {str(e)}",
            }


# Singleton instance
_hf_service: Optional[HuggingFaceService] = None


def get_hf_service(token: Optional[str] = None) -> HuggingFaceService:
    """Get or create HuggingFace service instance.

    Args:
        token: Optional HuggingFace token

    Returns:
        HuggingFaceService instance
    """
    global _hf_service
    if _hf_service is None or token is not None:
        _hf_service = HuggingFaceService(token=token)
    return _hf_service
