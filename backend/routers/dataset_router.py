"""
Dataset router - API endpoints for managing datasets.
Provides CRUD operations for dataset configurations and metadata.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict

from schemas import (
    DatasetCreateRequest,
    DatasetInfo,
    DatasetListResponse,
)
from services.dataset_registry import DatasetRegistryService


router = APIRouter(prefix="/datasets", tags=["datasets"])


def get_dataset_registry_service() -> DatasetRegistryService:
    """Dependency injection for DatasetRegistryService."""
    return DatasetRegistryService()


@router.post("", response_model=DatasetInfo, status_code=201)
async def create_dataset(
    request: DatasetCreateRequest,
    service: DatasetRegistryService = Depends(get_dataset_registry_service)
) -> DatasetInfo:
    """
    Create a new dataset with the specified configuration.

    Args:
        request: Dataset creation request with name, description, and embedder config

    Returns:
        DatasetInfo: Created dataset information including metadata

    Raises:
        HTTPException 400: If dataset with the same name already exists
        HTTPException 500: If dataset creation fails
    """
    return service.create_dataset(request)


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    service: DatasetRegistryService = Depends(get_dataset_registry_service)
) -> DatasetListResponse:
    """
    List all datasets with their metadata.

    Returns:
        DatasetListResponse: List of all datasets with counts and configurations

    Raises:
        HTTPException 500: If listing datasets fails
    """
    return service.list_datasets()


@router.get("/{name}", response_model=DatasetInfo)
async def get_dataset(
    name: str,
    service: DatasetRegistryService = Depends(get_dataset_registry_service)
) -> DatasetInfo:
    """
    Get information about a specific dataset.

    Args:
        name: Dataset name

    Returns:
        DatasetInfo: Dataset information including metadata and stats

    Raises:
        HTTPException 404: If dataset not found
        HTTPException 500: If retrieval fails
    """
    return service.get_dataset(name)


@router.delete("/{name}", response_model=Dict[str, str])
async def delete_dataset(
    name: str,
    service: DatasetRegistryService = Depends(get_dataset_registry_service)
) -> Dict[str, str]:
    """
    Delete a dataset and its associated collection.

    Args:
        name: Dataset name

    Returns:
        Dict with success message

    Raises:
        HTTPException 404: If dataset not found
        HTTPException 500: If deletion fails
    """
    return service.delete_dataset(name)


@router.post("/{name}/refresh", response_model=DatasetInfo)
async def refresh_dataset_counts(
    name: str,
    service: DatasetRegistryService = Depends(get_dataset_registry_service)
) -> DatasetInfo:
    """
    Refresh the document and chunk counts for a dataset.

    Args:
        name: Dataset name

    Returns:
        DatasetInfo: Updated dataset information

    Raises:
        HTTPException 404: If dataset not found
        HTTPException 500: If refresh fails
    """
    service.update_dataset_counts(name)
    return service.get_dataset(name)
