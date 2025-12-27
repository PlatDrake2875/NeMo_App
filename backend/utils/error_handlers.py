"""
Error handling utilities for FastAPI routes.
Provides decorators and helpers for standardized error handling across routers.
"""

import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from fastapi import HTTPException

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def handle_service_errors(operation: str) -> Callable[[F], F]:
    """
    Decorator for standardized route error handling.

    Catches common exceptions and converts them to appropriate HTTP responses:
    - HTTPException: Re-raised as-is
    - ValueError: Converted to 400 Bad Request
    - Exception: Converted to 500 Internal Server Error

    Args:
        operation: Description of the operation for error messages (e.g., "create dataset")

    Example:
        @router.post("")
        @handle_service_errors("create dataset")
        async def create_dataset(request: DatasetCreate):
            return service.create(request)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                logger.error("Failed to %s: %s", operation, e, exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to {operation}: {str(e)}"
                ) from e
        return wrapper  # type: ignore
    return decorator


def require_found(
    resource: Optional[Any],
    resource_name: str = "Resource",
    resource_id: Optional[Any] = None,
) -> Any:
    """
    Raise 404 HTTPException if resource is None.

    Args:
        resource: The resource to check (typically a database query result)
        resource_name: Human-readable name for error messages (e.g., "Dataset", "User")
        resource_id: Optional ID to include in error message

    Returns:
        The resource if it's not None

    Raises:
        HTTPException: 404 if resource is None

    Example:
        dataset = require_found(
            service.get_by_id(dataset_id),
            "Dataset",
            dataset_id
        )
    """
    if resource is None:
        detail = f"{resource_name} not found"
        if resource_id is not None:
            detail = f"{resource_name} with id {resource_id} not found"
        raise HTTPException(status_code=404, detail=detail)
    return resource


def handle_not_found_errors(resource_name: str = "Resource") -> Callable[[F], F]:
    """
    Decorator for endpoints that should return 404 when result is None.

    Args:
        resource_name: Human-readable name for error messages

    Example:
        @router.get("/{id}")
        @handle_not_found_errors("Dataset")
        async def get_dataset(id: int):
            return service.get_by_id(id)  # Returns None if not found
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)
            if result is None:
                # Try to extract ID from kwargs
                resource_id = kwargs.get("id") or kwargs.get("dataset_id")
                return require_found(result, resource_name, resource_id)
            return result
        return wrapper  # type: ignore
    return decorator
