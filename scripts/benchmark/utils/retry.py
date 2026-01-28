"""Retry utilities with exponential backoff."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import httpx

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# Status codes to retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception


async def retry_with_backoff(
    func: Callable[P, Awaitable[T]],
    *args: P.args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    **kwargs: P.kwargs,
) -> T:
    """
    Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful function execution

    Raises:
        RetryError: When all retry attempts are exhausted
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in RETRYABLE_STATUS_CODES:
                raise
            last_exception = e
            logger.warning(
                f"HTTP {e.response.status_code} on attempt {attempt + 1}/{max_retries + 1}"
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_exception = e
            logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries + 1}: {e}")
        except Exception:
            # Don't retry unknown exceptions
            raise

        if attempt < max_retries:
            delay = min(base_delay * (exponential_base**attempt), max_delay)
            logger.info(f"Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

    raise RetryError(
        f"All {max_retries + 1} attempts failed",
        last_exception=last_exception,
    )


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator to add retry with exponential backoff to an async function.

    Usage:
        @with_retry(max_retries=5)
        async def my_api_call():
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                **kwargs,
            )

        return wrapper

    return decorator
