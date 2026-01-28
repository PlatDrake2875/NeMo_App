"""Async polling utilities with timeout."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PollingTimeoutError(Exception):
    """Raised when polling exceeds timeout."""

    def __init__(self, message: str, last_status: Any = None):
        super().__init__(message)
        self.last_status = last_status


class PollingError(Exception):
    """Raised when polled operation fails."""

    def __init__(self, message: str, status: Any = None):
        super().__init__(message)
        self.status = status


async def poll_until_complete(
    status_func: Callable[[Any], Awaitable[dict]],
    resource_id: Any,
    *,
    timeout: float = 3600.0,  # 1 hour default
    interval: float = 2.0,
    status_field: str = "status",
    complete_statuses: set[str] | None = None,
    failed_statuses: set[str] | None = None,
    error_field: str | None = "error",
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    Poll a status endpoint until completion or failure.

    Args:
        status_func: Async function that takes resource_id and returns status dict
        resource_id: ID of the resource to poll
        timeout: Maximum time to wait in seconds
        interval: Time between polls in seconds
        status_field: Field name containing status value
        complete_statuses: Set of status values indicating completion
        failed_statuses: Set of status values indicating failure
        error_field: Field name containing error message (if any)
        progress_callback: Optional callback for progress updates

    Returns:
        Final status dict

    Raises:
        PollingTimeoutError: When timeout is exceeded
        PollingError: When status indicates failure
    """
    if complete_statuses is None:
        complete_statuses = {"completed", "complete", "done", "success", "finished"}
    if failed_statuses is None:
        failed_statuses = {"failed", "error", "cancelled", "canceled"}

    start_time = asyncio.get_event_loop().time()
    last_status: dict | None = None

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            raise PollingTimeoutError(
                f"Polling timed out after {timeout:.0f}s",
                last_status=last_status,
            )

        try:
            status = await status_func(resource_id)
            last_status = status

            if progress_callback:
                progress_callback(status)

            current_status = status.get(status_field, "").lower()

            if current_status in complete_statuses:
                logger.info(f"Operation completed with status: {current_status}")
                return status

            if current_status in failed_statuses:
                error_msg = status.get(error_field, "Unknown error") if error_field else None
                raise PollingError(
                    f"Operation failed with status '{current_status}': {error_msg}",
                    status=status,
                )

            logger.debug(f"Status: {current_status}, elapsed: {elapsed:.0f}s")

        except (PollingTimeoutError, PollingError):
            raise
        except Exception as e:
            logger.warning(f"Error polling status: {e}")
            # Continue polling on transient errors

        await asyncio.sleep(interval)


async def poll_task(
    get_task_func: Callable[[str], Awaitable[dict]],
    task_id: str,
    *,
    timeout: float = 3600.0,
    interval: float = 2.0,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    Specialized polling for background tasks.

    Args:
        get_task_func: Async function to get task status
        task_id: Task ID to poll
        timeout: Maximum time to wait
        interval: Time between polls
        progress_callback: Optional progress callback

    Returns:
        Final task dict with results
    """
    return await poll_until_complete(
        get_task_func,
        task_id,
        timeout=timeout,
        interval=interval,
        status_field="status",
        complete_statuses={"completed", "complete"},
        failed_statuses={"failed", "error", "cancelled"},
        error_field="error",
        progress_callback=progress_callback,
    )
