"""
Retry handler with exponential backoff for API calls.

Handles rate limits (429), server errors (5xx), and timeouts.
Does NOT retry on client errors (400, 401, 403, 404).
"""

import random
import time
import logging
from typing import Callable, Any, List

logger = logging.getLogger("posas")


class APICallError(Exception):
    """Raised when an API call fails after all retries."""

    def __init__(self, model: str, status_code: int, message: str, retries: int):
        self.model = model
        self.status_code = status_code
        self.retries = retries
        super().__init__(f"{model} failed (HTTP {status_code}) after {retries} retries: {message}")


def retry_api_call(
    call_fn: Callable[[], Any],
    model_name: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
    jitter_max: float = 1.0,
    retry_on: List[int] = None,
    do_not_retry_on: List[int] = None,
) -> Any:
    """
    Execute an API call with exponential backoff retry logic.

    Args:
        call_fn: A zero-argument callable that makes the API call.
                 It should raise an exception with a status_code attribute on failure.
        model_name: Name of the model (for logging).
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).
        jitter_max: Maximum random jitter added to delay.
        retry_on: HTTP status codes that trigger a retry.
        do_not_retry_on: HTTP status codes that should NOT be retried.

    Returns:
        The result of call_fn() on success.

    Raises:
        APICallError: If all retries are exhausted or a non-retryable error occurs.
    """
    if retry_on is None:
        retry_on = [429, 500, 502, 503]
    if do_not_retry_on is None:
        do_not_retry_on = [400, 401, 403, 404]

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            result = call_fn()
            if attempt > 0:
                logger.info(f"{model_name}: Succeeded on retry {attempt}")
            return result

        except Exception as e:
            last_exception = e

            status_code = getattr(e, "status_code", None)
            if status_code is None:
                response = getattr(e, "response", None)
                if response is not None:
                    status_code = getattr(response, "status_code", None)

            if status_code and status_code in do_not_retry_on:
                logger.error(
                    f"{model_name}: Non-retryable error (HTTP {status_code}): {e}"
                )
                raise APICallError(model_name, status_code, str(e), attempt)

            if attempt >= max_retries:
                logger.error(
                    f"{model_name}: All {max_retries} retries exhausted. "
                    f"Last error: {e}"
                )
                raise APICallError(
                    model_name, status_code or 0, str(e), max_retries
                )

            delay = (base_delay * (2 ** attempt)) + random.uniform(0, jitter_max)

            logger.warning(
                f"{model_name}: Attempt {attempt + 1} failed "
                f"(HTTP {status_code or 'unknown'}). "
                f"Retrying in {delay:.1f}s... ({max_retries - attempt} retries left)"
            )
            time.sleep(delay)

    raise APICallError(model_name, 0, str(last_exception), max_retries)
