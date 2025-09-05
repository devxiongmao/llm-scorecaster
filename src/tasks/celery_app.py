"""
Celery application configuration for async task processing.

This module configures Celery with Redis as both broker and result backend
for handling asynchronous metric computation tasks.
"""

import asyncio
import time
import logging
import sys
from typing import List, Dict, Any
import httpx

from celery import Celery, signals

from src.core.metrics.registry import metric_registry
from src.models.schemas import TextPair, MetricsRequest, TextPairResult
from src.core.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,  # This forces reconfiguration of the root logger
)

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "llm_scorecaster",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["src.tasks.celery_app"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Result settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    # Retry settings
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    task_max_retries=3,
)


@signals.setup_logging.connect
def setup_celery_logging(**_kwargs):
    """Override Celery's logging setup to prevent interference."""


@signals.task_prerun.connect
def task_prerun_handler(task, **_kwargs):
    """
    Handle task pre-run signal.

    Logs task start and records the start time in task headers
    for duration calculation.

    Args:
        task: The Celery task instance
        **_kwargs: Additional keyword arguments (unused)
    """
    if task.request.headers is None:
        task.request.headers = {}
    task.request.headers["started_at"] = time.time()
    logger.info("Task Started - Task ID: %s, Task Name: %s", task.request.id, task.name)


@signals.task_success.connect
def task_success_handler(sender, **_kwargs):
    """
    Handle task success signal.

    Logs successful task completion with duration information.

    Args:
        sender: The task instance that completed successfully
        **_kwargs: Additional keyword arguments (unused)
    """
    started_at = sender.request.headers.get("started_at")
    duration = time.time() - started_at if started_at else None
    logger.info(
        "Task Completed - Task ID: %s, Task Name: %s, Duration: %.3fs",
        sender.request.id,
        sender.name,
        duration or 0,
    )


@signals.task_retry.connect
def task_retry_handler(sender, reason, **_kwargs):
    """
    Handle task retry signal.

    Logs task retry attempts with the reason and duration information.

    Args:
        sender: The task instance being retried
        reason: The reason for the retry
        **_kwargs: Additional keyword arguments (unused)
    """
    started_at = sender.request.headers.get("started_at")
    duration = time.time() - started_at if started_at else None
    logger.error(
        "Task Retry - Task ID: %s, Task Name: %s, Reason: %s, Duration: %.3fs",
        sender.request.id,
        sender.name,
        str(reason),
        duration or 0,
    )


@signals.task_failure.connect
def task_failure_handler(sender, exception, _traceback, **_kwargs):
    """
    Handle task failure signal.

    Logs task failures with exception details and duration information.

    Args:
        sender: The task instance that failed
        exception: The exception that caused the failure
        _traceback: The traceback object (unused)
        **_kwargs: Additional keyword arguments (unused)
    """
    started_at = sender.request.headers.get("started_at")
    duration = time.time() - started_at if started_at else None
    logger.error(
        "Task Failed - Task ID: %s, Task Name: %s, Exception: %s, Duration: %.3fs",
        sender.request.id,
        sender.name,
        str(exception),
        duration or 0,
    )


def compute_metrics_for_request(request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compute metrics for a request. This is the same logic as the sync version
    but adapted to work with serialized data.
    """
    # Reconstruct the request from serialized data
    request = MetricsRequest(**request_data)

    # Discover and get the requested metrics
    metric_registry.discover_metrics()
    metrics = metric_registry.get_metrics([m.value for m in request.metrics])

    results = []

    # Convert Pydantic TextPairs to the format metrics expect
    text_pairs = [
        TextPair(reference=pair.reference, candidate=pair.candidate)
        for pair in request.text_pairs
    ]

    for pair_idx, text_pair in enumerate(text_pairs):
        pair_results = []

        # Compute each requested metric for this text pair
        for _, metric_instance in metrics.items():
            result = metric_instance.compute_single(
                text_pair.reference, text_pair.candidate
            )
            pair_results.append(result)

        # Create the response format (serialize to dict for Celery)
        result_data = TextPairResult(
            pair_index=pair_idx,
            reference=text_pair.reference,
            candidate=text_pair.candidate,
            metrics=pair_results,
        )
        results.append(result_data.model_dump())

    return results


async def send_webhook_notification(
    webhook_url: str,
    job_id: str,
    result_data: Dict[str, Any],
) -> bool:
    """
    Send webhook notification with results.

    Args:
        webhook_url: The URL to send the webhook to
        job_id: The job ID for reference
        result_data: The computed results to send

    Returns:
        bool: True if successful, False otherwise
    """
    webhook_payload = {
        "job_id": job_id,
        "status": "COMPLETED" if result_data.get("success") else "FAILED",
        "timestamp": time.time(),
        "data": result_data,
    }

    for attempt in range(settings.max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.max_timeout) as client:
                response = await client.post(
                    webhook_url,
                    json=webhook_payload,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code in [200, 201, 202, 204]:
                    logger.info(
                        "Webhook sent successfully for job %s to %s",
                        job_id,
                        webhook_url,
                    )
                    return True

                logger.warning(
                    "Webhook attempt %d failed for job %s: HTTP %d - %s",
                    attempt + 1,
                    job_id,
                    response.status_code,
                    response.text,
                )

        except httpx.TimeoutException:
            logger.warning(
                "Webhook attempt %d timed out for job %s", attempt + 1, job_id
            )
        except Exception as e:
            logger.warning(
                "Webhook attempt %d failed for job %s: %s", attempt + 1, job_id, str(e)
            )

        if attempt < settings.max_retries:
            await asyncio.sleep(2**attempt)  # Exponential backoff

    logger.error("All webhook attempts failed for job %s", job_id)
    return False


def _compute_metrics_task_logic(self, request_data):
    try:
        start_time = time.time()

        # Update task state to indicate processing has started
        self.update_state(
            state="PROCESSING",
            meta={
                "message": "Computing metrics...",
                "progress": 0,
                "total_pairs": len(request_data.get("text_pairs", [])),
                "total_metrics": len(request_data.get("metrics", [])),
            },
        )

        # Simulate some processing updates (optional, for better UX)
        total_operations = len(request_data.get("text_pairs", [])) * len(
            request_data.get("metrics", [])
        )

        # Process the metrics
        results = compute_metrics_for_request(request_data)

        # Update progress to 50% after computation
        self.update_state(
            state="PROCESSING",
            meta={
                "message": "Finalizing results...",
                "progress": 50,
                "total_pairs": len(request_data.get("text_pairs", [])),
                "total_metrics": len(request_data.get("metrics", [])),
            },
        )

        processing_time = time.time() - start_time

        # Final result
        final_result = {
            "success": True,
            "message": (
                f"Successfully calculated {len(request_data.get('metrics', []))} metrics "
                f"for {len(request_data.get('text_pairs', []))} text pairs"
            ),
            "results": results,
            "processing_time_seconds": round(processing_time, 3),
            "total_operations": total_operations,
        }

        # Send webhook notification if webhook_url is provided
        webhook_url = request_data.get("webhook_url")
        if webhook_url:
            try:
                # Run the webhook notification asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                webhook_success = loop.run_until_complete(
                    send_webhook_notification(
                        str(webhook_url), self.request.id, final_result
                    )
                )
                loop.close()

                # Add webhook status to the result
                final_result["webhook_sent"] = webhook_success

            except Exception as webhook_error:
                logger.error(
                    "Failed to send webhook for job %s: %s",
                    self.request.id,
                    str(webhook_error),
                )
                final_result["webhook_sent"] = False
                final_result["webhook_error"] = str(webhook_error)

        return final_result

    except Exception as exc:
        # Log the error (you might want to use proper logging here)
        error_message = f"Task failed: {str(exc)}"

        # Update task state to failed
        self.update_state(
            state="FAILURE",
            meta={
                "error": error_message,
                "exc_type": type(exc).__name__,
                "exc_message": str(exc),
            },
        )

        # Send webhook notification for failed job if webhook_url is provided
        webhook_url = request_data.get("webhook_url")
        if webhook_url:
            try:
                error_result = {
                    "success": False,
                    "error": error_message,
                    "exc_type": type(exc).__name__,
                    "exc_message": str(exc),
                }

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    send_webhook_notification(
                        str(webhook_url), self.request.id, error_result
                    )
                )
                loop.close()

            except Exception as webhook_error:
                logger.error(
                    "Failed to send error webhook for job %s: %s",
                    self.request.id,
                    str(webhook_error),
                )

        # Re-raise the exception so Celery can handle it properly
        raise exc


@celery_app.task(bind=True, name="compute_metrics_async")
def compute_metrics_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task for computing metrics asynchronously.

    Args:
        request_data: Serialized MetricsRequest data

    Returns:
        Dictionary containing the results and metadata
    """
    return _compute_metrics_task_logic(self, request_data)


@celery_app.task(name="health_check")
def health_check_task() -> Dict[str, str]:
    """Simple health check task to verify Celery is working."""
    return {"status": "healthy", "message": "Celery worker is running"}
