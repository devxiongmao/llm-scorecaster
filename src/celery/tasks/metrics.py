"""
Metrics computation tasks.

This module contains Celery tasks related to computing metrics for text pairs.
"""

import time
import logging
from typing import Dict, Any, List, cast
from celery import Task

from src.core.computation import compute_metrics_core
from src.models.schemas import MetricsRequest
from ..celery_app import celery_app
from ..utils.webhooks import send_webhook_notification

logger = logging.getLogger(__name__)


def compute_metrics_for_request(request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compute metrics for a request.

    This function reconstructs the MetricsRequest from serialized data
    and computes the metrics using the core computation logic.

    Args:
        request_data: Serialized MetricsRequest data

    Returns:
        List of computed metric results
    """
    # Reconstruct the request from serialized data
    request = MetricsRequest(**request_data)
    results = compute_metrics_core(request)
    # Serialize results for Celery
    return [result.model_dump() for result in results]


def _compute_metrics_task_logic(
    task_instance, request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Core logic for computing metrics asynchronously.

    Args:
        task_instance: The Celery task instance (for state updates)
        request_data: Serialized MetricsRequest data

    Returns:
        Dictionary containing the results and metadata

    Raises:
        Exception: Re-raises any exceptions that occur during processing
    """
    try:
        start_time = time.time()

        # Update task state to indicate processing has started
        task_instance.update_state(
            state="PROCESSING",
            meta={
                "message": "Computing metrics...",
                "progress": 0,
                "total_pairs": len(request_data.get("text_pairs", [])),
                "total_metrics": len(request_data.get("metrics", [])),
            },
        )

        # Calculate total operations for progress tracking
        total_operations = len(request_data.get("text_pairs", [])) * len(
            request_data.get("metrics", [])
        )

        # Process the metrics
        results = compute_metrics_for_request(request_data)

        # Update progress to 50% after computation
        task_instance.update_state(
            state="PROCESSING",
            meta={
                "message": "Finalizing results...",
                "progress": 50,
                "total_pairs": len(request_data.get("text_pairs", [])),
                "total_metrics": len(request_data.get("metrics", [])),
            },
        )

        processing_time = time.time() - start_time

        # Build final result
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
                webhook_success = send_webhook_notification(
                    str(webhook_url),
                    task_instance.request.id,
                    final_result,
                )

                # Add webhook status to the result
                final_result["webhook_sent"] = webhook_success

            except Exception as webhook_error:
                logger.error(
                    "Failed to send webhook for job %s: %s",
                    task_instance.request.id,
                    str(webhook_error),
                )
                final_result["webhook_sent"] = False
                final_result["webhook_error"] = str(webhook_error)

        return final_result

    except Exception as exc:
        error_message = f"Task failed: {str(exc)}"

        # Update task state to failed
        task_instance.update_state(
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

                send_webhook_notification(
                    str(webhook_url), task_instance.request.id, error_result
                )

            except Exception as webhook_error:
                logger.error(
                    "Failed to send error webhook for job %s: %s",
                    task_instance.request.id,
                    str(webhook_error),
                )

        # Re-raise the exception so Celery can handle it properly
        raise exc


@celery_app.task(bind=True, name="compute_metrics_async")
def _compute_metrics_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task for computing metrics asynchronously.

    This task processes a metrics computation request, updates progress,
    and optionally sends webhook notifications upon completion.

    Args:
        request_data: Serialized MetricsRequest data

    Returns:
        Dictionary containing the results and metadata
    """
    return _compute_metrics_task_logic(self, request_data)


compute_metrics_task = cast(Task, _compute_metrics_task)
