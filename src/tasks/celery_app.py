"""
Celery application configuration for async task processing.

This module configures Celery with Redis as both broker and result backend
for handling asynchronous metric computation tasks.
"""

import time
from typing import List, Dict, Any

from celery import Celery

from src.core.metrics.registry import metric_registry
from src.models.schemas import TextPair, MetricsRequest, TextPairResult
from src.core.settings import settings

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
        results.append(result_data.dict())

    return results


@celery_app.task(bind=True, name="compute_metrics_async")
def compute_metrics_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task for computing metrics asynchronously.

    Args:
        request_data: Serialized MetricsRequest data

    Returns:
        Dictionary containing the results and metadata
    """
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
        return {
            "success": True,
            "message": f"Successfully calculated {len(request_data.get('metrics', []))} metrics for {len(request_data.get('text_pairs', []))} text pairs",
            "results": results,
            "processing_time_seconds": round(processing_time, 3),
            "total_operations": total_operations,
        }

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

        # Re-raise the exception so Celery can handle it properly
        raise exc


@celery_app.task(name="health_check")
def health_check_task() -> Dict[str, str]:
    """Simple health check task to verify Celery is working."""
    return {"status": "healthy", "message": "Celery worker is running"}
