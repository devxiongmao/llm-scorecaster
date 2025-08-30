"""
Version 1 of the asynchronous API for the LLM-scorecaster.

This module provides the routes used for the asynchronous operation of the API.
It receives requests, queues them for processing with Celery, and returns a job ID
that can be used to check status and retrieve results.
"""

import uuid
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from src.models.schemas import (
    MetricsRequest,
    AsyncJobResponse,
)
from src.api.auth.dependencies import verify_api_key
from src.tasks.celery_app import compute_metrics_task, health_check_task

router = APIRouter()


@router.post("/evaluate", response_model=AsyncJobResponse)
async def evaluate_metrics_async(
    request: MetricsRequest, _authenticated: bool = Depends(verify_api_key)
) -> AsyncJobResponse:
    """
    Asynchronously evaluate metrics for the provided text pairs.

    This endpoint queues a metrics computation job and returns immediately
    with a job ID. Use the job ID with the /jobs endpoints to check status
    and retrieve results when complete.

    Args:
        request: The metrics request containing text pairs and requested metrics

    Returns:
        AsyncJobResponse containing the job ID and status information
    """
    try:
        # Generate a unique job ID for tracking
        job_id = str(uuid.uuid4())

        # Convert the Pydantic model to dict for Celery serialization
        request_data = request.model_dump()

        # Submit the task to Celery with custom task ID
        task = compute_metrics_task.apply_async(args=[request_data], task_id=job_id) # type: ignore

        # Verify the task was submitted successfully
        if not task.id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to submit task to queue",
            )

        return AsyncJobResponse(
            job_id=job_id,
            status="PENDING",
            message="Job queued successfully. Use the job ID to check status and retrieve results.",
            estimated_completion_time=len(request.text_pairs)
            * len(request.metrics)
            * 0.5,  # Rough estimate in seconds
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue async job: {str(e)}",
        ) from e


@router.get("/health")
async def health_check_async(
    _authenticated: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Health check endpoint for the async API.

    This endpoint checks if the Celery workers are responsive and ready
    to process tasks.
    """
    try:
        # Use a short timeout to quickly determine if workers are available
        task = health_check_task.apply_async() # type: ignore

        try:
            # Wait up to 5 seconds for the health check
            result = task.get(timeout=5.0)

            return {
                "status": "healthy",
                "message": "Async API and workers are operational",
                "worker_status": result.get("status", "unknown"),
                "celery_available": True,
            }

        except Exception as worker_exc:
            return {
                "status": "degraded",
                "message": f"Async API is running but workers may be unavailable: {str(worker_exc)}",
                "worker_status": "unavailable",
                "celery_available": False,
            }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        ) from e
