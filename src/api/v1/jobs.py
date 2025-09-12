"""
Job management endpoints for the LLM-scorecaster async API.

This module provides endpoints for checking job status and retrieving
results from asynchronous metric computation tasks.
"""

from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from celery.result import AsyncResult

from src.models.schemas import (
    JobStatusResponse,
    MetricsResponse,
    TextPairResult,
)
from src.api.auth.dependencies import verify_api_key
from src.celery.celery_app import celery_app

router = APIRouter()


def get_celery_task_info(task_id: str) -> AsyncResult:
    """Get Celery task information by task ID."""
    return AsyncResult(task_id, app=celery_app)


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str, _authenticated: bool = Depends(verify_api_key)
) -> JobStatusResponse:
    """
    Get the current status of an async job.

    Args:
        job_id: The unique identifier for the job

    Returns:
        JobStatusResponse containing current status and progress information
    """
    try:
        task = get_celery_task_info(job_id)
        return _build_status_response(job_id, task)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {str(e)}",
        ) from e


def _build_status_response(job_id: str, task) -> JobStatusResponse:
    """Build appropriate JobStatusResponse based on task state."""

    # Status mapping with response builders
    status_handlers = {
        "PENDING": lambda: JobStatusResponse(
            job_id=job_id,
            status="PENDING",
            message="Job is queued and waiting to be processed",
            progress=0,
        ),
        "PROCESSING": lambda: _build_processing_response(job_id, task),
        "SUCCESS": lambda: JobStatusResponse(
            job_id=job_id,
            status="COMPLETED",
            message="Job completed successfully. Results are ready.",
            progress=100,
            completed=True,
        ),
        "FAILURE": lambda: _build_failure_response(job_id, task),
        "RETRY": lambda: JobStatusResponse(
            job_id=job_id,
            status="RETRYING",
            message="Job failed and is being retried",
            progress=0,
        ),
        "REVOKED": lambda: JobStatusResponse(
            job_id=job_id,
            status="CANCELLED",
            message="Job was cancelled",
            failed=True,
        ),
    }

    # Get handler or use default
    handler = status_handlers.get(
        task.state, lambda: _build_unknown_response(job_id, task)
    )
    return handler()


def _build_processing_response(job_id: str, task) -> JobStatusResponse:
    """Build response for PROCESSING state."""
    task_info = task.info or {}
    return JobStatusResponse(
        job_id=job_id,
        status="PROCESSING",
        message=task_info.get("message", "Job is being processed"),
        progress=task_info.get("progress", 0),
        total_pairs=task_info.get("total_pairs"),
        total_metrics=task_info.get("total_metrics"),
    )


def _build_failure_response(job_id: str, task) -> JobStatusResponse:
    """Build response for FAILURE state."""
    task_info = task.info or {}
    error_message = task_info.get("error", "Job failed with unknown error")
    return JobStatusResponse(
        job_id=job_id,
        status="FAILED",
        message=error_message,
        error=error_message,
        failed=True,
    )


def _build_unknown_response(job_id: str, task) -> JobStatusResponse:
    """Build response for unknown task states."""
    return JobStatusResponse(
        job_id=job_id,
        status=task.state,
        message=f"Job is in state: {task.state}",
    )


@router.get("/results/{job_id}", response_model=MetricsResponse)
async def get_job_results(
    job_id: str, _authenticated: bool = Depends(verify_api_key)
) -> MetricsResponse:
    """
    Retrieve the results of a completed async job.

    Args:
        job_id: The unique identifier for the job

    Returns:
        MetricsResponse containing the computed metrics results

    Raises:
        HTTPException: If job is not completed, failed, or doesn't exist
    """
    try:
        task = get_celery_task_info(job_id)

        if task.state == "PENDING":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is still pending. Check status first.",
            )

        if task.state == "PROCESSING":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is still processing. Check status first.",
            )

        if task.state == "SUCCESS":
            # Get the task result
            result = task.result

            if not isinstance(result, dict):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Invalid result format from completed job",
                )

            # Convert dict results back to Pydantic models
            results = []
            for result_data in result.get("results", []):
                if isinstance(result_data, dict):
                    results.append(TextPairResult(**result_data))
                else:
                    results.append(result_data)

            return MetricsResponse(
                success=result.get("success", True),
                message=result.get("message", "Job completed successfully"),
                results=results,
                processing_time_seconds=result.get("processing_time_seconds", 0),
            )

        if task.state == "FAILURE":
            task_info = task.info or {}
            error_message = task_info.get("error", "Job failed with unknown error")

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job failed: {error_message}",
            )

        if task.state == "REVOKED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job was cancelled and results are not available",
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is in unexpected state: {task.state}",
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job results: {str(e)}",
        ) from e


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    _authenticated: bool = Depends(verify_api_key),
    terminate: bool = Query(
        False, description="Force terminate the task if it's running"
    ),
) -> Dict[str, Any]:
    """
    Cancel an async job.

    Args:
        job_id: The unique identifier for the job
        terminate: If True, forcefully terminate running tasks (use with caution)

    Returns:
        Dictionary with cancellation status
    """
    try:
        task = get_celery_task_info(job_id)

        if task.state == "SUCCESS":
            return {
                "job_id": job_id,
                "message": "Job already completed, cannot cancel",
                "status": "already_completed",
            }

        if task.state in ["FAILURE", "REVOKED"]:
            return {
                "job_id": job_id,
                "message": f"Job already in terminal state: {task.state}",
                "status": "already_terminal",
            }

        # Cancel the task
        task.revoke(terminate=terminate)

        action = "terminated" if terminate else "cancelled"
        return {
            "job_id": job_id,
            "message": f"Job {action} successfully",
            "status": "cancelled",
            "terminated": terminate,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}",
        ) from e


@router.get("/")
async def list_active_jobs(
    _authenticated: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    List active jobs (optional endpoint for monitoring).

    Note: This requires additional Celery configuration to work properly
    and may not be available in all setups.

    Returns:
        Dictionary containing active job information
    """
    try:
        # Get active tasks from Celery
        # Note: This requires flower or additional monitoring setup in production
        inspect = celery_app.control.inspect()

        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()

        if not active_tasks and not scheduled_tasks:
            return {
                "message": "No active jobs found",
                "active_jobs": [],
                "scheduled_jobs": [],
                "total_count": 0,
            }

        # Process active tasks
        active_jobs = []
        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    active_jobs.append(
                        {
                            "job_id": task.get("id"),
                            "worker": worker,
                            "name": task.get("name"),
                            "status": "PROCESSING",
                        }
                    )

        # Process scheduled tasks
        scheduled_jobs = []
        if scheduled_tasks:
            for worker, tasks in scheduled_tasks.items():
                for task in tasks:
                    scheduled_jobs.append(
                        {
                            "job_id": (
                                task.get("id") if isinstance(task, dict) else None
                            ),
                            "worker": worker,
                            "status": "SCHEDULED",
                        }
                    )

        return {
            "message": f"Found {len(active_jobs)} active and {len(scheduled_jobs)} scheduled jobs",
            "active_jobs": active_jobs,
            "scheduled_jobs": scheduled_jobs,
            "total_count": len(active_jobs) + len(scheduled_jobs),
        }

    except Exception as e:
        # This endpoint is optional and may fail in some configurations
        return {
            "message": f"Unable to retrieve job list: {str(e)}",
            "active_jobs": [],
            "scheduled_jobs": [],
            "total_count": 0,
            "error": str(e),
        }
