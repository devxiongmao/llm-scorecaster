"""
Health check tasks.

This module contains Celery tasks for monitoring and health checking.
"""

from typing import Dict, cast
from celery import Task

from ..celery_app import celery_app


@celery_app.task(name="health_check")
def _health_check_task() -> Dict[str, str]:
    """
    Simple health check task to verify Celery is working.

    This task can be used by monitoring systems to verify that
    Celery workers are running and able to process tasks.

    Returns:
        Dictionary with health status information
    """
    return {"status": "healthy", "message": "Celery worker is running"}


health_check_task = cast(Task, _health_check_task)
