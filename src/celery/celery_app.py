"""
Core Celery application setup.

This module contains only the Celery app initialization and basic configuration.
All tasks, signals, and utilities are organized in separate modules.
"""

import time
import logging
import sys
from celery import Celery, signals

from src.core.settings import settings
from .config import get_celery_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "llm_scorecaster",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "src.celery.tasks.metrics",
        "src.celery.tasks.health",
    ],
)

# Apply configuration
celery_app.conf.update(get_celery_config())


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


logger.info("Celery application configured successfully")
