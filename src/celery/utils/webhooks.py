"""
Webhook utilities for task notifications.

This module provides functionality for sending webhook notifications
when tasks complete or fail.
"""

import time
import logging
from time import sleep
from typing import Dict, Any

import requests
from urllib3.exceptions import ReadTimeoutError

from src.core.settings import settings

logger = logging.getLogger(__name__)


def send_webhook_notification(
    webhook_url: str,
    job_id: str,
    result_data: Dict[str, Any],
    retry_count: int = 0,
) -> bool:
    """
    Send webhook notification with results.

    Args:
        webhook_url: The URL to send the webhook to
        job_id: The job ID for reference
        result_data: The computed results to send
        retry_count: Current retry attempt count

    Returns:
        bool: True if successful, False otherwise
    """
    webhook_payload = {
        "job_id": job_id,
        "status": "COMPLETED" if result_data.get("success") else "FAILED",
        "timestamp": time.time(),
        "data": result_data,
    }

    try:
        response = requests.post(
            webhook_url,
            headers={"Content-Type": "application/json"},
            json=webhook_payload,
            timeout=settings.max_timeout,
        )

        response.raise_for_status()

        logger.info(
            "Webhook sent successfully for job %s to %s",
            job_id,
            webhook_url,
        )
        return True

    except ReadTimeoutError as e:
        # Occasionally, the webhook server is slow to respond or unresponsive.
        # It normally succeeds on a retry, so better to retry here then
        # let the entire task fail resulting in it restarting.
        if retry_count < settings.max_retries:
            sleep(_calculate_backoff(retry_count))
            return send_webhook_notification(
                webhook_url, job_id, result_data, retry_count + 1
            )

        logger.error("All webhook attempts failed for job %s due to timeout", job_id)
        raise e

    except Exception as e:
        logger.warning(
            "Webhook attempt %d failed for job %s: %s", retry_count + 1, job_id, str(e)
        )

        if retry_count < settings.max_retries:
            sleep(_calculate_backoff(retry_count))
            return send_webhook_notification(
                webhook_url, job_id, result_data, retry_count + 1
            )

        logger.error("All webhook attempts failed for job %s", job_id)
        return False


def _calculate_backoff(retry_count: int) -> float:
    """Calculate exponential backoff delay."""
    return 2 * (2**retry_count)
