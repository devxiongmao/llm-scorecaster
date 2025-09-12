"""
Webhook utilities for task notifications.

This module provides functionality for sending webhook notifications
when tasks complete or fail.
"""

import time
import asyncio
import logging
from typing import Dict, Any

import httpx

from src.core.settings import settings

logger = logging.getLogger(__name__)


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
