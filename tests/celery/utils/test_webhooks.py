"""
Tests for webhook utilities.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from src.celery.utils.webhooks import send_webhook_notification


@pytest.mark.asyncio
async def test_send_webhook_notification_success():
    """Test successful webhook notification."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_context = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_context
        mock_context.post.return_value = mock_response

        result = await send_webhook_notification(
            webhook_url="https://example.com/webhook",
            job_id="test-job-123",
            result_data={"success": True, "message": "Task completed"},
        )

    assert result is True
    mock_context.post.assert_called_once()

    # Verify the payload structure
    call_args = mock_context.post.call_args
    assert call_args[1]["json"]["job_id"] == "test-job-123"
    assert call_args[1]["json"]["status"] == "COMPLETED"
    assert call_args[1]["json"]["data"] == {
        "success": True,
        "message": "Task completed",
    }
    assert "timestamp" in call_args[1]["json"]


@pytest.mark.asyncio
async def test_send_webhook_notification_failed_status():
    """Test webhook notification with failed task status."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_context = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_context
        mock_context.post.return_value = mock_response

        result = await send_webhook_notification(
            webhook_url="https://example.com/webhook",
            job_id="test-job-456",
            result_data={"success": False, "error": "Task failed"},
        )

    assert result is True

    # Verify the payload shows FAILED status
    call_args = mock_context.post.call_args
    assert call_args[1]["json"]["status"] == "FAILED"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 201, 202, 204])
async def test_send_webhook_notification_success_status_codes(status_code):
    """Test webhook notification succeeds for various HTTP success codes."""
    mock_response = MagicMock()
    mock_response.status_code = status_code

    with patch("httpx.AsyncClient") as mock_client:
        mock_context = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_context
        mock_context.post.return_value = mock_response

        result = await send_webhook_notification(
            webhook_url="https://example.com/webhook",
            job_id="test-job-789",
            result_data={"success": True},
        )

    assert result is True


@pytest.mark.asyncio
async def test_send_webhook_notification_http_error_retries():
    """Test webhook notification retries on HTTP errors."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("httpx.AsyncClient") as mock_client, patch(
        "src.core.settings.settings"
    ) as mock_settings, patch("asyncio.sleep", new_callable=AsyncMock):

        mock_settings.max_retries = 2
        mock_settings.max_timeout = 30

        mock_context = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_context
        mock_context.post.return_value = mock_response

        result = await send_webhook_notification(
            webhook_url="https://example.com/webhook",
            job_id="test-job-error",
            result_data={"success": True},
        )

    assert result is False
    # Should be called max_retries + 1 times (initial + retries)
    assert mock_context.post.call_count == 3


@pytest.mark.asyncio
async def test_send_webhook_notification_timeout_exception():
    """Test webhook notification handles timeout exceptions."""
    with patch("httpx.AsyncClient") as mock_client, patch(
        "src.core.settings.settings"
    ) as mock_settings, patch("asyncio.sleep", new_callable=AsyncMock):

        mock_settings.max_retries = 1
        mock_settings.max_timeout = 30

        mock_context = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_context
        mock_context.post.side_effect = httpx.TimeoutException("Timeout")

        result = await send_webhook_notification(
            webhook_url="https://example.com/webhook",
            job_id="test-job-timeout",
            result_data={"success": True},
        )

    assert result is False
    assert mock_context.post.call_count == 2


@pytest.mark.asyncio
async def test_send_webhook_notification_general_exception():
    """Test webhook notification handles general exceptions."""
    with patch("httpx.AsyncClient") as mock_client, patch(
        "src.core.settings.settings"
    ) as mock_settings, patch("asyncio.sleep", new_callable=AsyncMock):

        mock_settings.max_retries = 1
        mock_settings.max_timeout = 30

        mock_context = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_context
        mock_context.post.side_effect = Exception("Connection error")

        result = await send_webhook_notification(
            webhook_url="https://example.com/webhook",
            job_id="test-job-exception",
            result_data={"success": True},
        )

    assert result is False
    assert mock_context.post.call_count == 2


@pytest.mark.asyncio
async def test_send_webhook_notification_exponential_backoff():
    """Test webhook notification uses exponential backoff between retries."""
    with patch("httpx.AsyncClient") as mock_client, patch(
        "src.core.settings.settings"
    ) as mock_settings, patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        mock_settings.max_retries = 2
        mock_settings.max_timeout = 30

        mock_context = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_context
        mock_context.post.side_effect = Exception("Error")

        await send_webhook_notification(
            webhook_url="https://example.com/webhook",
            job_id="test-job-backoff",
            result_data={"success": True},
        )

    # Should sleep 2^0=1 second, then 2^1=2 seconds, then 2^2=4 seconds
    expected_calls = [pytest.approx(1), pytest.approx(2), pytest.approx(4)]
    actual_calls = [call[0][0] for call in mock_sleep.call_args_list]
    assert actual_calls == expected_calls


@pytest.mark.asyncio
async def test_send_webhook_notification_success_after_retry():
    """Test webhook notification succeeds after initial failure."""
    mock_response_fail = MagicMock()
    mock_response_fail.status_code = 500
    mock_response_fail.text = "Error"

    mock_response_success = MagicMock()
    mock_response_success.status_code = 200

    with patch("httpx.AsyncClient") as mock_client, patch(
        "src.core.settings.settings"
    ) as mock_settings, patch("asyncio.sleep", new_callable=AsyncMock):

        mock_settings.max_retries = 2
        mock_settings.max_timeout = 30

        mock_context = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_context
        mock_context.post.side_effect = [mock_response_fail, mock_response_success]

        result = await send_webhook_notification(
            webhook_url="https://example.com/webhook",
            job_id="test-job-retry-success",
            result_data={"success": True},
        )

    assert result is True
    assert mock_context.post.call_count == 2
