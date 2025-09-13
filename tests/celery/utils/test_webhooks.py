"""Tests for the webhook notification utility functions."""

import pytest
from pytest_httpserver import HTTPServer
from urllib3.connectionpool import ConnectionPool
from urllib3.exceptions import ReadTimeoutError
from werkzeug import Response

from src.celery.utils.webhooks import send_webhook_notification, _calculate_backoff

from src.core.settings import settings

WEBHOOK_PATH = "/webhook"


@pytest.fixture(name="webhook_url")
def webhook_url_fixture(httpserver: HTTPServer):
    """A simple webhook URL fixture"""
    return httpserver.url_for(WEBHOOK_PATH)


@pytest.fixture(name="job_id")
def job_id_fixture():
    """A simple job ID fixture"""
    return "test-job-123"


@pytest.fixture(name="success_result_data")
def success_result_data_fixture():
    """A simple successful result data fixture"""
    return {"success": True, "message": "Task completed", "output": {"foo": "bar"}}


@pytest.fixture(name="failed_result_data")
def failed_result_data_fixture():
    """A simple failed result data fixture"""
    return {"success": False, "error": "Task failed", "details": {"error_code": 500}}


def test_send_webhook_notification_posts_successfully_with_success_result(
    httpserver: HTTPServer,
    webhook_url: str,
    job_id: str,
    success_result_data: dict,
):
    """Test sending a webhook notification with successful result data"""
    httpserver.expect_request(WEBHOOK_PATH).respond_with_response(
        response=Response(status=200)
    )

    result = send_webhook_notification(webhook_url, job_id, success_result_data)

    assert result is True


def test_send_webhook_notification_posts_successfully_with_failed_result(
    httpserver: HTTPServer,
    webhook_url: str,
    job_id: str,
    failed_result_data: dict,
):
    """Test sending a webhook notification with failed result data"""
    httpserver.expect_request(WEBHOOK_PATH).respond_with_response(
        response=Response(status=204)
    )

    result = send_webhook_notification(webhook_url, job_id, failed_result_data)

    assert result is True


@pytest.mark.parametrize("success_status_code", [200, 201, 202, 204])
def test_send_webhook_notification_succeeds_on_valid_status_codes(
    httpserver: HTTPServer,
    webhook_url: str,
    job_id: str,
    success_result_data: dict,
    success_status_code: int,
):
    """Test that various 2xx status codes are treated as success"""
    httpserver.expect_request(WEBHOOK_PATH).respond_with_response(
        response=Response(status=success_status_code)
    )

    result = send_webhook_notification(webhook_url, job_id, success_result_data)

    assert result is True


def test_send_webhook_notification_sends_correct_payload_structure(
    httpserver: HTTPServer,
    webhook_url: str,
    job_id: str,
    success_result_data: dict,
):
    """Test that the webhook payload has the correct structure and content"""
    request_data = None

    def capture_request(request):
        nonlocal request_data
        request_data = request.get_json()
        return Response(status=200)

    httpserver.expect_request(WEBHOOK_PATH).respond_with_handler(capture_request)

    send_webhook_notification(webhook_url, job_id, success_result_data)

    assert request_data is not None
    assert request_data.get("job_id") == job_id
    assert request_data.get("status") == "COMPLETED"
    assert request_data.get("data") == success_result_data
    assert request_data.get("timestamp") is not None
    assert isinstance(request_data.get("timestamp"), (int, float))


def test_send_webhook_notification_sends_failed_status_for_unsuccessful_result(
    httpserver: HTTPServer,
    webhook_url: str,
    job_id: str,
    failed_result_data: dict,
):
    """Test that a failed result data sends status as FAILED"""
    request_data = None

    def capture_request(request):
        nonlocal request_data
        request_data = request.get_json()
        return Response(status=200)

    httpserver.expect_request(WEBHOOK_PATH).respond_with_handler(capture_request)

    send_webhook_notification(webhook_url, job_id, failed_result_data)

    assert request_data
    assert request_data.get("status") == "FAILED"


@pytest.mark.parametrize("error_status_code", [400, 404, 500, 502, 504])
def test_send_webhook_notification_returns_false_on_http_errors(
    httpserver: HTTPServer,
    webhook_url: str,
    job_id: str,
    success_result_data: dict,
    error_status_code: int,
):
    """Test that various error status codes result in a False return value"""
    httpserver.expect_request(WEBHOOK_PATH).respond_with_response(
        response=Response(status=error_status_code)
    )

    result = send_webhook_notification(webhook_url, job_id, success_result_data)

    assert result is False


class FakeResponse:
    """A simple fake response object for mocking requests responses"""

    status_code = 200

    def raise_for_status(self):
        """Mock raise_for_status method"""

    def json(self):
        """Mock json method"""


@pytest.fixture(name="fake_post_with_timeout")
def fake_post_with_timeout_fixture(webhook_url):
    """A fake post function that simulates a ReadTimeoutError on the first call
    and succeeds on subsequent calls."""
    call_count = []

    def fake_post(*_args, **_kwargs):
        """Simulate a ReadTimeoutError on the first call"""
        call_count.append(1)

        if len(call_count) == 1:
            raise ReadTimeoutError(
                url=webhook_url,
                message="read timeout",
                pool=ConnectionPool(host="localhost", port=80),
            )

        return FakeResponse()

    return fake_post


def test_send_webhook_notification_can_succeed_on_retry_after_timeout(
    monkeypatch,
    fake_post_with_timeout,
    webhook_url: str,
    job_id: str,
    success_result_data: dict,
):
    """Test that a ReadTimeoutError can be retried and eventually succeed"""
    monkeypatch.setattr(
        "src.celery.utils.webhooks.requests.post", fake_post_with_timeout
    )
    monkeypatch.setattr("src.celery.utils.webhooks.sleep", lambda seconds: None)

    result = send_webhook_notification(webhook_url, job_id, success_result_data, 0)

    assert result is True


@pytest.fixture(name="fake_post_always_timeout")
def fake_post_always_timeout_fixture(webhook_url):
    """A fake post function that always raises a ReadTimeoutError."""

    def fake_post(*_args, **_kwargs):
        """Always raise a ReadTimeoutError"""
        raise ReadTimeoutError(
            url=webhook_url,
            message="read timeout",
            pool=ConnectionPool(host="localhost", port=80),
        )

    return fake_post


def test_send_webhook_notification_exhausts_retries_on_readtimeout(
    monkeypatch,
    fake_post_always_timeout,
    webhook_url: str,
    job_id: str,
    success_result_data: dict,
):
    """Test that retries are exhausted and ReadTimeoutError is raised"""
    monkeypatch.setattr(
        "src.celery.utils.webhooks.requests.post", fake_post_always_timeout
    )
    monkeypatch.setattr("src.celery.utils.webhooks.sleep", lambda seconds: None)

    with pytest.raises(ReadTimeoutError) as exc:
        send_webhook_notification(
            webhook_url, job_id, success_result_data, settings.max_retries
        )

    assert isinstance(exc.value, ReadTimeoutError)


@pytest.fixture(name="fake_post_with_generic_error")
def fake_post_with_generic_error_fixture():
    """A fake post function that simulates a ConnectionError on the first two calls
    and succeeds on the third call."""
    call_count = []

    def fake_post(*_args, **_kwargs):
        """Simulate a ConnectionError on the first two calls"""
        call_count.append(1)

        if len(call_count) <= 2:  # Fail twice, then succeed
            raise ConnectionError("Connection failed")

        return FakeResponse()

    return fake_post


def test_send_webhook_notification_can_succeed_on_retry_after_generic_error(
    monkeypatch,
    fake_post_with_generic_error,
    webhook_url: str,
    job_id: str,
    success_result_data: dict,
):
    """Test that a generic ConnectionError can be retried and eventually succeed"""
    monkeypatch.setattr(
        "src.celery.utils.webhooks.requests.post", fake_post_with_generic_error
    )
    monkeypatch.setattr("src.celery.utils.webhooks.sleep", lambda seconds: None)

    result = send_webhook_notification(webhook_url, job_id, success_result_data, 0)

    assert result is True


@pytest.fixture(name="fake_post_always_generic_error")
def fake_post_always_generic_error_fixture():
    """A fake post function that always raises a ConnectionError."""

    def fake_post(*_args, **_kwargs):
        """Always raise a ConnectionError"""
        raise ConnectionError("Connection failed")

    return fake_post


def test_send_webhook_notification_returns_false_after_exhausting_retries_on_generic_error(
    monkeypatch,
    fake_post_always_generic_error,
    webhook_url: str,
    job_id: str,
    success_result_data: dict,
):
    """Test that retries are exhausted and ConnectionError is raised"""
    monkeypatch.setattr(
        "src.celery.utils.webhooks.requests.post", fake_post_always_generic_error
    )
    monkeypatch.setattr("src.celery.utils.webhooks.sleep", lambda seconds: None)

    result = send_webhook_notification(webhook_url, job_id, success_result_data, 0)

    assert result is False


def test_calculate_backoff_function():
    """Test the exponential backoff calculation"""
    assert _calculate_backoff(0) == 2.0  # 2 * 2^0
    assert _calculate_backoff(1) == 4.0  # 2 * 2^1
    assert _calculate_backoff(2) == 8.0  # 2 * 2^2
    assert _calculate_backoff(3) == 16.0  # 2 * 2^3
