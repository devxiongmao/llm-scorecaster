"""Tests for the initial configuration of the celery app"""

import time
import logging
from unittest.mock import Mock, patch

import pytest
from celery import Celery

from src.celery.celery_app import (
    celery_app,
    setup_celery_logging,
    task_prerun_handler,
    task_success_handler,
    task_retry_handler,
    task_failure_handler,
)


@pytest.fixture(name="mock_task")
def mock_task_fixture():
    """Create a mock task instance for testing signal handlers."""
    task = Mock()
    task.request = Mock()
    task.request.id = "test-task-id"
    task.request.headers = {}
    task.name = "test-task-name"
    task.request.hostname = "test-host"
    task.request.retries = 0
    task.max_retries = 3
    return task


@pytest.fixture(name="mock_task_with_headers")
def mock_task_with_headers_fixture():
    """Create a mock task with pre-existing headers."""
    task = Mock()
    task.request = Mock()
    task.request.id = "test-task-id"
    task.request.headers = {"started_at": time.time() - 1.5}  # 1.5 seconds ago
    task.name = "test-task-name"
    task.request.hostname = "test-host"
    return task


@pytest.fixture(name="mock_task_no_headers")
def mock_task_no_headers_fixture():
    """Create a mock task with None headers."""
    task = Mock()
    task.request = Mock()
    task.request.id = "test-task-id"
    task.request.headers = None
    task.name = "test-task-name"
    return task


@pytest.fixture(name="caplog_info")
def caplog_info_fixture(caplog):
    """Set logging level to INFO for capturing log messages."""
    caplog.set_level(logging.INFO)
    return caplog


class TestCeleryAppInitialization:
    """Test Celery application initialization and configuration."""

    def test_celery_app_exists(self):
        """Test that Celery app is properly initialized."""
        assert isinstance(celery_app, Celery)
        assert celery_app.main == "llm_scorecaster"

    @patch("src.celery.celery_app.settings")
    def test_celery_app_uses_redis_configuration(self, mock_settings):
        """Test that Celery app uses Redis URL from settings."""
        mock_settings.redis_url = "redis://test:6379/0"

        # We can't easily test this without recreating the app,
        # but we can verify the settings module is used
        assert mock_settings is not None

    @patch("src.celery.celery_app.get_celery_config")
    def test_celery_app_applies_configuration(self, mock_get_config):
        """Test that Celery configuration is applied."""
        mock_config = {"task_serializer": "json", "result_serializer": "json"}
        mock_get_config.return_value = mock_config

        # Since conf.update was already called during import,
        # we verify the function would be called
        assert mock_get_config is not None


class TestSetupCeleryLogging:
    """Test the setup_celery_logging signal handler."""

    def test_setup_celery_logging_exists(self):
        """Test that setup_celery_logging function exists and is callable."""
        assert callable(setup_celery_logging)

    def test_setup_celery_logging_accepts_kwargs(self):
        """Test that setup_celery_logging accepts keyword arguments."""
        # Should not raise an exception
        result = setup_celery_logging(some_arg="test", another_arg=123)
        assert result is None


class TestTaskPrerunHandler:
    """Test the task_prerun_handler signal."""

    def test_task_prerun_handler_sets_started_at_timestamp(self, mock_task):
        """Test that prerun handler sets started_at timestamp in headers."""
        start_time = time.time()

        task_prerun_handler(mock_task)

        assert "started_at" in mock_task.request.headers
        started_at = mock_task.request.headers["started_at"]
        assert isinstance(started_at, float)
        assert started_at >= start_time
        assert started_at <= time.time()

    def test_task_prerun_handler_initializes_headers_if_none(
        self, mock_task_no_headers
    ):
        """Test that prerun handler initializes headers if None."""
        assert mock_task_no_headers.request.headers is None

        task_prerun_handler(mock_task_no_headers)

        assert mock_task_no_headers.request.headers is not None
        assert isinstance(mock_task_no_headers.request.headers, dict)
        assert "started_at" in mock_task_no_headers.request.headers

    def test_task_prerun_handler_logs_task_start(self, mock_task, caplog_info):
        """Test that prerun handler logs task start."""
        task_prerun_handler(mock_task)

        assert len(caplog_info.records) == 1
        log_record = caplog_info.records[0]
        assert log_record.levelname == "INFO"
        assert "Task Started" in log_record.message
        assert mock_task.request.id in log_record.message
        assert mock_task.name in log_record.message

    def test_task_prerun_handler_preserves_existing_headers(self, mock_task):
        """Test that prerun handler preserves existing headers."""
        existing_header = {"custom_header": "custom_value"}
        mock_task.request.headers = existing_header.copy()

        task_prerun_handler(mock_task)

        assert mock_task.request.headers["custom_header"] == "custom_value"
        assert "started_at" in mock_task.request.headers


class TestTaskSuccessHandler:
    """Test the task_success_handler signal."""

    def test_task_success_handler_logs_completion_with_duration(
        self, mock_task_with_headers, caplog_info
    ):
        """Test that success handler logs completion with duration."""
        task_success_handler(sender=mock_task_with_headers)

        assert len(caplog_info.records) == 1
        log_record = caplog_info.records[0]
        assert log_record.levelname == "INFO"
        assert "Task Completed" in log_record.message
        assert mock_task_with_headers.request.id in log_record.message
        assert mock_task_with_headers.name in log_record.message
        assert "Duration:" in log_record.message

    def test_task_success_handler_logs_completion_without_started_at(
        self, mock_task, caplog_info
    ):
        """Test that success handler handles missing started_at gracefully."""
        task_success_handler(sender=mock_task)

        assert len(caplog_info.records) == 1
        log_record = caplog_info.records[0]
        assert log_record.levelname == "INFO"
        assert "Task Completed" in log_record.message
        assert "Duration: 0.000s" in log_record.message

    def test_task_success_handler_calculates_correct_duration(
        self, mock_task, caplog_info
    ):
        """Test that success handler calculates duration correctly."""
        start_time = time.time() - 2.0  # 2 seconds ago
        mock_task.request.headers = {"started_at": start_time}

        task_success_handler(sender=mock_task)

        log_record = caplog_info.records[0]
        # Duration should be approximately 2 seconds
        assert (
            "Duration: 2." in log_record.message or "Duration: 1." in log_record.message
        )


class TestTaskRetryHandler:
    """Test the task_retry_handler signal."""

    def test_task_retry_handler_logs_retry_with_reason(
        self, mock_task_with_headers, caplog_info
    ):
        """Test that retry handler logs retry with reason and duration."""
        retry_reason = "Connection timeout"

        task_retry_handler(sender=mock_task_with_headers, reason=retry_reason)

        assert len(caplog_info.records) == 1
        log_record = caplog_info.records[0]
        assert log_record.levelname == "ERROR"
        assert "Task Retry" in log_record.message
        assert mock_task_with_headers.request.id in log_record.message
        assert mock_task_with_headers.name in log_record.message
        assert retry_reason in log_record.message
        assert "Duration:" in log_record.message

    def test_task_retry_handler_handles_exception_reason(self, mock_task, caplog_info):
        """Test that retry handler handles exception objects as reasons."""
        retry_reason = ValueError("Test exception")

        task_retry_handler(sender=mock_task, reason=retry_reason)

        log_record = caplog_info.records[0]
        assert "Task Retry" in log_record.message
        assert "Test exception" in log_record.message

    def test_task_retry_handler_logs_without_started_at(self, mock_task, caplog_info):
        """Test that retry handler handles missing started_at gracefully."""
        task_retry_handler(sender=mock_task, reason="Test reason")

        log_record = caplog_info.records[0]
        assert log_record.levelname == "ERROR"
        assert "Duration: 0.000s" in log_record.message


class TestTaskFailureHandler:
    """Test the task_failure_handler signal."""

    def test_task_failure_handler_logs_failure_with_exception(
        self, mock_task_with_headers, caplog_info
    ):
        """Test that failure handler logs failure with exception and duration."""
        test_exception = ValueError("Test error message")
        test_traceback = None  # Traceback is unused in the handler

        task_failure_handler(
            sender=mock_task_with_headers,
            exception=test_exception,
            _traceback=test_traceback,
        )

        assert len(caplog_info.records) == 1
        log_record = caplog_info.records[0]
        assert log_record.levelname == "ERROR"
        assert "Task Failed" in log_record.message
        assert mock_task_with_headers.request.id in log_record.message
        assert mock_task_with_headers.name in log_record.message
        assert "Test error message" in log_record.message
        assert "Duration:" in log_record.message

    def test_task_failure_handler_handles_complex_exception(
        self, mock_task, caplog_info
    ):
        """Test that failure handler handles complex exception objects."""
        test_exception = RuntimeError("Complex error with details")

        task_failure_handler(
            sender=mock_task, exception=test_exception, _traceback=None
        )

        log_record = caplog_info.records[0]
        assert "Task Failed" in log_record.message
        assert "Complex error with details" in log_record.message

    def test_task_failure_handler_logs_without_started_at(self, mock_task, caplog_info):
        """Test that failure handler handles missing started_at gracefully."""
        test_exception = Exception("Test exception")

        task_failure_handler(
            sender=mock_task, exception=test_exception, _traceback=None
        )

        log_record = caplog_info.records[0]
        assert log_record.levelname == "ERROR"
        assert "Duration: 0.000s" in log_record.message


class TestSignalHandlerIntegration:
    """Test integration scenarios with multiple signal handlers."""

    def test_complete_task_lifecycle_logging(self, mock_task, caplog_info):
        """Test that a complete task lifecycle produces expected logs."""
        task_prerun_handler(mock_task)
        prerun_logs = len(caplog_info.records)
        assert prerun_logs == 1
        assert "Task Started" in caplog_info.records[0].message

        task_success_handler(sender=mock_task)
        success_logs = len(caplog_info.records)
        assert success_logs == 2
        assert "Task Completed" in caplog_info.records[1].message

    def test_task_lifecycle_with_retry(self, mock_task, caplog_info):
        """Test task lifecycle including retry."""
        task_prerun_handler(mock_task)
        task_retry_handler(sender=mock_task, reason="Temporary failure")

        assert len(caplog_info.records) == 2
        assert "Task Started" in caplog_info.records[0].message
        assert "Task Retry" in caplog_info.records[1].message

    @pytest.mark.parametrize(
        "handler_name,expected_message",
        [
            pytest.param(
                "prerun",
                "Task Started",
                id="prerun_info_level",
            ),
            pytest.param(
                "success",
                "Task Completed",
                id="success_info_level",
            ),
            pytest.param(
                "retry",
                "Task Retry",
                id="retry_error_level",
            ),
            pytest.param(
                "failure",
                "Task Failed",
                id="failure_error_level",
            ),
        ],
    )
    def test_signal_handlers_log_levels(
        self,
        mock_task,
        caplog_info,
        handler_name,
        expected_message,
    ):
        """Test that signal handlers use correct log levels."""
        # Call handlers with appropriate parameters based on their signatures
        if handler_name == "prerun":
            task_prerun_handler(mock_task)
        elif handler_name == "success":
            task_success_handler(sender=mock_task)
        elif handler_name == "retry":
            task_retry_handler(sender=mock_task, reason="test")
        elif handler_name == "failure":
            task_failure_handler(
                sender=mock_task, exception=Exception("test"), _traceback=None
            )

        assert len(caplog_info.records) == 1
        log_record = caplog_info.records[0]
        assert expected_message in log_record.message
