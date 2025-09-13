"""
Tests for compute_metrics_task Celery task.

This test suite covers the async metrics computation task including:
- Successful computation scenarios
- Error handling and failure cases
- Webhook notifications (success and failure)
- Progress tracking and state updates
- Various input configurations
"""

import json
from unittest.mock import patch, MagicMock
import pytest
from werkzeug import Request, Response
from pytest_httpserver import HTTPServer

from src.celery.tasks.metrics import (
    compute_metrics_for_request,
    _compute_metrics_task_logic,
)


class TestComputeMetricsForRequest:
    """Test the core computation function."""

    def test_compute_metrics_single_pair_bleu(self):
        """Test computing BLEU score for a single text pair."""
        request_data = {
            "text_pairs": [
                {
                    "reference": "The cat sat on the mat",
                    "candidate": "A cat was sitting on a mat",
                }
            ],
            "metrics": ["bleu_score"],
        }

        results = compute_metrics_for_request(request_data)

        assert len(results) == 1
        result = results[0]
        assert result["pair_index"] == 0
        assert result["reference"] == "The cat sat on the mat"
        assert result["candidate"] == "A cat was sitting on a mat"
        assert len(result["metrics"]) == 1
        assert result["metrics"][0]["metric_name"] == "bleu_score"
        assert isinstance(result["metrics"][0]["score"], float)
        assert 0.0 <= result["metrics"][0]["score"] <= 1.0

    def test_compute_metrics_multiple_pairs_bleu(self):
        """Test computing BLEU score for multiple text pairs."""
        request_data = {
            "text_pairs": [
                {
                    "reference": "The cat sat on the mat",
                    "candidate": "A cat was sitting on a mat",
                },
                {
                    "reference": "Hello world, how are you?",
                    "candidate": "Hi there world, how are you doing?",
                },
            ],
            "metrics": ["bleu_score"],
        }

        results = compute_metrics_for_request(request_data)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert result["pair_index"] == i
            assert len(result["metrics"]) == 1
            assert result["metrics"][0]["metric_name"] == "bleu_score"
            assert isinstance(result["metrics"][0]["score"], float)

    def test_compute_metrics_with_batch_size(self):
        """Test computing metrics with batch processing."""
        request_data = {
            "text_pairs": [
                {
                    "reference": "The cat sat on the mat",
                    "candidate": "A cat was sitting on a mat",
                },
                {
                    "reference": "Hello world, how are you?",
                    "candidate": "Hi there world, how are you doing?",
                },
            ],
            "metrics": ["bleu_score"],
            "batch_size": 2,
        }

        results = compute_metrics_for_request(request_data)

        assert len(results) == 2
        for result in results:
            assert len(result["metrics"]) == 1
            assert result["metrics"][0]["metric_name"] == "bleu_score"

    def test_compute_metrics_empty_strings(self):
        """Test computing metrics with empty strings."""
        request_data = {
            "text_pairs": [{"reference": "", "candidate": ""}],
            "metrics": ["bleu_score"],
        }

        results = compute_metrics_for_request(request_data)

        assert len(results) == 1
        result = results[0]
        assert result["reference"] == ""
        assert result["candidate"] == ""
        assert len(result["metrics"]) == 1


class TestComputeMetricsTaskLogic:
    """Test the core task logic function."""

    def test_successful_computation(self):
        """Test successful metrics computation task logic."""
        # Create a mock task instance
        mock_task = MagicMock()
        mock_task.request.id = "test-job-123"

        request_data = {
            "text_pairs": [
                {
                    "reference": "The cat sat on the mat",
                    "candidate": "A cat was sitting on a mat",
                }
            ],
            "metrics": ["bleu_score"],
        }

        result = _compute_metrics_task_logic(mock_task, request_data)

        # Verify the result structure
        assert result["success"] is True
        assert "Successfully calculated 1 metrics for 1 text pairs" in result["message"]
        assert len(result["results"]) == 1
        assert "processing_time_seconds" in result
        assert result["total_operations"] == 1

        # Verify task state updates were called
        assert mock_task.update_state.call_count == 2

        # Check the first call (PROCESSING start)
        first_call = mock_task.update_state.call_args_list[0]
        assert first_call[1]["state"] == "PROCESSING"
        assert first_call[1]["meta"]["message"] == "Computing metrics..."
        assert first_call[1]["meta"]["progress"] == 0

        # Check the second call (PROCESSING finalize)
        second_call = mock_task.update_state.call_args_list[1]
        assert second_call[1]["state"] == "PROCESSING"
        assert second_call[1]["meta"]["message"] == "Finalizing results..."
        assert second_call[1]["meta"]["progress"] == 50

    def test_computation_with_multiple_pairs_and_progress_tracking(self):
        """Test computation with multiple pairs for proper progress tracking."""
        mock_task = MagicMock()
        mock_task.request.id = "test-job-456"

        request_data = {
            "text_pairs": [
                {
                    "reference": "The cat sat on the mat",
                    "candidate": "A cat was sitting on a mat",
                },
                {"reference": "Hello world", "candidate": "Hi world"},
                {"reference": "Good morning", "candidate": "Morning!"},
            ],
            "metrics": ["bleu_score"],
        }

        result = _compute_metrics_task_logic(mock_task, request_data)

        assert result["success"] is True
        assert "Successfully calculated 1 metrics for 3 text pairs" in result["message"]
        assert len(result["results"]) == 3
        assert result["total_operations"] == 3

    def test_task_failure_handling(self):
        """Test proper error handling when computation fails."""
        mock_task = MagicMock()
        mock_task.request.id = "test-job-error"

        # Create invalid request data that will cause an error
        request_data = {
            "text_pairs": [],  # Empty list should cause validation error
            "metrics": ["bleu_score"],
        }

        with pytest.raises(Exception):
            _compute_metrics_task_logic(mock_task, request_data)

        # Verify failure state was set
        failure_calls = [
            call
            for call in mock_task.update_state.call_args_list
            if call[1].get("state") == "FAILURE"
        ]
        assert len(failure_calls) >= 1


class TestWebhookIntegration:
    """Test webhook notification functionality."""

    @pytest.fixture
    def http_server(self):
        """Create a test HTTP server for webhook testing."""
        server = HTTPServer(host="127.0.0.1", port=0)
        server.start()
        yield server
        server.stop()

    def test_successful_computation_with_webhook(self, http_server):
        """Test successful computation with webhook notification."""
        webhook_data = []

        def webhook_handler(request: Request) -> Response:
            """Handler for webhook requests."""
            webhook_data.append(json.loads(request.get_data(as_text=True)))
            return Response("OK", status=200)

        http_server.expect_request("/webhook").respond_with_handler(webhook_handler)
        webhook_url = f"http://127.0.0.1:{http_server.port}/webhook"

        mock_task = MagicMock()
        mock_task.request.id = "test-webhook-job"

        request_data = {
            "text_pairs": [
                {
                    "reference": "The cat sat on the mat",
                    "candidate": "A cat was sitting on a mat",
                }
            ],
            "metrics": ["bleu_score"],
            "webhook_url": webhook_url,
        }

        result = _compute_metrics_task_logic(mock_task, request_data)

        # Verify the task completed successfully
        assert result["success"] is True
        assert result["webhook_sent"] is True

        # Verify webhook was called with correct data
        assert len(webhook_data) == 1
        webhook_payload = webhook_data[0]
        assert webhook_payload["job_id"] == "test-webhook-job"
        assert webhook_payload["status"] == "COMPLETED"
        assert webhook_payload["data"]["success"] is True
        assert len(webhook_payload["data"]["results"]) == 1

    def test_failed_computation_with_webhook(self, http_server):
        """Test failed computation with webhook error notification."""
        webhook_data = []

        def webhook_handler(request: Request) -> Response:
            webhook_data.append(json.loads(request.get_data(as_text=True)))
            return Response("OK", status=200)

        http_server.expect_request("/webhook").respond_with_handler(webhook_handler)
        webhook_url = f"http://127.0.0.1:{http_server.port}/webhook"

        mock_task = MagicMock()
        mock_task.request.id = "test-webhook-error-job"

        # Invalid request data to trigger error
        request_data = {
            "text_pairs": [],  # This should cause a validation error
            "metrics": ["bleu_score"],
            "webhook_url": webhook_url,
        }

        with pytest.raises(Exception):
            _compute_metrics_task_logic(mock_task, request_data)

        # Verify webhook was called for the error
        assert len(webhook_data) == 1
        webhook_payload = webhook_data[0]
        assert webhook_payload["job_id"] == "test-webhook-error-job"
        assert webhook_payload["status"] == "FAILED"
        assert webhook_payload["data"]["success"] is False

    def test_webhook_failure_handling(self, http_server):
        """Test handling when webhook endpoint is unreachable."""
        # Use a non-existent port for webhook URL
        webhook_url = "http://127.0.0.1:99999/webhook"

        mock_task = MagicMock()
        mock_task.request.id = "test-webhook-fail-job"

        request_data = {
            "text_pairs": [
                {
                    "reference": "The cat sat on the mat",
                    "candidate": "A cat was sitting on a mat",
                }
            ],
            "metrics": ["bleu_score"],
            "webhook_url": webhook_url,
        }

        result = _compute_metrics_task_logic(mock_task, request_data)

        # Task should still succeed even if webhook fails
        assert result["success"] is True
        assert result["webhook_sent"] is False
        assert "webhook_error" in result

    def test_computation_without_webhook(self):
        """Test computation without webhook URL."""
        mock_task = MagicMock()
        mock_task.request.id = "test-no-webhook-job"

        request_data = {
            "text_pairs": [
                {
                    "reference": "The cat sat on the mat",
                    "candidate": "A cat was sitting on a mat",
                }
            ],
            "metrics": ["bleu_score"],
            # No webhook_url provided
        }

        result = _compute_metrics_task_logic(mock_task, request_data)

        assert result["success"] is True
        assert "webhook_sent" not in result


def test_task_execution_sync():
    """Test synchronous execution of the Celery task (for testing purposes)."""
    request_data = {
        "text_pairs": [
            {
                "reference": "The cat sat on the mat",
                "candidate": "A cat was sitting on a mat",
            }
        ],
        "metrics": ["bleu_score"],
    }

    # Execute the task function directly (not through Celery)
    # This tests the task logic without requiring a running Celery worker
    mock_task = MagicMock()
    mock_task.request.id = "direct-test-job"

    with patch("src.celery.tasks.metrics._compute_metrics_task") as mock_celery_task:
        mock_celery_task.return_value = _compute_metrics_task_logic(
            mock_task, request_data
        )
        result = mock_celery_task(request_data)

    assert result["success"] is True
    assert len(result["results"]) == 1


def test_very_long_texts():
    """Test computation with very long texts."""
    long_text = "This is a very long sentence. " * 1000

    request_data = {
        "text_pairs": [
            {"reference": long_text, "candidate": long_text[:500] + " modified ending."}
        ],
        "metrics": ["bleu_score"],
    }

    results = compute_metrics_for_request(request_data)
    assert len(results) == 1
    assert isinstance(results[0]["metrics"][0]["score"], float)


def test_unicode_and_special_characters():
    """Test computation with Unicode and special characters."""
    request_data = {
        "text_pairs": [
            {
                "reference": "Héllo wörld! 🌍 Testing ñoñó characters.",
                "candidate": "Hello world! 🌎 Testing nono characters.",
            }
        ],
        "metrics": ["bleu_score"],
    }

    results = compute_metrics_for_request(request_data)
    assert len(results) == 1
    assert isinstance(results[0]["metrics"][0]["score"], float)


def test_identical_texts():
    """Test computation with identical reference and candidate texts."""
    identical_text = "The quick brown fox jumps over the lazy dog."

    request_data = {
        "text_pairs": [{"reference": identical_text, "candidate": identical_text}],
        "metrics": ["bleu_score"],
    }

    results = compute_metrics_for_request(request_data)
    assert len(results) == 1
    # BLEU score should be 1.0 for identical texts
    assert results[0]["metrics"][0]["score"] == 1.0


def test_large_batch_processing():
    """Test processing a large batch of text pairs."""
    # Create a larger dataset
    text_pairs = []
    for i in range(50):
        text_pairs.append(
            {
                "reference": f"This is reference text number {i}.",
                "candidate": f"This is candidate text number {i} with slight modification.",
            }
        )

    request_data = {
        "text_pairs": text_pairs,
        "metrics": ["bleu_score"],
        "batch_size": 10,
    }

    results = compute_metrics_for_request(request_data)
    assert len(results) == 50
    for result in results:
        assert len(result["metrics"]) == 1
        assert result["metrics"][0]["metric_name"] == "bleu_score"
