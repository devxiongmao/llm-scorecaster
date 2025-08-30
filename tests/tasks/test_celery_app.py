from unittest.mock import patch, Mock
from typing import Dict, Any
from pydantic import ValidationError
import pytest

from src.tasks.celery_app import (
    _compute_metrics_task_logic,
    health_check_task,
    compute_metrics_for_request,
    celery_app,
)


@pytest.fixture(name="sample_request_data")
def sample_request_data_fixture():
    """Sample request data for testing."""
    return {
        "text_pairs": [
            {
                "reference": "The quick brown fox jumps over the lazy dog.",
                "candidate": "A swift auburn fox leaps over a sleepy canine.",
            },
            {
                "reference": "Hello world, how are you?",
                "candidate": "Hi world, how are you doing?",
            },
        ],
        "metrics": ["bert_score"],
        "batch_size": 32,
    }


@pytest.fixture(name="single_pair_request_data")
def single_pair_request_data_fixture():
    """Single text pair request for testing."""
    return {
        "text_pairs": [
            {
                "reference": "Test reference text",
                "candidate": "Test candidate text",
            }
        ],
        "metrics": ["bert_score", "bleu_score"],
        "batch_size": 32,
    }


@pytest.fixture(name="empty_request_data")
def empty_request_data_fixture():
    """Empty request data for testing edge cases."""
    return {
        "text_pairs": [],
        "metrics": [],
        "batch_size": 32,
    }


@pytest.fixture(name="mock_metric_instance")
def mock_metric_instance_fixture():
    """Mock metric instance for testing."""
    mock_metric = Mock()
    mock_metric.compute_single.return_value = {
        "metric_name": "bert_score",
        "score": 0.85,
        "details": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
    }
    return mock_metric


@pytest.fixture(name="mock_metric_registry")
def mock_metric_registry_fixture(mock_metric_instance):
    """Mock metric registry for testing."""
    mock_registry = Mock()
    mock_registry.discover_metrics.return_value = None
    mock_registry.get_metrics.return_value = {"bert_score": mock_metric_instance}
    return mock_registry


class TestComputeMetricsForRequest:
    """Tests for the compute_metrics_for_request function."""

    @patch("src.tasks.celery_app.metric_registry")
    def test_compute_metrics_for_request_success(
        self, mock_registry, sample_request_data: Dict[str, Any], mock_metric_instance
    ):
        """Test successful metrics computation for request."""
        # Setup mock registry
        mock_registry.discover_metrics.return_value = None
        mock_registry.get_metrics.return_value = {"bert_score": mock_metric_instance}

        results = compute_metrics_for_request(sample_request_data)

        # Verify registry interactions
        mock_registry.discover_metrics.assert_called_once()
        mock_registry.get_metrics.assert_called_once_with(["bert_score"])

        # Verify results structure
        assert len(results) == 2  # Two text pairs
        assert isinstance(results, list)

        # Check first result
        first_result = results[0]
        assert first_result["pair_index"] == 0
        assert (
            first_result["reference"] == "The quick brown fox jumps over the lazy dog."
        )
        assert (
            first_result["candidate"]
            == "A swift auburn fox leaps over a sleepy canine."
        )
        assert len(first_result["metrics"]) == 1
        assert first_result["metrics"][0]["metric_name"] == "bert_score"
        assert first_result["metrics"][0]["score"] == 0.85

        # Check second result
        second_result = results[1]
        assert second_result["pair_index"] == 1
        assert second_result["reference"] == "Hello world, how are you?"
        assert second_result["candidate"] == "Hi world, how are you doing?"

        # Verify metric computation was called for each pair
        assert mock_metric_instance.compute_single.call_count == 2

    @patch("src.tasks.celery_app.metric_registry")
    def test_compute_metrics_for_request_multiple_metrics(
        self, mock_registry, single_pair_request_data: Dict[str, Any]
    ):
        """Test metrics computation with multiple metrics."""
        # Setup multiple mock metrics
        mock_bert = Mock()
        mock_bert.compute_single.return_value = {
            "metric_name": "bert_score",
            "score": 0.85,
            "details": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
        }

        mock_bleu = Mock()
        mock_bleu.compute_single.return_value = {
            "metric_name": "bleu_score",
            "score": 0.75,
            "details": {"bleu_1": 0.8, "bleu_2": 0.7},
        }

        mock_registry.discover_metrics.return_value = None
        mock_registry.get_metrics.return_value = {
            "bert_score": mock_bert,
            "bleu_score": mock_bleu,
        }

        results = compute_metrics_for_request(single_pair_request_data)

        # Verify results structure
        assert len(results) == 1  # One text pair
        result = results[0]
        assert len(result["metrics"]) == 2  # Two metrics

        # Check both metrics are present
        metric_names = [m["metric_name"] for m in result["metrics"]]
        assert "bert_score" in metric_names
        assert "bleu_score" in metric_names

        # Verify both metrics were computed
        mock_bert.compute_single.assert_called_once()
        mock_bleu.compute_single.assert_called_once()

    @patch("src.tasks.celery_app.metric_registry")
    def test_compute_metrics_for_request_empty_data(
        self, mock_registry, empty_request_data: Dict[str, Any]
    ):
        """Test metrics computation with empty data raises ValidationError."""
        with pytest.raises(ValidationError):
            compute_metrics_for_request(empty_request_data)

        # Since validation fails, these methods shouldn't be called
        mock_registry.discover_metrics.assert_not_called()
        mock_registry.get_metrics.assert_not_called()

    @patch("src.tasks.celery_app.metric_registry")
    def test_compute_metrics_for_request_metric_computation_failure(
        self, mock_registry, sample_request_data: Dict[str, Any]
    ):
        """Test handling of metric computation failures."""
        mock_metric = Mock()
        mock_metric.compute_single.side_effect = Exception("Metric computation failed")

        mock_registry.discover_metrics.return_value = None
        mock_registry.get_metrics.return_value = {"bert_score": mock_metric}

        with pytest.raises(Exception, match="Metric computation failed"):
            compute_metrics_for_request(sample_request_data)

    @patch("src.tasks.celery_app.metric_registry")
    def test_compute_metrics_for_request_invalid_request_data(self, _):
        """Test handling of invalid request data."""
        invalid_data = {"invalid": "data"}

        with pytest.raises(Exception):  # Pydantic validation error
            compute_metrics_for_request(invalid_data)

    @patch("src.tasks.celery_app.metric_registry")
    def test_compute_metrics_for_request_registry_discovery_failure(
        self, mock_registry, sample_request_data: Dict[str, Any]
    ):
        """Test handling of metric registry discovery failure."""
        mock_registry.discover_metrics.side_effect = Exception(
            "Registry discovery failed"
        )

        with pytest.raises(Exception, match="Registry discovery failed"):
            compute_metrics_for_request(sample_request_data)

    @patch("src.tasks.celery_app.metric_registry")
    def test_compute_metrics_for_request_result_serialization(
        self, mock_registry, sample_request_data: Dict[str, Any], mock_metric_instance
    ):
        """Test that results are properly serialized as dicts."""
        mock_registry.discover_metrics.return_value = None
        mock_registry.get_metrics.return_value = {"bert_score": mock_metric_instance}

        results = compute_metrics_for_request(sample_request_data)

        # Verify all results are dicts (serializable)
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, dict)
            assert isinstance(result["metrics"], list)
            for metric in result["metrics"]:
                assert isinstance(metric, dict)


class TestComputeMetricsTask:
    """Tests for the compute_metrics_task Celery task."""

    def test_compute_metrics_task_success(self, sample_request_data: Dict[str, Any]):
        """Test successful execution of compute_metrics_task."""
        # Create a mock task instance
        mock_task = Mock()
        mock_task.update_state = Mock()

        # Mock the compute_metrics_for_request function
        expected_results = [
            {
                "pair_index": 0,
                "reference": "Test ref",
                "candidate": "Test cand",
                "metrics": [{"metric_name": "bert_score", "score": 0.85}],
            }
        ]

        with patch(
            "src.tasks.celery_app.compute_metrics_for_request",
            return_value=expected_results,
        ):
            with patch(
                "src.tasks.celery_app.time.time", side_effect=[1000.0, 1002.5]
            ):  # 2.5 second duration
                result = _compute_metrics_task_logic(mock_task, sample_request_data)

        # Verify task state updates
        assert mock_task.update_state.call_count == 2

        # Check first update (start processing)
        first_call = mock_task.update_state.call_args_list[0]
        assert first_call[1]["state"] == "PROCESSING"
        assert first_call[1]["meta"]["message"] == "Computing metrics..."
        assert first_call[1]["meta"]["progress"] == 0
        assert first_call[1]["meta"]["total_pairs"] == 2
        assert first_call[1]["meta"]["total_metrics"] == 1

        # Check second update (finalizing)
        second_call = mock_task.update_state.call_args_list[1]
        assert second_call[1]["state"] == "PROCESSING"
        assert second_call[1]["meta"]["message"] == "Finalizing results..."
        assert second_call[1]["meta"]["progress"] == 50

        # Verify final result
        assert result["success"] is True
        assert "Successfully calculated 1 metrics for 2 text pairs" in result["message"]
        assert result["results"] == expected_results
        assert result["processing_time_seconds"] == 2.5
        assert result["total_operations"] == 2  # 2 pairs * 1 metric

    def test_compute_metrics_task_empty_request(
        self, empty_request_data: Dict[str, Any]
    ):
        """Test compute_metrics_task with empty request data."""
        mock_task = Mock()
        mock_task.update_state = Mock()

        with patch("src.tasks.celery_app.compute_metrics_for_request", return_value=[]):
            with patch("src.tasks.celery_app.time.time", side_effect=[1000.0, 1001.0]):
                result = _compute_metrics_task_logic(mock_task, empty_request_data)

        # Verify result for empty data
        assert result["success"] is True
        assert not result["results"]
        assert result["total_operations"] == 0
        assert "Successfully calculated 0 metrics for 0 text pairs" in result["message"]

    def test_compute_metrics_task_computation_failure(
        self, sample_request_data: Dict[str, Any]
    ):
        """Test compute_metrics_task handling of computation failures."""
        mock_task = Mock()
        mock_task.update_state = Mock()

        # Mock compute_metrics_for_request to raise an exception
        with patch(
            "src.tasks.celery_app.compute_metrics_for_request",
            side_effect=ValueError("Computation error"),
        ):
            with pytest.raises(ValueError, match="Computation error"):
                _compute_metrics_task_logic(mock_task, sample_request_data)

        # Verify task state was updated to FAILURE
        failure_call = mock_task.update_state.call_args_list[-1]
        assert failure_call[1]["state"] == "FAILURE"
        assert "Task failed: Computation error" in failure_call[1]["meta"]["error"]
        assert failure_call[1]["meta"]["exc_type"] == "ValueError"

    def test_compute_metrics_task_processing_time_calculation(
        self, sample_request_data: Dict[str, Any]
    ):
        """Test that processing time is calculated correctly."""
        mock_task = Mock()
        mock_task.update_state = Mock()

        # Mock time.time to return specific values
        start_time = 1000.0
        end_time = 1003.456
        expected_duration = round(end_time - start_time, 3)

        with patch("src.tasks.celery_app.compute_metrics_for_request", return_value=[]):
            with patch(
                "src.tasks.celery_app.time.time", side_effect=[start_time, end_time]
            ):
                result = _compute_metrics_task_logic(mock_task, sample_request_data)

        assert result["processing_time_seconds"] == expected_duration

    def test_compute_metrics_task_total_operations_calculation(self):
        """Test calculation of total operations."""
        mock_task = Mock()
        mock_task.update_state = Mock()

        request_data = {
            "text_pairs": [
                {"ref": "a", "cand": "b"},
                {"ref": "c", "cand": "d"},
            ],  # 2 pairs
            "metrics": ["bert_score", "bleu_score", "rouge_score"],  # 3 metrics
        }

        with patch("src.tasks.celery_app.compute_metrics_for_request", return_value=[]):
            with patch("src.tasks.celery_app.time.time", side_effect=[1000.0, 1001.0]):
                result = _compute_metrics_task_logic(mock_task, request_data)

        # Should be 2 pairs * 3 metrics = 6 operations
        assert result["total_operations"] == 6

    def test_compute_metrics_task_state_updates_content(
        self, sample_request_data: Dict[str, Any]
    ):
        """Test content of task state updates."""
        mock_task = Mock()
        mock_task.update_state = Mock()

        with patch("src.tasks.celery_app.compute_metrics_for_request", return_value=[]):
            with patch("src.tasks.celery_app.time.time", side_effect=[1000.0, 1001.0]):
                _compute_metrics_task_logic(mock_task, sample_request_data)

        # Verify all state updates have required fields
        for call in mock_task.update_state.call_args_list:
            meta = call[1]["meta"]
            assert "message" in meta
            assert "progress" in meta
            assert "total_pairs" in meta
            assert "total_metrics" in meta
            assert isinstance(meta["progress"], int)

    def test_compute_metrics_task_exception_handling_types(self):
        """Test handling of different exception types."""
        mock_task = Mock()
        mock_task.update_state = Mock()

        test_exceptions = [
            (ValueError("Value error"), "ValueError"),
            (KeyError("Key error"), "KeyError"),
            (RuntimeError("Runtime error"), "RuntimeError"),
        ]

        for exception, expected_type in test_exceptions:
            mock_task.update_state.reset_mock()

            with patch(
                "src.tasks.celery_app.compute_metrics_for_request",
                side_effect=exception,
            ):
                with pytest.raises(type(exception)):
                    _compute_metrics_task_logic(
                        mock_task, {"text_pairs": [], "metrics": []}
                    )

            # Verify exception type is correctly recorded
            failure_call = mock_task.update_state.call_args_list[-1]
            assert failure_call[1]["meta"]["exc_type"] == expected_type
            assert str(exception) in failure_call[1]["meta"]["exc_message"]


class TestHealthCheckTask:
    """Tests for the health_check_task."""

    def test_health_check_task_success(self):
        """Test successful health check task."""
        result = health_check_task()

        assert isinstance(result, dict)
        assert result["status"] == "healthy"
        assert result["message"] == "Celery worker is running"

    def test_health_check_task_return_type(self):
        """Test that health check returns correct types."""
        result = health_check_task()

        assert isinstance(result, dict)
        assert isinstance(result["status"], str)
        assert isinstance(result["message"], str)

    def test_health_check_task_required_fields(self):
        """Test that health check returns all required fields."""
        result = health_check_task()

        required_fields = ["status", "message"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_health_check_task_consistent_response(self):
        """Test that health check returns consistent responses."""
        result1 = health_check_task()
        result2 = health_check_task()

        assert result1 == result2


class TestCeleryAppConfiguration:
    """Tests for Celery app configuration."""

    def test_celery_app_configuration(self):
        """Test that Celery app is properly configured."""
        # Check basic app properties
        assert celery_app.main == "llm_scorecaster"

        # Check configuration values
        conf = celery_app.conf
        assert conf.task_serializer == "json"
        assert conf.accept_content == ["json"]
        assert conf.result_serializer == "json"
        assert conf.timezone == "UTC"
        assert conf.enable_utc is True

    def test_celery_app_task_settings(self):
        """Test Celery task-specific settings."""
        conf = celery_app.conf

        assert conf.result_expires == 3600
        assert conf.result_persistent is True
        assert conf.worker_prefetch_multiplier == 1
        assert conf.task_acks_late is True
        assert conf.worker_max_tasks_per_child == 1000

    def test_celery_app_retry_settings(self):
        """Test Celery retry settings."""
        conf = celery_app.conf

        assert conf.task_reject_on_worker_lost is True
        assert conf.task_default_retry_delay == 60
        assert conf.task_max_retries == 3

    def test_celery_app_includes_tasks(self):
        """Test that Celery app includes the task module."""
        assert "src.tasks.celery_app" in celery_app.conf.include


class TestIntegrationScenarios:
    """Integration tests for complete task workflows."""

    @patch("src.tasks.celery_app.metric_registry")
    def test_full_workflow_single_metric(self, mock_registry, mock_metric_instance):
        """Test complete workflow with single metric."""
        mock_registry.discover_metrics.return_value = None
        mock_registry.get_metrics.return_value = {"bert_score": mock_metric_instance}

        request_data = {
            "text_pairs": [{"reference": "Hello world", "candidate": "Hi world"}],
            "metrics": ["bert_score"],
            "batch_size": 32,
        }

        mock_task = Mock()
        mock_task.update_state = Mock()

        result = _compute_metrics_task_logic(mock_task, request_data)

        # Verify complete workflow
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["total_operations"] == 1

        # Verify metric was computed
        mock_metric_instance.compute_single.assert_called_once_with(
            "Hello world", "Hi world"
        )

        # Verify task state updates occurred
        assert mock_task.update_state.call_count == 2

    @patch("src.tasks.celery_app.metric_registry")
    def test_full_workflow_multiple_metrics_and_pairs(self, mock_registry):
        """Test complete workflow with multiple metrics and text pairs."""
        # Setup multiple metrics
        mock_bert = Mock()
        mock_bert.compute_single.return_value = {
            "metric_name": "bert_score",
            "score": 0.85,
            "details": {},
        }

        mock_bleu = Mock()
        mock_bleu.compute_single.return_value = {
            "metric_name": "bleu_score",
            "score": 0.75,
            "details": {},
        }

        mock_registry.discover_metrics.return_value = None
        mock_registry.get_metrics.return_value = {
            "bert_score": mock_bert,
            "bleu_score": mock_bleu,
        }

        request_data = {
            "text_pairs": [
                {"reference": "First ref", "candidate": "First cand"},
                {"reference": "Second ref", "candidate": "Second cand"},
            ],
            "metrics": ["bert_score", "bleu_score"],
            "batch_size": 32,
        }

        mock_task = Mock()
        mock_task.update_state = Mock()

        with patch("src.tasks.celery_app.time.time", side_effect=[1000.0, 1005.0]):
            result = _compute_metrics_task_logic(mock_task, request_data)

        # Verify results
        assert result["success"] is True
        assert len(result["results"]) == 2  # Two text pairs
        assert result["total_operations"] == 4  # 2 pairs * 2 metrics

        # Verify each result has both metrics
        for pair_result in result["results"]:
            assert len(pair_result["metrics"]) == 2
            metric_names = [m["metric_name"] for m in pair_result["metrics"]]
            assert "bert_score" in metric_names
            assert "bleu_score" in metric_names

        # Verify each metric was computed for each pair
        assert mock_bert.compute_single.call_count == 2
        assert mock_bleu.compute_single.call_count == 2
