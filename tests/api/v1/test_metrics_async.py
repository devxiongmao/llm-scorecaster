"""Tests for the metrics async endpoints."""

from dataclasses import dataclass
from unittest.mock import patch, Mock
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from celery.exceptions import WorkerLostError

from src.models.schemas import MetricType, MetricsRequest, TextPair
from tests.test_utils import mock_headers

EVALUATE_URL = "/evaluate"
HEALTH_URL = "/health"


@pytest.fixture(scope="module", name="valid_request_body_webhook")
def valid_request_body_webhook_fixture():
    """Create a valid request body for the metrics async endpoints with a webhook URL."""
    return MetricsRequest(
        text_pairs=[
            TextPair(
                reference="The astronaut is going to the moon.",
                candidate="Eating vegetables provides a variety of benefits",
            ),
            TextPair(
                reference="The quick brown fox jumps over the lazy dog.",
                candidate="A swift auburn fox leaps over a sleepy canine.",
            ),
        ],
        metrics=[MetricType("bert_score")],
        webhook_url="http://www.this-is-a-fake-domain.com/results",
    ).model_dump()


@pytest.fixture(scope="module", name="single_pair_request")
def single_pair_request_fixture():
    """Create a single pair request body for the metrics async endpoints."""
    return MetricsRequest(
        text_pairs=[
            TextPair(reference="Test reference text", candidate="Test candidate text")
        ],
        metrics=[MetricType("bert_score"), MetricType("bleu_score")],
    ).model_dump()


@pytest.fixture(scope="module", name="large_batch_request")
def large_batch_request_fixture():
    """Create a large batch request body for the metrics async endpoints."""
    return MetricsRequest(
        text_pairs=[
            TextPair(reference=f"Reference {i}", candidate=f"Candidate {i}")
            for i in range(10)
        ],
        metrics=[
            MetricType("bert_score"),
            MetricType("bleu_score"),
            MetricType("rouge_score"),
        ],
    ).model_dump()


@pytest.fixture(name="mock_celery_task")
def mock_celery_task_fixture():
    """Mock Celery task result."""
    task = Mock()
    task.id = "test-task-id-123"
    task.state = "PENDING"
    return task


@dataclass
class AuthTestCase:
    """Container for authentication test case parameters."""

    headers: dict[str, str]
    expected_status: int
    test_id: str


@pytest.fixture
def auth_test_cases():
    """Fixture providing authentication test cases."""
    return [
        AuthTestCase(
            headers=mock_headers(),
            expected_status=status.HTTP_200_OK,
            test_id="returns 200 when headers are valid",
        ),
        AuthTestCase(
            headers={"Authorization": "Bearer invalid-key"},
            expected_status=status.HTTP_401_UNAUTHORIZED,
            test_id="returns 401 when api key is invalid",
        ),
        AuthTestCase(
            headers={},
            expected_status=status.HTTP_403_FORBIDDEN,
            test_id="returns 403 when no authorization header",
        ),
    ]


class TestEvaluateMetricsAsync:
    """Tests for the evaluate_metrics_async endpoint."""

    @patch("src.api.v1.metrics_async.uuid.uuid4")
    @patch("src.api.v1.metrics_async.compute_metrics_task")
    def test_evaluate_metrics_async_success(
        self,
        mock_compute_task,
        mock_uuid,
        async_client: TestClient,
        valid_request_body: dict,
    ):
        """Test successful async metrics evaluation."""
        # Mock UUID generation
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-job-id-123")

        # Mock Celery task
        mock_task = Mock()
        mock_task.id = "test-job-id-123"
        mock_compute_task.apply_async.return_value = mock_task

        response = async_client.post(
            EVALUATE_URL,
            json=valid_request_body,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["job_id"] == "test-job-id-123"
        assert data["status"] == "PENDING"
        assert (
            "Job queued successfully. Use the job ID to check status and retrieve results."
            in data["message"]
        )
        assert "estimated_completion_time" in data

        # Verify task was called with correct parameters
        mock_compute_task.apply_async.assert_called_once()
        call_args = mock_compute_task.apply_async.call_args
        assert call_args.kwargs["task_id"] == "test-job-id-123"

        # Check if args are passed as positional arguments or keyword arguments
        if call_args.args:
            request_data = call_args.args[0]
        else:
            # Check if passed as keyword argument
            request_data = call_args.kwargs.get("args", [None])[0]

        assert isinstance(request_data, dict)
        assert len(request_data["text_pairs"]) == 2
        assert len(request_data["metrics"]) == 1

    @patch("src.api.v1.metrics_async.uuid.uuid4")
    @patch("src.api.v1.metrics_async.compute_metrics_task")
    def test_evaluate_metrics_async_success_with_webhook(
        self,
        mock_compute_task,
        mock_uuid,
        async_client: TestClient,
        valid_request_body_webhook: dict,
    ):
        """Test successful async metrics evaluation."""
        # Mock UUID generation
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-job-id-123")

        # Mock Celery task
        mock_task = Mock()
        mock_task.id = "test-job-id-123"
        mock_compute_task.apply_async.return_value = mock_task

        response = async_client.post(
            EVALUATE_URL,
            json=valid_request_body_webhook,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["job_id"] == "test-job-id-123"
        assert data["status"] == "PENDING"
        assert (
            "Job queued successfully. "
            + "Results will be sent to webhook URL: http://www.this-is-a-fake-domain.com/results"
            in data["message"]
        )
        assert "estimated_completion_time" in data

        # Verify task was called with correct parameters
        mock_compute_task.apply_async.assert_called_once()
        call_args = mock_compute_task.apply_async.call_args
        assert call_args.kwargs["task_id"] == "test-job-id-123"

        # Check if args are passed as positional arguments or keyword arguments
        if call_args.args:
            request_data = call_args.args[0]
        else:
            # Check if passed as keyword argument
            request_data = call_args.kwargs.get("args", [None])[0]

        assert isinstance(request_data, dict)
        assert len(request_data["text_pairs"]) == 2
        assert len(request_data["metrics"]) == 1

    @patch("src.api.v1.metrics_async.uuid.uuid4")
    @patch("src.api.v1.metrics_async.compute_metrics_task")
    def test_evaluate_metrics_async_single_pair(
        self,
        mock_compute_task,
        mock_uuid,
        async_client: TestClient,
        single_pair_request: dict,
    ):
        """Test async evaluation with single text pair and multiple metrics."""
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="single-pair-job")

        mock_task = Mock()
        mock_task.id = "single-pair-job"
        mock_compute_task.apply_async.return_value = mock_task

        response = async_client.post(
            EVALUATE_URL,
            json=single_pair_request,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["job_id"] == "single-pair-job"
        assert data["status"] == "PENDING"
        # Estimated time: 1 pair * 2 metrics * 0.5 = 1.0 second
        assert data["estimated_completion_time"] == 1.0

    @patch("src.api.v1.metrics_async.uuid.uuid4")
    @patch("src.api.v1.metrics_async.compute_metrics_task")
    def test_evaluate_metrics_async_large_batch(
        self,
        mock_compute_task,
        mock_uuid,
        async_client: TestClient,
        large_batch_request: dict,
    ):
        """Test async evaluation with large batch."""
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="large-batch-job")

        mock_task = Mock()
        mock_task.id = "large-batch-job"
        mock_compute_task.apply_async.return_value = mock_task

        response = async_client.post(
            EVALUATE_URL,
            json=large_batch_request,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["job_id"] == "large-batch-job"
        assert data["status"] == "PENDING"
        # Estimated time: 10 pairs * 3 metrics * 0.5 = 15.0 seconds
        assert data["estimated_completion_time"] == 15.0

    @patch("src.api.v1.metrics_async.uuid.uuid4")
    @patch("src.api.v1.metrics_async.compute_metrics_task")
    def test_evaluate_metrics_async_task_submission_failure(
        self,
        mock_compute_task,
        mock_uuid,
        async_client: TestClient,
        valid_request_body: dict,
    ):
        """Test handling of Celery task submission failure."""
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="failed-job")

        # Mock task with no ID (submission failure)
        mock_task = Mock()
        mock_task.id = None
        mock_compute_task.apply_async.return_value = mock_task

        response = async_client.post(
            EVALUATE_URL,
            json=valid_request_body,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to submit task to queue" in data["detail"]

    @patch("src.api.v1.metrics_async.uuid.uuid4")
    @patch("src.api.v1.metrics_async.compute_metrics_task")
    def test_evaluate_metrics_async_celery_exception(
        self,
        mock_compute_task,
        mock_uuid,
        async_client: TestClient,
        valid_request_body: dict,
    ):
        """Test handling of Celery exceptions during task submission."""
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="exception-job")

        # Mock Celery exception
        mock_compute_task.apply_async.side_effect = Exception(
            "Celery broker unreachable"
        )

        response = async_client.post(
            EVALUATE_URL,
            json=valid_request_body,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to queue async job" in data["detail"]
        assert "Celery broker unreachable" in data["detail"]

    @patch("src.api.v1.metrics_async.uuid.uuid4")
    def test_evaluate_metrics_async_uuid_generation_failure(
        self, mock_uuid, async_client: TestClient, valid_request_body: dict
    ):
        """Test handling of UUID generation failure."""
        mock_uuid.side_effect = Exception("UUID generation failed")

        response = async_client.post(
            EVALUATE_URL,
            json=valid_request_body,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to queue async job" in data["detail"]
        assert "UUID generation failed" in data["detail"]

    def test_evaluate_metrics_async_invalid_request_body(
        self, async_client: TestClient
    ):
        """Test async evaluation with invalid request body."""
        invalid_request = {
            "text_pairs": [
                {
                    "reference": "Test reference",
                    # Missing candidate field
                }
            ],
            "metrics": ["bert_score"],
        }

        response = async_client.post(
            EVALUATE_URL,
            json=invalid_request,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_evaluate_metrics_async_empty_metrics(self, async_client: TestClient):
        """Test async evaluation with empty metrics list."""
        empty_metrics_request = {
            "text_pairs": [
                {"reference": "Test reference", "candidate": "Test candidate"}
            ],
            "metrics": [],
        }

        response = async_client.post(
            EVALUATE_URL,
            json=empty_metrics_request,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_evaluate_metrics_async_missing_request_body(
        self, async_client: TestClient
    ):
        """Test async evaluation without request body."""
        response = async_client.post(
            EVALUATE_URL,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(
                AuthTestCase(mock_headers(), status.HTTP_200_OK, "valid_headers"),
                id="returns 200 when headers are valid",
            ),
            pytest.param(
                AuthTestCase(
                    {"Authorization": "Bearer invalid-key"},
                    status.HTTP_401_UNAUTHORIZED,
                    "invalid_key",
                ),
                id="returns 401 when api key is invalid",
            ),
            pytest.param(
                AuthTestCase({}, status.HTTP_403_FORBIDDEN, "no_auth_header"),
                id="returns 403 when no authorization header",
            ),
        ],
    )
    @patch("src.api.v1.metrics_async.compute_metrics_task")
    def test_evaluate_metrics_async_authentication(
        self,
        mock_compute_task,
        async_client: TestClient,
        test_case: AuthTestCase,
        valid_request_body: dict,
    ):
        """Test authentication for async evaluate endpoint."""
        mock_task = Mock()
        mock_compute_task.apply_async.return_value = mock_task

        response = async_client.post(
            EVALUATE_URL,
            json=valid_request_body,
            headers=test_case.headers,
        )
        assert response.status_code == test_case.expected_status

    @patch("src.api.v1.metrics_async.uuid.uuid4")
    @patch("src.api.v1.metrics_async.compute_metrics_task")
    def test_evaluate_metrics_async_serialization_check(
        self, mock_compute_task, mock_uuid, async_client: TestClient
    ):
        """Test that Pydantic models are properly converted to dict for Celery."""
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="serialization-test")

        mock_task = Mock()
        mock_task.id = "serialization-test"
        mock_compute_task.apply_async.return_value = mock_task

        request_body = MetricsRequest(
            text_pairs=[TextPair(reference="Test ref", candidate="Test cand")],
            metrics=[MetricType("bert_score")],
        ).model_dump()

        response = async_client.post(
            EVALUATE_URL,
            json=request_body,
            headers=mock_headers(),
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify that the data passed to Celery is a dict (serializable)
        call_args = mock_compute_task.apply_async.call_args

        # Check if args are passed as positional arguments or keyword arguments
        if call_args.args:
            request_data = call_args.args[0]
        else:
            # Check if passed as keyword argument
            request_data = call_args.kwargs.get("args", [None])[0]

        assert isinstance(request_data, dict)
        assert "text_pairs" in request_data
        assert "metrics" in request_data


class TestHealthCheckAsync:
    """Tests for the health_check_async endpoint."""

    @patch("src.api.v1.metrics_async.health_check_task")
    def test_health_check_async_healthy(
        self, mock_health_task, async_client: TestClient
    ):
        """Test health check when workers are healthy."""
        # Mock successful health check
        mock_task = Mock()
        mock_task.get.return_value = {"status": "healthy"}
        mock_health_task.apply_async.return_value = mock_task

        response = async_client.get(HEALTH_URL, headers=mock_headers())

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "healthy"
        assert "Async API and workers are operational" in data["message"]
        assert data["worker_status"] == "healthy"
        assert data["celery_available"] is True

    @patch("src.api.v1.metrics_async.health_check_task")
    def test_health_check_async_workers_unavailable(
        self, mock_health_task, async_client: TestClient
    ):
        """Test health check when workers are unavailable."""
        # Mock task that times out
        mock_task = Mock()
        mock_task.get.side_effect = Exception("Worker timeout")
        mock_health_task.apply_async.return_value = mock_task

        response = async_client.get(HEALTH_URL, headers=mock_headers())

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "degraded"
        assert "workers may be unavailable" in data["message"]
        assert data["worker_status"] == "unavailable"
        assert data["celery_available"] is False

    @patch("src.api.v1.metrics_async.health_check_task")
    def test_health_check_async_worker_lost_error(
        self, mock_health_task, async_client: TestClient
    ):
        """Test health check with WorkerLostError."""
        mock_task = Mock()
        mock_task.get.side_effect = WorkerLostError("Worker process died")
        mock_health_task.apply_async.return_value = mock_task

        response = async_client.get(HEALTH_URL, headers=mock_headers())

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "degraded"
        assert "Worker process died" in data["message"]
        assert data["worker_status"] == "unavailable"
        assert data["celery_available"] is False

    @patch("src.api.v1.metrics_async.health_check_task")
    def test_health_check_async_task_submission_failure(
        self, mock_health_task, async_client: TestClient
    ):
        """Test health check when task submission fails."""
        mock_health_task.apply_async.side_effect = Exception("Broker connection failed")

        response = async_client.get(HEALTH_URL, headers=mock_headers())

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Health check failed" in data["detail"]
        assert "Broker connection failed" in data["detail"]

    @patch("src.api.v1.metrics_async.health_check_task")
    def test_health_check_async_timeout_handling(
        self, mock_health_task, async_client: TestClient
    ):
        """Test health check timeout handling."""
        mock_task = Mock()

        mock_task.get.side_effect = TimeoutError("Task timed out after 5 seconds")
        mock_health_task.apply_async.return_value = mock_task

        response = async_client.get(HEALTH_URL, headers=mock_headers())

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "degraded"
        assert "Task timed out after 5 seconds" in data["message"]
        assert data["worker_status"] == "unavailable"
        assert data["celery_available"] is False

    @patch("src.api.v1.metrics_async.health_check_task")
    def test_health_check_async_worker_status_unknown(
        self, mock_health_task, async_client: TestClient
    ):
        """Test health check when worker returns unknown status."""
        mock_task = Mock()
        mock_task.get.return_value = {}  # No status field
        mock_health_task.apply_async.return_value = mock_task

        response = async_client.get(HEALTH_URL, headers=mock_headers())

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "healthy"
        assert data["worker_status"] == "unknown"  # Default value when status missing
        assert data["celery_available"] is True

    @pytest.mark.parametrize(
        "headers,expected_status",
        [
            pytest.param(
                mock_headers(),
                status.HTTP_200_OK,
                id="returns 200 when headers are valid",
            ),
            pytest.param(
                {"Authorization": "Bearer invalid-key"},
                status.HTTP_401_UNAUTHORIZED,
                id="returns 401 when api key is invalid",
            ),
            pytest.param(
                {},
                status.HTTP_403_FORBIDDEN,
                id="returns 403 when no authorization header",
            ),
        ],
    )
    @patch("src.api.v1.metrics_async.health_check_task")
    def test_health_check_async_authentication(
        self,
        mock_health_task,
        async_client: TestClient,
        headers: dict[str, str],
        expected_status: int,
    ):
        """Test authentication for health check endpoint."""
        mock_task = Mock()
        mock_task.get.return_value = {"status": "healthy"}
        mock_health_task.apply_async.return_value = mock_task

        response = async_client.get(HEALTH_URL, headers=headers)
        assert response.status_code == expected_status

    @patch("src.api.v1.metrics_async.health_check_task")
    def test_health_check_async_response_structure(
        self, mock_health_task, async_client: TestClient
    ):
        """Test that health check response has correct structure."""
        mock_task = Mock()
        mock_task.get.return_value = {"status": "operational", "worker_id": "worker-1"}
        mock_health_task.apply_async.return_value = mock_task

        response = async_client.get(HEALTH_URL, headers=mock_headers())

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check all expected fields are present
        required_fields = ["status", "message", "worker_status", "celery_available"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert data["worker_status"] == "operational"
        assert data["celery_available"] is True

    @patch("src.api.v1.metrics_async.health_check_task")
    def test_health_check_async_task_get_timeout_value(
        self, mock_health_task, async_client: TestClient
    ):
        """Test that health check uses correct timeout value."""
        mock_task = Mock()
        mock_task.get.return_value = {"status": "healthy"}
        mock_health_task.apply_async.return_value = mock_task

        response = async_client.get(HEALTH_URL, headers=mock_headers())

        assert response.status_code == status.HTTP_200_OK

        # Verify that get() was called with timeout=5.0
        mock_task.get.assert_called_once_with(timeout=5.0)
