from unittest.mock import patch, Mock
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.v1.jobs import router as jobs_router
from tests.test_utils import mock_domain_app, mock_headers

# URL constants
STATUS_URL = "/status/{job_id}"
RESULTS_URL = "/results/{job_id}"
CANCEL_URL = "/{job_id}"
LIST_JOBS_URL = "/"


@pytest.fixture(name="client")
def client_fixture():
    with TestClient(mock_domain_app(jobs_router)) as client:
        yield client


@pytest.fixture(name="mock_celery_task")
def mock_celery_task_fixture():
    """Mock Celery AsyncResult object."""
    task = Mock()
    task.state = "SUCCESS"
    task.info = {}
    task.result = {
        "success": True,
        "message": "Job completed successfully",
        "results": [
            {
                "pair_index": 0,
                "reference": "Test reference",
                "candidate": "Test candidate",
                "metrics": [
                    {
                        "metric_name": "bert_score",
                        "score": 0.85,
                        "details": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
                    }
                ],
            }
        ],
        "processing_time_seconds": 2.5,
    }
    return task


class TestGetJobStatus:
    """Tests for the get_job_status endpoint."""

    def test_get_job_status_pending(self, client: TestClient, mock_celery_task):
        """Test getting status of a pending job."""
        mock_celery_task.state = "PENDING"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                STATUS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "PENDING"
        assert data["message"] == "Job is queued and waiting to be processed"
        assert data["progress"] == 0

    def test_get_job_status_processing(self, client: TestClient, mock_celery_task):
        """Test getting status of a processing job."""
        mock_celery_task.state = "PROCESSING"
        mock_celery_task.info = {
            "message": "Processing batch 2 of 5",
            "progress": 40,
            "total_pairs": 100,
            "total_metrics": 3,
        }

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                STATUS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "PROCESSING"
        assert data["message"] == "Processing batch 2 of 5"
        assert data["progress"] == 40
        assert data["total_pairs"] == 100
        assert data["total_metrics"] == 3

    def test_get_job_status_success(self, client: TestClient, mock_celery_task):
        """Test getting status of a successful job."""
        mock_celery_task.state = "SUCCESS"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                STATUS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "COMPLETED"
        assert data["message"] == "Job completed successfully. Results are ready."
        assert data["progress"] == 100
        assert data["completed"] is True

    def test_get_job_status_failure(self, client: TestClient, mock_celery_task):
        """Test getting status of a failed job."""
        mock_celery_task.state = "FAILURE"
        mock_celery_task.info = {"error": "Invalid metric configuration"}

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                STATUS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "FAILED"
        assert data["message"] == "Invalid metric configuration"
        assert data["error"] == "Invalid metric configuration"
        assert data["failed"] is True

    def test_get_job_status_retry(self, client: TestClient, mock_celery_task):
        """Test getting status of a retrying job."""
        mock_celery_task.state = "RETRY"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                STATUS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "RETRYING"
        assert data["message"] == "Job failed and is being retried"
        assert data["progress"] == 0

    def test_get_job_status_revoked(self, client: TestClient, mock_celery_task):
        """Test getting status of a cancelled job."""
        mock_celery_task.state = "REVOKED"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                STATUS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "CANCELLED"
        assert data["message"] == "Job was cancelled"
        assert data["failed"] is True

    def test_get_job_status_unknown_state(self, client: TestClient, mock_celery_task):
        """Test getting status of a job in unknown state."""
        mock_celery_task.state = "UNKNOWN_STATE"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                STATUS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "UNKNOWN_STATE"
        assert data["message"] == "Job is in state: UNKNOWN_STATE"

    def test_get_job_status_exception_handling(self, client: TestClient):
        """Test exception handling in get_job_status."""
        with patch(
            "src.api.v1.jobs.get_celery_task_info",
            side_effect=Exception("Connection error"),
        ):
            response = client.get(
                STATUS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to retrieve job status" in data["detail"]
        assert "Connection error" in data["detail"]

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
    def test_get_job_status_authentication(
        self,
        client: TestClient,
        headers: dict[str, str],
        expected_status: int,
        mock_celery_task,
    ):
        """Test authentication for get job status endpoint."""
        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                STATUS_URL.format(job_id="test-job-123"),
                headers=headers,
            )
        assert response.status_code == expected_status


class TestGetJobResults:
    """Tests for the get_job_results endpoint."""

    def test_get_job_results_success(self, client: TestClient, mock_celery_task):
        """Test getting results from a successful job."""
        mock_celery_task.state = "SUCCESS"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                RESULTS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Job completed successfully"
        assert data["processing_time_seconds"] == 2.5
        assert len(data["results"]) == 1

        result = data["results"][0]
        assert result["pair_index"] == 0
        assert result["reference"] == "Test reference"
        assert result["candidate"] == "Test candidate"
        assert len(result["metrics"]) == 1
        assert result["metrics"][0]["metric_name"] == "bert_score"
        assert result["metrics"][0]["score"] == 0.85

    def test_get_job_results_pending_job(self, client: TestClient, mock_celery_task):
        """Test getting results from a pending job."""
        mock_celery_task.state = "PENDING"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                RESULTS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Job is still pending" in data["detail"]

    def test_get_job_results_processing_job(self, client: TestClient, mock_celery_task):
        """Test getting results from a processing job."""
        mock_celery_task.state = "PROCESSING"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                RESULTS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Job is still processing" in data["detail"]

    def test_get_job_results_failed_job(self, client: TestClient, mock_celery_task):
        """Test getting results from a failed job."""
        mock_celery_task.state = "FAILURE"
        mock_celery_task.info = {"error": "Processing failed"}

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                RESULTS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Job failed: Processing failed" in data["detail"]

    def test_get_job_results_cancelled_job(self, client: TestClient, mock_celery_task):
        """Test getting results from a cancelled job."""
        mock_celery_task.state = "REVOKED"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                RESULTS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Job was cancelled and results are not available" in data["detail"]

    def test_get_job_results_invalid_result_format(
        self, client: TestClient, mock_celery_task
    ):
        """Test getting results when task result has invalid format."""
        mock_celery_task.state = "SUCCESS"
        mock_celery_task.result = "invalid_result_format"  # Not a dict

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                RESULTS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Invalid result format from completed job" in data["detail"]

    def test_get_job_results_exception_handling(self, client: TestClient):
        """Test exception handling in get_job_results."""
        with patch(
            "src.api.v1.jobs.get_celery_task_info",
            side_effect=Exception("Database error"),
        ):
            response = client.get(
                RESULTS_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to retrieve job results" in data["detail"]
        assert "Database error" in data["detail"]

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
    def test_get_job_results_authentication(
        self,
        client: TestClient,
        headers: dict[str, str],
        expected_status: int,
        mock_celery_task,
    ):
        """Test authentication for get job results endpoint."""
        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.get(
                RESULTS_URL.format(job_id="test-job-123"),
                headers=headers,
            )
        assert response.status_code == expected_status


class TestCancelJob:
    """Tests for the cancel_job endpoint."""

    def test_cancel_job_success(self, client: TestClient, mock_celery_task):
        """Test successful job cancellation."""
        mock_celery_task.state = "PROCESSING"
        mock_celery_task.revoke = Mock()

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.delete(
                CANCEL_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["message"] == "Job cancelled successfully"
        assert data["status"] == "cancelled"
        assert data["terminated"] is False

        # Verify revoke was called
        mock_celery_task.revoke.assert_called_once_with(terminate=False)

    def test_cancel_job_with_terminate(self, client: TestClient, mock_celery_task):
        """Test job cancellation with terminate flag."""
        mock_celery_task.state = "PROCESSING"
        mock_celery_task.revoke = Mock()

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.delete(
                CANCEL_URL.format(job_id="test-job-123") + "?terminate=true",
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["message"] == "Job terminated successfully"
        assert data["status"] == "cancelled"
        assert data["terminated"] is True

        # Verify revoke was called with terminate=True
        mock_celery_task.revoke.assert_called_once_with(terminate=True)

    def test_cancel_completed_job(self, client: TestClient, mock_celery_task):
        """Test cancelling an already completed job."""
        mock_celery_task.state = "SUCCESS"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.delete(
                CANCEL_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["message"] == "Job already completed, cannot cancel"
        assert data["status"] == "already_completed"

    def test_cancel_already_failed_job(self, client: TestClient, mock_celery_task):
        """Test cancelling an already failed job."""
        mock_celery_task.state = "FAILURE"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.delete(
                CANCEL_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["message"] == "Job already in terminal state: FAILURE"
        assert data["status"] == "already_terminal"

    def test_cancel_already_revoked_job(self, client: TestClient, mock_celery_task):
        """Test cancelling an already cancelled job."""
        mock_celery_task.state = "REVOKED"

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.delete(
                CANCEL_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["message"] == "Job already in terminal state: REVOKED"
        assert data["status"] == "already_terminal"

    def test_cancel_job_exception_handling(self, client: TestClient):
        """Test exception handling in cancel_job."""
        with patch(
            "src.api.v1.jobs.get_celery_task_info",
            side_effect=Exception("Network error"),
        ):
            response = client.delete(
                CANCEL_URL.format(job_id="test-job-123"),
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to cancel job" in data["detail"]
        assert "Network error" in data["detail"]

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
    def test_cancel_job_authentication(
        self,
        client: TestClient,
        headers: dict[str, str],
        expected_status: int,
        mock_celery_task,
    ):
        """Test authentication for cancel job endpoint."""
        mock_celery_task.state = "PROCESSING"
        mock_celery_task.revoke = Mock()

        with patch(
            "src.api.v1.jobs.get_celery_task_info", return_value=mock_celery_task
        ):
            response = client.delete(
                CANCEL_URL.format(job_id="test-job-123"),
                headers=headers,
            )
        assert response.status_code == expected_status


class TestListActiveJobs:
    """Tests for the list_active_jobs endpoint."""

    def test_list_active_jobs_success(self, client: TestClient):
        """Test successful listing of active jobs."""
        mock_inspect = Mock()
        mock_inspect.active.return_value = {
            "worker1": [
                {"id": "job-123", "name": "compute_metrics", "worker": "worker1"},
                {"id": "job-456", "name": "compute_metrics", "worker": "worker1"},
            ]
        }
        mock_inspect.scheduled.return_value = {
            "worker1": [{"id": "job-789", "worker": "worker1"}]
        }

        with patch(
            "src.api.v1.jobs.celery_app.control.inspect", return_value=mock_inspect
        ):
            response = client.get(
                LIST_JOBS_URL,
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "Found 2 active and 1 scheduled jobs" in data["message"]
        assert len(data["active_jobs"]) == 2
        assert len(data["scheduled_jobs"]) == 1
        assert data["total_count"] == 3

        # Check active job structure
        active_job = data["active_jobs"][0]
        assert active_job["job_id"] == "job-123"
        assert active_job["worker"] == "worker1"
        assert active_job["name"] == "compute_metrics"
        assert active_job["status"] == "PROCESSING"

    def test_list_active_jobs_no_jobs(self, client: TestClient):
        """Test listing when no active jobs exist."""
        mock_inspect = Mock()
        mock_inspect.active.return_value = None
        mock_inspect.scheduled.return_value = None

        with patch(
            "src.api.v1.jobs.celery_app.control.inspect", return_value=mock_inspect
        ):
            response = client.get(
                LIST_JOBS_URL,
                headers=mock_headers(),
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "No active jobs found"
        assert data["active_jobs"] == []
        assert data["scheduled_jobs"] == []
        assert data["total_count"] == 0

    def test_list_active_jobs_exception_handling(self, client: TestClient):
        """Test exception handling in list_active_jobs."""
        with patch(
            "src.api.v1.jobs.celery_app.control.inspect",
            side_effect=Exception("Celery error"),
        ):
            response = client.get(
                LIST_JOBS_URL,
                headers=mock_headers(),
            )

        assert (
            response.status_code == status.HTTP_200_OK
        )  # This endpoint handles exceptions gracefully
        data = response.json()
        assert "Unable to retrieve job list" in data["message"]
        assert data["active_jobs"] == []
        assert data["scheduled_jobs"] == []
        assert data["total_count"] == 0
        assert data["error"] == "Celery error"

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
    def test_list_active_jobs_authentication(
        self, client: TestClient, headers: dict[str, str], expected_status: int
    ):
        """Test authentication for list active jobs endpoint."""
        mock_inspect = Mock()
        mock_inspect.active.return_value = {}
        mock_inspect.scheduled.return_value = {}

        with patch(
            "src.api.v1.jobs.celery_app.control.inspect", return_value=mock_inspect
        ):
            response = client.get(
                LIST_JOBS_URL,
                headers=headers,
            )
        assert response.status_code == expected_status
