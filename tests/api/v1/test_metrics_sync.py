"""Tests for the metrics sync endpoints."""

from unittest.mock import patch, Mock
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.models.schemas import MetricType, MetricsRequest, TextPair
from tests.test_utils import mock_headers

INDEX_URL = "/"
EVALUATE_URL = "/evaluate"
CONFIGURE_URL = "/configure"


@pytest.fixture(scope="module", name="single_pair_request")
def single_pair_request_fixture():
    """Create a single pair request body for the metrics sync endpoints."""
    return MetricsRequest(
        text_pairs=[
            TextPair(reference="Test reference text", candidate="Test candidate text")
        ],
        metrics=[MetricType("bert_score")],
    ).model_dump()


@pytest.fixture(scope="module", name="empty_metrics_request")
def empty_metrics_request_fixture():
    """Create an empty metrics request body for the metrics sync endpoints."""
    return {
        "text_pairs": [{"reference": "Test reference", "candidate": "Test candidate"}],
        "metrics": [],
    }


@pytest.fixture(scope="module", name="invalid_request_body")
def invalid_request_body_fixture():
    """Create an invalid request body for the metrics sync endpoints."""
    return {
        "text_pairs": [
            {
                "reference": "Test reference",
                # Missing candidate field
            }
        ],
        "metrics": ["bert_score"],
    }


@pytest.fixture(scope="module", name="valid_config_request")
def valid_config_request_fixture():
    """Create a valid configuration request for metrics."""
    return {
        "configs": {
            "rouge_score": {"rouge_types": ["rouge1", "rougeL"], "use_stemmer": True},
            "bleu_score": {
                "max_n": 4,
                "smooth_method": "exp",
                "tokenize": "13a",
                "lowercase": False,
            },
        }
    }


@pytest.fixture(scope="module", name="invalid_config_request")
def invalid_config_request_fixture():
    """Create an invalid configuration request with bad parameters."""
    return {
        "configs": {
            "rouge_score": {"rouge_types": ["invalid_rouge_type"], "use_stemmer": True}
        }
    }


@pytest.fixture(scope="module", name="empty_config_request")
def empty_config_request_fixture():
    """Create an empty configuration request."""
    return {"configs": {}}


def test_get_available_metrics_success(sync_client: TestClient):
    """Test successful retrieval of available metrics."""
    response = sync_client.get(
        INDEX_URL,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["success"] is True
    assert "Available metrics retrieved successfully." in data["message"]
    assert isinstance(data["results"], list)

    # Check structure of first metric info
    first_metric = data["results"][0]
    assert "name" in first_metric
    assert "type" in first_metric
    assert "description" in first_metric
    assert "requires_model_download" in first_metric
    assert "class_name" in first_metric


@patch("src.api.v1.metrics_sync.metric_registry")
def test_get_available_metrics_exception_handling(
    mock_metric_registry, sync_client: TestClient
):
    """Test exception handling in get_available_metrics endpoint."""
    mock_metric_registry.list_available_metrics.side_effect = Exception("Test error")

    response = sync_client.get(
        INDEX_URL,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert "An error occurred while retrieving available metrics:" in data["detail"]
    assert "Test error" in data["detail"]


@patch("src.api.v1.metrics_sync.metric_registry")
def test_configure_metrics_success_all(
    mock_metric_registry, sync_client: TestClient, valid_config_request
):
    """Test successful configuration of all requested metrics."""
    # Mock metric instances
    mock_rouge_metric = Mock()
    mock_bleu_metric = Mock()

    mock_metric_registry.get_metrics.return_value = {
        "rouge_score": mock_rouge_metric,
        "bleu_score": mock_bleu_metric,
    }

    response = sync_client.post(
        CONFIGURE_URL,
        json=valid_config_request,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["success"] is True
    assert "Successfully configured 2 metrics" in data["message"]
    assert set(data["configured_metrics"]) == {"rouge_score", "bleu_score"}
    assert data["failed_metrics"] is None

    # Verify configure was called on both metrics
    mock_rouge_metric.configure.assert_called_once()
    mock_bleu_metric.configure.assert_called_once()


@patch("src.api.v1.metrics_sync.metric_registry")
def test_configure_metrics_partial_success(
    mock_metric_registry, sync_client: TestClient
):
    """Test partial success when some metrics fail to configure."""
    mock_rouge_metric = Mock()
    mock_rouge_metric.configure.side_effect = ValueError("Invalid rouge configuration")
    mock_bleu_metric = Mock()

    mock_metric_registry.get_metrics.return_value = {
        "rouge_score": mock_rouge_metric,
        "bleu_score": mock_bleu_metric,
    }

    request_body = {
        "configs": {
            "rouge_score": {"rouge_types": ["rouge1"], "use_stemmer": True},
            "bleu_score": {"max_n": 4, "smooth_method": "exp"},
        }
    }

    response = sync_client.post(
        CONFIGURE_URL,
        json=request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["success"] is True
    assert "Configured 1 metrics, 1 failed" in data["message"]
    assert data["configured_metrics"] == ["bleu_score"]
    assert "rouge_score" in data["failed_metrics"]
    assert "Invalid rouge configuration" in data["failed_metrics"]["rouge_score"]


@patch("src.api.v1.metrics_sync.metric_registry")
def test_configure_metrics_metric_not_found(
    mock_metric_registry, sync_client: TestClient
):
    """Test handling when requested metric is not found in registry."""
    mock_metric_registry.get_metrics.return_value = {}

    request_body = {
        "configs": {"rouge_score": {"rouge_types": ["rouge1"], "use_stemmer": True}}
    }

    response = sync_client.post(
        CONFIGURE_URL,
        json=request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["success"] is False
    assert "Failed to configure any metrics" in data["message"]
    assert data["configured_metrics"] is None
    assert "rouge_score" in data["failed_metrics"]
    assert "not found in registry" in data["failed_metrics"]["rouge_score"]


@patch("src.api.v1.metrics_sync.metric_registry")
def test_configure_metrics_not_implemented_error(
    mock_metric_registry, sync_client: TestClient
):
    """Test handling when metric doesn't implement configure method."""
    mock_metric = Mock()
    mock_metric.configure.side_effect = NotImplementedError("Configure not implemented")

    mock_metric_registry.get_metrics.return_value = {"rouge_score": mock_metric}

    request_body = {
        "configs": {"rouge_score": {"rouge_types": ["rouge1"], "use_stemmer": True}}
    }

    response = sync_client.post(
        CONFIGURE_URL,
        json=request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["success"] is False
    assert "Failed to configure any metrics" in data["message"]
    assert "rouge_score" in data["failed_metrics"]
    assert "Configure not implemented" in data["failed_metrics"]["rouge_score"]


def test_configure_metrics_empty_config(sync_client: TestClient, empty_config_request):
    """Test handling of empty configuration request."""
    response = sync_client.post(
        CONFIGURE_URL,
        json=empty_config_request,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["success"] is False
    assert "Failed to configure any metrics" in data["message"]
    assert data["configured_metrics"] is None
    assert data["failed_metrics"] is None


def test_configure_metrics_invalid_json(sync_client: TestClient):
    """Test handling of invalid JSON in request body."""
    response = sync_client.post(
        CONFIGURE_URL,
        data="invalid json",  # type: ignore
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_configure_metrics_missing_auth(sync_client: TestClient, valid_config_request):
    """Test that authentication is required."""
    response = sync_client.post(
        CONFIGURE_URL,
        json=valid_config_request,
        # No auth headers
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


@patch("src.api.v1.metrics_sync.metric_registry")
def test_configure_metrics_registry_exception(
    mock_metric_registry, sync_client: TestClient, valid_config_request
):
    """Test exception handling when metric registry throws an error."""
    mock_metric_registry.get_metrics.side_effect = Exception("Registry error")

    response = sync_client.post(
        CONFIGURE_URL,
        json=valid_config_request,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert "An error occurred while configuring metrics:" in data["detail"]
    assert "Registry error" in data["detail"]


@patch("src.api.v1.metrics_sync.metric_registry")
def test_configure_metrics_single_metric_success(
    mock_metric_registry, sync_client: TestClient
):
    """Test successful configuration of a single metric."""
    mock_rouge_metric = Mock()

    mock_metric_registry.get_metrics.return_value = {"rouge_score": mock_rouge_metric}

    request_body = {
        "configs": {
            "rouge_score": {"rouge_types": ["rouge1", "rouge2"], "use_stemmer": False}
        }
    }

    response = sync_client.post(
        CONFIGURE_URL,
        json=request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["success"] is True
    assert "Successfully configured 1 metrics" in data["message"]
    assert data["configured_metrics"] == ["rouge_score"]
    assert data["failed_metrics"] is None

    # Verify the correct config was passed
    mock_rouge_metric.configure.assert_called_once()
    call_args = mock_rouge_metric.configure.call_args[0][0]
    assert call_args == {"rouge_types": ["rouge1", "rouge2"], "use_stemmer": False}


def test_evaluate_metrics_success(sync_client: TestClient, valid_request_body: dict):
    """Test successful metrics evaluation."""
    response = sync_client.post(
        EVALUATE_URL,
        json=valid_request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["success"] is True
    assert "Successfully calculated" in data["message"]
    assert "processing_time_seconds" in data
    assert isinstance(data["processing_time_seconds"], float)
    assert len(data["results"]) == 2  # Two text pairs

    # Check first result structure
    first_result = data["results"][0]
    assert first_result["pair_index"] == 0
    assert "reference" in first_result
    assert "candidate" in first_result
    assert len(first_result["metrics"]) == 1

    # Check metric structure
    bert_metric = next(
        m for m in first_result["metrics"] if m["metric_name"] == "bert_score"
    )
    assert "score" in bert_metric
    assert "details" in bert_metric
    assert bert_metric["details"] is not None  # bert_score should have details


def test_evaluate_metrics_single_pair(
    sync_client: TestClient, single_pair_request: dict
):
    """Test evaluation with single text pair."""
    response = sync_client.post(
        EVALUATE_URL,
        json=single_pair_request,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["success"] is True
    assert len(data["results"]) == 1
    assert len(data["results"][0]["metrics"]) == 1


def test_evaluate_metrics_empty_metrics_list(
    sync_client: TestClient, empty_metrics_request: dict
):
    """Test evaluation with empty metrics list."""
    response = sync_client.post(
        EVALUATE_URL,
        json=empty_metrics_request,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_evaluate_metrics_invalid_request_body(
    sync_client: TestClient, invalid_request_body: dict
):
    """Test evaluation with invalid request body."""
    response = sync_client.post(
        EVALUATE_URL,
        json=invalid_request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_evaluate_metrics_missing_request_body(sync_client: TestClient):
    """Test evaluation without request body."""
    response = sync_client.post(
        EVALUATE_URL,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


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
def test_evaluate_metrics_authentication(
    sync_client: TestClient,
    headers: dict[str, str],
    expected_status: int,
    valid_request_body: dict,
):
    """Test authentication for evaluate endpoint."""
    response = sync_client.post(
        EVALUATE_URL,
        json=valid_request_body,
        headers=headers,
    )
    assert response.status_code == expected_status


@patch("src.api.v1.metrics_sync.compute_metrics_core")
def test_evaluate_metrics_exception_handling(
    mock_generate_results, sync_client: TestClient, valid_request_body: dict
):
    """Test exception handling in evaluate endpoint."""
    # Make the function raise an exception
    mock_generate_results.side_effect = Exception("Test error")

    response = sync_client.post(
        EVALUATE_URL,
        json=valid_request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert "An error occurred while processing metrics" in data["detail"]
    assert "Test error" in data["detail"]


# To-do: Readd the metrics
def test_evaluate_metrics_different_metric_types(sync_client: TestClient):
    """Test evaluation with different metric types."""
    request_body = MetricsRequest(
        text_pairs=[TextPair(reference="Test reference", candidate="Test candidate")],
        metrics=[
            MetricType("bert_score"),
            MetricType("bleu_score"),
            MetricType("rouge_score"),
        ],
    ).model_dump()

    response = sync_client.post(
        EVALUATE_URL,
        json=request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    metrics = data["results"][0]["metrics"]

    # Check that all metrics are present
    metric_names = [m["metric_name"] for m in metrics]
    assert "bert_score" in metric_names
    assert "bleu_score" in metric_names
    assert "rouge_score" in metric_names

    # Check that bert_score, bleu_score and rouge_score have details while others might not
    bert_metric = next(m for m in metrics if m["metric_name"] == "bert_score")
    assert bert_metric["details"] is not None

    bleu_metric = next(m for m in metrics if m["metric_name"] == "bleu_score")
    assert bleu_metric["details"] is not None

    rouge_metric = next(m for m in metrics if m["metric_name"] == "rouge_score")
    assert rouge_metric["details"] is not None


def test_evaluate_metrics_response_message_content(
    sync_client: TestClient, valid_request_body: dict
):
    """Test that response message contains correct information."""
    response = sync_client.post(
        EVALUATE_URL,
        json=valid_request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    expected_message = "Successfully calculated 1 metrics for 2 text pairs"
    assert data["message"] == expected_message


def test_evaluate_metrics_pair_index_assignment(sync_client: TestClient):
    """Test that pair indices are assigned correctly."""
    request_body = MetricsRequest(
        text_pairs=[
            TextPair(reference="First ref", candidate="First cand"),
            TextPair(reference="Second ref", candidate="Second cand"),
            TextPair(reference="Third ref", candidate="Third cand"),
        ],
        metrics=[MetricType("bert_score")],
    ).model_dump()

    response = sync_client.post(
        EVALUATE_URL,
        json=request_body,
        headers=mock_headers(),
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    results = data["results"]

    assert len(results) == 3
    assert results[0]["pair_index"] == 0
    assert results[1]["pair_index"] == 1
    assert results[2]["pair_index"] == 2

    # Check that reference and candidate texts are preserved
    assert results[0]["reference"] == "First ref"
    assert results[0]["candidate"] == "First cand"
