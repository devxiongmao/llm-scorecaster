"""Tests for the main FastAPI application endpoints and functionality."""

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel


from src.main import app
from src.core.settings import settings


@pytest.fixture(name="client")
def client_fixture():
    """Create a test client for the main FastAPI application endpoints."""
    with TestClient(app) as client:
        yield client


def test_root_returns_200(client: TestClient):
    """Test that the root endpoint returns a 200 status code."""
    response = client.get("/")
    assert response.status_code == 200


def test_root_returns_correct_structure(client: TestClient):
    """Test that the root endpoint returns a correct structure."""
    response = client.get("/")
    json_response = response.json()

    expected_keys = ["message", "version", "docs_url", "health_check"]
    assert all(key in json_response for key in expected_keys)


def test_root_returns_expected_values(client: TestClient):
    """Test that the root endpoint returns expected values."""
    response = client.get("/")
    json_response = response.json()

    assert json_response["message"] == "Welcome to llm-scorecaster"
    assert json_response["version"] == settings.version
    assert json_response["docs_url"] == "/docs"
    assert json_response["health_check"] == "/health"


def test_health_returns_200(client: TestClient):
    """Test that the health endpoint returns a 200 status code."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_correct_structure(client: TestClient):
    """Test that the health endpoint returns a correct structure."""
    response = client.get("/health")
    json_response = response.json()

    expected_keys = ["status", "service", "version"]
    assert all(key in json_response for key in expected_keys)


def test_health_returns_expected_values(client: TestClient):
    """Test that the health endpoint returns expected values."""
    response = client.get("/health")
    json_response = response.json()

    assert json_response["status"] == "healthy"
    assert json_response["service"] == "llm-scorecaster"
    assert json_response["version"] == settings.version


def test_health_and_root_have_same_version(client: TestClient):
    """Test that the health and root endpoints have the same version."""
    root_response = client.get("/")
    health_response = client.get("/health")

    assert root_response.json()["version"] == health_response.json()["version"]


def test_nonexistent_endpoint_returns_404(client: TestClient):
    """Test that a nonexistent endpoint returns a 404 status code."""
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404


def test_wrong_method_returns_405(client: TestClient):
    """Test that a wrong method returns a 405 status code."""
    response = client.post("/health")
    assert response.status_code == 405


def test_docs_endpoint_accessible(client: TestClient):
    """Test that the docs endpoint is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_redoc_endpoint_accessible(client: TestClient):
    """Test that the redoc endpoint is accessible."""
    response = client.get("/redoc")
    assert response.status_code == 200


def test_cors_headers_present(client: TestClient):
    """Test that the CORS headers are present."""
    response = client.get("/", headers={"Origin": "https://example.com"})
    assert response.status_code == 200
    # Check if CORS headers are present
    headers_lower = [h.lower() for h in response.headers.keys()]
    assert "access-control-allow-origin" in headers_lower


def test_endpoints_return_json(client: TestClient):
    """Test that the endpoints return JSON."""
    endpoints = ["/", "/health"]

    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.headers["content-type"] == "application/json"


class ValidationRequest(BaseModel):
    """
    Simple Validation Request Model to confirm correct operation of FastAPI.

    This class a standard inferface for us to confirm we're never overwritting or changing any
    core functionality of FastAPI that would otherwise cause this software not work.
    """

    name: str
    age: int


@app.post("/validation-exception")
def validation_exception(request: ValidationRequest):
    """Validation exception endpoint."""
    return {"name": request.name, "age": request.age}


def test_validation_exception_returns_422(client: TestClient):
    """Test that the validation exception endpoint returns a 422 status code."""
    response = client.post("/validation-exception", json={})
    assert response.status_code == 422


def test_validation_exception_returns_validation_errors(client: TestClient):
    """Test that the validation exception endpoint returns validation errors."""
    response = client.post("/validation-exception", json={"name": 123})
    assert response.status_code == 422

    json_response = response.json()
    assert "detail" in json_response
    assert len(json_response["detail"]) == 2

    # Check the name field error
    name_error = json_response["detail"][0]
    assert name_error["loc"] == ["body", "name"]
    assert name_error["msg"] == "Input should be a valid string"
    assert name_error["type"] == "string_type"
    assert name_error["input"] == 123

    # Check the age field error
    age_error = json_response["detail"][1]
    assert age_error["loc"] == ["body", "age"]
    assert age_error["msg"] == "Field required"
    assert age_error["type"] == "missing"


def test_validation_exception_success(client: TestClient):
    """Test that the validation exception endpoint returns a 200 status code."""
    response = client.post("/validation-exception", json={"name": "John", "age": 30})
    assert response.status_code == 200
    assert response.json() == {"name": "John", "age": 30}
