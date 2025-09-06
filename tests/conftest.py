"""Shared pytest fixtures for API tests."""

import pytest
from fastapi.testclient import TestClient

from src.models.schemas import MetricType, MetricsRequest, TextPair
from src.api.v1.metrics_sync import router as metrics_sync_router
from src.api.v1.metrics_async import router as metrics_async_router
from tests.test_utils import mock_domain_app


@pytest.fixture(name="sync_client")
def sync_client_fixture():
    """Create a test client for the metrics sync endpoints."""
    with TestClient(mock_domain_app(metrics_sync_router)) as client:
        yield client


@pytest.fixture(name="async_client")
def async_client_fixture():
    """Create a test client for the metrics async endpoints."""
    with TestClient(mock_domain_app(metrics_async_router)) as client:
        yield client


@pytest.fixture(scope="module", name="valid_request_body")
def valid_request_body_fixture():
    """Create a valid request body for the metrics endpoints."""
    return MetricsRequest(
        text_pairs=[
            TextPair(
                reference="The quick brown fox jumps over the lazy dog.",
                candidate="A swift auburn fox leaps over a sleepy canine.",
            ),
            TextPair(
                reference="Hello world, how are you?",
                candidate="Hi world, how are you doing?",
            ),
        ],
        metrics=[MetricType("bert_score")],
    ).model_dump()


@pytest.fixture(name="sample_text_pair")
def sample_text_pair_fixture():
    """Single text pair for testing."""
    return TextPair(
        reference="The quick brown fox jumps over the lazy dog",
        candidate="A fast brown fox leaps over a sleepy dog",
    )


@pytest.fixture(name="sample_text_pairs")
def sample_text_pairs_fixture():
    """Multiple text pairs for testing."""
    return [
        TextPair(reference="Hello world", candidate="Hi world"),
        TextPair(reference="Python is great", candidate="Python is awesome"),
        TextPair(reference="Testing code", candidate="Code testing"),
    ]


@pytest.fixture(name="empty_text_pairs")
def empty_text_pairs_fixture():
    """Empty list of text pairs."""
    return []
