"""Test utilities for the LLM Scorecaster application."""

from fastapi import APIRouter, FastAPI

from src.core.settings import settings


def mock_domain_app(router: APIRouter):
    """
    Create a mock FastAPI application with the given router.

    This utility function is used in tests to create a minimal FastAPI
    application instance with a specific router for testing purposes.

    Args:
        router: The APIRouter to include in the mock application

    Returns:
        FastAPI: A configured FastAPI application instance
    """
    app = FastAPI()
    app.include_router(router)
    return app


def mock_headers():
    """
    Generate mock authorization headers for testing.

    Creates a dictionary containing authorization headers with a bearer token
    using the API key from settings. Used for testing authenticated endpoints.

    Returns:
        dict: Dictionary containing authorization headers with bearer token
    """
    return {"Authorization": f"Bearer {settings.api_key}"}
