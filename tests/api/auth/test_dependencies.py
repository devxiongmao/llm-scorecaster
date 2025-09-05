"""Tests for the authentication dependencies."""

from unittest.mock import Mock, patch
import pytest
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.auth.dependencies import verify_api_key
from src.api.auth.dependencies import security


@pytest.fixture(name="mock_token")
def mock_token_fixture():
    """Create a mock HTTPAuthorizationCredentials token."""
    token = Mock(spec=HTTPAuthorizationCredentials)
    return token


@pytest.fixture(name="valid_api_key")
def valid_api_key_fixture():
    """Return a valid API key for testing."""
    return "test-api-key-123"


@patch("src.api.auth.dependencies.settings")
async def test_verify_api_key_success(mock_settings, mock_token, valid_api_key):
    """Test successful API key verification."""
    mock_settings.api_key = valid_api_key
    mock_token.credentials = valid_api_key

    result = await verify_api_key(mock_token)

    assert result is True


@patch("src.core.settings")
async def test_verify_api_key_invalid_key(mock_settings, mock_token, valid_api_key):
    """Test API key verification with invalid key."""
    mock_settings.api_key = valid_api_key
    mock_token.credentials = "invalid-api-key"

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(mock_token)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@patch("src.core.settings")
async def test_verify_api_key_empty_credentials(
    mock_settings, mock_token, valid_api_key
):
    """Test API key verification with empty credentials."""
    mock_settings.api_key = valid_api_key
    mock_token.credentials = ""

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(mock_token)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"


@patch("src.core.settings")
async def test_verify_api_key_none_credentials(
    mock_settings, mock_token, valid_api_key
):
    """Test API key verification with None credentials."""
    mock_settings.api_key = valid_api_key
    mock_token.credentials = None

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(mock_token)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"


@patch("src.core.settings")
async def test_verify_api_key_case_sensitive(mock_settings, mock_token):
    """Test that API key verification is case sensitive."""
    mock_settings.api_key = "TestApiKey"
    mock_token.credentials = "testapikey"

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(mock_token)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


@patch("src.core.settings")
async def test_verify_api_key_with_whitespace(mock_settings, mock_token):
    """Test API key verification with whitespace in credentials."""
    valid_key = "test-api-key"
    mock_settings.api_key = valid_key
    mock_token.credentials = f" {valid_key} "

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(mock_token)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


@patch("src.core.settings")
async def test_verify_api_key_similar_but_different(mock_settings, mock_token):
    """Test API key verification with similar but different key."""
    mock_settings.api_key = "test-api-key-123"
    mock_token.credentials = "test-api-key-124"  # One character different

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(mock_token)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


@patch("src.core.settings")
async def test_verify_api_key_empty_settings_key(mock_settings, mock_token):
    """Test API key verification when settings key is empty."""
    mock_settings.api_key = ""
    mock_token.credentials = "some-key"

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(mock_token)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


@patch("src.api.auth.dependencies.settings")
async def test_verify_api_key_both_empty(mock_settings, mock_token):
    """Test API key verification when both keys are empty."""
    mock_settings.api_key = ""
    mock_token.credentials = ""

    result = await verify_api_key(mock_token)

    assert result is True


@patch("src.api.auth.dependencies.settings")
async def test_verify_api_key_unicode_characters(mock_settings, mock_token):
    """Test API key verification with unicode characters."""
    unicode_key = "test-api-key-🔑-123"
    mock_settings.api_key = unicode_key
    mock_token.credentials = unicode_key

    result = await verify_api_key(mock_token)

    assert result is True


@patch("src.api.auth.dependencies.settings")
async def test_verify_api_key_long_key(mock_settings, mock_token):
    """Test API key verification with very long key."""
    long_key = "a" * 1000  # Very long key
    mock_settings.api_key = long_key
    mock_token.credentials = long_key

    result = await verify_api_key(mock_token)

    assert result is True


def test_security_dependency_type():
    """Test that the security dependency is properly configured."""

    assert isinstance(security, HTTPBearer)


@patch("src.api.auth.dependencies.settings")
async def test_dependency_returns_boolean(mock_settings):
    """Test that the function returns a boolean when used as dependency."""
    mock_settings.api_key = "test-key"
    mock_token = Mock(spec=HTTPAuthorizationCredentials)
    mock_token.credentials = "test-key"

    result = await verify_api_key(mock_token)

    assert isinstance(result, bool)
    assert result is True
