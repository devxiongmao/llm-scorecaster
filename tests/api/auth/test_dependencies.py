import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from src.api.auth.dependencies import verify_api_key


class TestVerifyApiKey:
    """Test cases for the verify_api_key function."""

    @pytest.fixture
    def mock_token(self):
        """Create a mock HTTPAuthorizationCredentials token."""
        token = Mock(spec=HTTPAuthorizationCredentials)
        return token

    @pytest.fixture
    def valid_api_key(self):
        """Return a valid API key for testing."""
        return "test-api-key-123"

    @patch("src.core.settings")
    async def test_verify_api_key_success(
        self, mock_settings, mock_token, valid_api_key
    ):
        """Test successful API key verification."""
        # Arrange
        mock_settings.api_key = valid_api_key
        mock_token.credentials = valid_api_key

        # Act
        result = await verify_api_key(mock_token)

        # Assert
        assert result is True

    @patch("src.core.settings")
    async def test_verify_api_key_invalid_key(
        self, mock_settings, mock_token, valid_api_key
    ):
        """Test API key verification with invalid key."""
        # Arrange
        mock_settings.api_key = valid_api_key
        mock_token.credentials = "invalid-api-key"

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}

    @patch("src.core.settings")
    async def test_verify_api_key_empty_credentials(
        self, mock_settings, mock_token, valid_api_key
    ):
        """Test API key verification with empty credentials."""
        # Arrange
        mock_settings.api_key = valid_api_key
        mock_token.credentials = ""

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"

    @patch("src.core.settings")
    async def test_verify_api_key_none_credentials(
        self, mock_settings, mock_token, valid_api_key
    ):
        """Test API key verification with None credentials."""
        # Arrange
        mock_settings.api_key = valid_api_key
        mock_token.credentials = None

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"

    @patch("src.core.settings")
    async def test_verify_api_key_case_sensitive(self, mock_settings, mock_token):
        """Test that API key verification is case sensitive."""
        # Arrange
        mock_settings.api_key = "TestApiKey"
        mock_token.credentials = "testapikey"

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @patch("src.core.settings")
    async def test_verify_api_key_with_whitespace(self, mock_settings, mock_token):
        """Test API key verification with whitespace in credentials."""
        # Arrange
        valid_key = "test-api-key"
        mock_settings.api_key = valid_key
        mock_token.credentials = f" {valid_key} "

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @patch("src.core.settings")
    async def test_verify_api_key_similar_but_different(
        self, mock_settings, mock_token
    ):
        """Test API key verification with similar but different key."""
        # Arrange
        mock_settings.api_key = "test-api-key-123"
        mock_token.credentials = "test-api-key-124"  # One character different

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @patch("src.core.settings")
    async def test_verify_api_key_empty_settings_key(self, mock_settings, mock_token):
        """Test API key verification when settings key is empty."""
        # Arrange
        mock_settings.api_key = ""
        mock_token.credentials = "some-key"

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @patch("src.core.settings")
    async def test_verify_api_key_both_empty(self, mock_settings, mock_token):
        """Test API key verification when both keys are empty."""
        # Arrange
        mock_settings.api_key = ""
        mock_token.credentials = ""

        # Act
        result = await verify_api_key(mock_token)

        # Assert
        assert result is True

    @patch("src.core.settings")
    async def test_verify_api_key_unicode_characters(self, mock_settings, mock_token):
        """Test API key verification with unicode characters."""
        # Arrange
        unicode_key = "test-api-key-🔑-123"
        mock_settings.api_key = unicode_key
        mock_token.credentials = unicode_key

        # Act
        result = await verify_api_key(mock_token)

        # Assert
        assert result is True

    @patch("src.core.settings")
    async def test_verify_api_key_long_key(self, mock_settings, mock_token):
        """Test API key verification with very long key."""
        # Arrange
        long_key = "a" * 1000  # Very long key
        mock_settings.api_key = long_key
        mock_token.credentials = long_key

        # Act
        result = await verify_api_key(mock_token)

        # Assert
        assert result is True


# Integration-style tests (if you want to test the dependency injection)
class TestVerifyApiKeyIntegration:
    """Integration tests for verify_api_key with actual FastAPI dependency injection."""

    def test_security_dependency_type(self):
        """Test that the security dependency is properly configured."""
        from src.api.auth.dependencies import security
        from fastapi.security import HTTPBearer

        assert isinstance(security, HTTPBearer)

    @patch("src.core.settings")
    async def test_dependency_returns_boolean(self, mock_settings):
        """Test that the function returns a boolean when used as dependency."""
        # Arrange
        mock_settings.api_key = "test-key"
        mock_token = Mock(spec=HTTPAuthorizationCredentials)
        mock_token.credentials = "test-key"

        # Act
        result = await verify_api_key(mock_token)

        # Assert
        assert isinstance(result, bool)
        assert result is True
