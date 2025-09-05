"""Application settings loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


def get_version() -> str:
    """Read version from VERSION file in the repository root."""
    try:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        if version_file.exists():
            return version_file.read_text().strip()
        return "1.0.0"
    except Exception:
        return "1.0.0"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Authentication
    api_key: str = Field(
        default="your-secret-api-key-here",
        description="API key for authentication. Must be set by user.",
    )

    # Redis configuration
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for queue and result storage",
    )

    version: str = Field(
        default_factory=get_version, description="API version read from VERSION file"
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
    )
    debug: bool = Field(default=False, description="Enable debug mode for development")

    # Webhook settings
    max_timeout: int = Field(
        default=30, description="Request timeout in seconds for the webhook POST"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retry attempts for the webhook"
    )


settings = Settings()
