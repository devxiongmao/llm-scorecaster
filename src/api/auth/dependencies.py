from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer
from src.core.settings import settings

security = HTTPBearer()


async def verify_api_key(token: str = Depends(security)) -> bool:
    """
    Verify the provided API key against the configured key.

    Args:
        token: The bearer token from the Authorization header

    Returns:
        bool: True if authentication is successful

    Raises:
        HTTPException: If the API key is invalid
    """
    if token.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True
