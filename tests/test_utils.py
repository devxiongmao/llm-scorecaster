from fastapi import APIRouter, FastAPI

from src.core.settings import settings


def mock_domain_app(router: APIRouter):
    app = FastAPI()
    app.include_router(router)
    return app


def mock_headers():
    return {"Authorization": f"Bearer {settings.api_key}"}
