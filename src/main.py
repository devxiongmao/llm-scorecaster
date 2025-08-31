from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.settings import settings
from src.api.v1 import metrics_sync, metrics_async, jobs

app = FastAPI(
    title="llm-scorecaster",
    version=settings.version,
    description="An open-source LLM metrics evaluation API supporting BERT score, BLEU, ROUGE, and more",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    metrics_sync.router,
    prefix="/api/v1/metrics",
    tags=["synchronous-metrics"],
)

app.include_router(
    metrics_async.router, prefix="/api/v1/async", tags=["asynchronous-metrics"]
)

app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])


@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Welcome to llm-scorecaster",
        "version": settings.version,
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "llm-scorecaster",
        "version": settings.version,
    }
