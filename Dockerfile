FROM python:3.12.8-slim AS builder

WORKDIR /app

RUN pip install poetry

# Configure build environment variables
# - Set PYTHON_PATH to current directory
# - Disable interactive prompts
# - Create virtual environment in project directory to copy to runtime
# - Enable virtual environment creation
# - Set poetry cache directory to /tmp for caching via Buildkit
ENV PYTHONPATH="." \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml poetry.lock ./

# Install dependencies
# Use Buildkit mount cache to speed up dependency installation
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --extras "all" --no-root


FROM python:3.12.8-slim AS runtime

WORKDIR /app

# Install system dependencies and set up non-root user for security
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r python && \
    useradd -r -g python python && \
    chown -R python:python /app && \
    chmod 755 /app && \
    pip install poetry

# Configure runtime environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copy virtual environment from builder stage
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY . /app