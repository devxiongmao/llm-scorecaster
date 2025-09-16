"""
Version 1 of the synchronous API for the LLM-scorecaster.

This module provides the routes used for the synchronous operation of the API.
It receives requests and constructs responses that provide back the metric scores
based on user queries.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status
from src.core.computation import compute_metrics_core
from src.models.schemas import (
    IndexResponse,
    MetricsRequest,
    MetricsResponse,
)
from src.api.auth.dependencies import verify_api_key
from src.core.metrics.registry import metric_registry


router = APIRouter()


@router.get("/", response_model=IndexResponse)
async def get_available_metrics(
    _authenticated: bool = Depends(verify_api_key),
) -> IndexResponse:
    """
    Retrieve the list of available metrics.

    This endpoint returns a list of all metrics that can be evaluated
    by the API. It does not perform any metric calculations.
    """
    try:
        available_metrics = metric_registry.list_available_metrics()

        metric_info = [
            metric_registry.get_metric_info(name) for name in available_metrics
        ]

        return IndexResponse(
            success=True,
            message="Available metrics retrieved successfully.",
            results=metric_info,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving available metrics: {str(e)}",
        ) from e


@router.post("/evaluate", response_model=MetricsResponse)
async def evaluate_metrics_sync(
    request: MetricsRequest, _authenticated: bool = Depends(verify_api_key)
) -> MetricsResponse:
    """
    Synchronously evaluate metrics for the provided text pairs.

    This endpoint calculates the requested metrics for each text pair
    and returns results immediately. Suitable for real-time evaluation
    of small to medium-sized batches.
    """
    try:
        start_time = time.time()

        results = compute_metrics_core(request)

        processing_time = time.time() - start_time

        return MetricsResponse(
            success=True,
            message=(
                f"Successfully calculated {len(request.metrics)} metrics for "
                f"{len(request.text_pairs)} text pairs"
            ),
            results=results,
            processing_time_seconds=round(processing_time, 3),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing metrics: {str(e)}",
        ) from e
