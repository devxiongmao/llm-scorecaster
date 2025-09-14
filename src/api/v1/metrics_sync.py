"""
Version 1 of the synchronous API for the LLM-scorecaster.

This module provides the routes used for the synchronous operation of the API.
It receives requests and constructs responses that provide back the metric scores
based on user queries.
"""

import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from src.core.computation import compute_metrics_core
from src.models.schemas import (
    MetricsRequest,
    MetricsResponse,
)
from src.api.auth.dependencies import verify_api_key

router = APIRouter()


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
