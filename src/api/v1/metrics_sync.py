"""
Version 1 of the synchronous API for the LLM-scorecaster.

This module provides the routes used for the synchronous operation of the API.
It receives requests and constructs responses that provide back the metric scores
based on user queries.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from src.models.schemas import (
    MetricsRequest,
    MetricsResponse,
    TextPairResult,
)
from src.api.auth.dependencies import verify_api_key
from src.core.metrics.registry import metric_registry
from src.models.schemas import TextPair
import time
import asyncio
from typing import List

router = APIRouter()


def compute_metrics_for_request(request: MetricsRequest) -> List[TextPairResult]:
    """Compute actual metrics using the registry."""

    # Discover and get the requested metrics
    metric_registry.discover_metrics()
    metrics = metric_registry.get_metrics([m.value for m in request.metrics])

    results = []

    # Convert Pydantic TextPairs to the format metrics expect
    text_pairs = [
        TextPair(reference=pair.reference, candidate=pair.candidate)
        for pair in request.text_pairs
    ]

    for pair_idx, text_pair in enumerate(text_pairs):
        pair_results = []

        # Compute each requested metric for this text pair
        for metric_name, metric_instance in metrics.items():
            result = metric_instance.compute_single(
                text_pair.reference, text_pair.candidate
            )
            pair_results.append(result)

        # Create the response format
        results.append(
            TextPairResult(
                pair_index=pair_idx,
                reference=text_pair.reference,
                candidate=text_pair.candidate,
                metrics=pair_results,
            )
        )

    return results


@router.post("/evaluate", response_model=MetricsResponse)
async def evaluate_metrics_sync(
    request: MetricsRequest, authenticated: bool = Depends(verify_api_key)
) -> MetricsResponse:
    """
    Synchronously evaluate metrics for the provided text pairs.

    This endpoint calculates the requested metrics for each text pair
    and returns results immediately. Suitable for real-time evaluation
    of small to medium-sized batches.
    """
    try:
        start_time = time.time()

        # Simulate processing time based on batch size
        processing_delay = len(request.text_pairs) * len(request.metrics) * 0.01
        await asyncio.sleep(min(processing_delay, 2.0))  # Cap at 2 seconds for demo

        results = compute_metrics_for_request(request)

        processing_time = time.time() - start_time

        return MetricsResponse(
            success=True,
            message=f"Successfully calculated {len(request.metrics)} metrics for {len(request.text_pairs)} text pairs",
            results=results,
            processing_time_seconds=round(processing_time, 3),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing metrics: {str(e)}",
        )
