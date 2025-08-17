from fastapi import APIRouter, Depends, HTTPException, status
from src.models.schemas import (
    MetricsRequest,
    MetricsResponse,
    TextPairResult,
    MetricResult,
)
from src.api.auth.dependencies import verify_api_key
import time
import random
import asyncio
from typing import List

router = APIRouter()


def generate_placeholder_results(request: MetricsRequest) -> List[TextPairResult]:
    """
    Generate placeholder metric results for testing purposes.
    This will be replaced with actual metric calculations later.
    """
    results = []

    for idx, text_pair in enumerate(request.text_pairs):
        # Generate placeholder metrics for each requested metric type
        metrics = []
        for metric_name in request.metrics:
            # Generate realistic placeholder scores based on metric type
            if metric_name == "bert_score":
                score = round(random.uniform(0.7, 0.95), 3)
                details = {
                    "precision": round(score + random.uniform(-0.05, 0.05), 3),
                    "recall": round(score + random.uniform(-0.05, 0.05), 3),
                    "f1": score,
                }
            elif metric_name in ["bleu"]:
                score = round(random.uniform(0.2, 0.8), 3)
                details = None
            elif metric_name.startswith("rouge"):
                score = round(random.uniform(0.3, 0.85), 3)
                details = None
            elif metric_name == "align_score":
                score = round(random.uniform(0.6, 0.9), 3)
                details = None
            else:
                score = round(random.uniform(0.5, 0.9), 3)
                details = None

            metrics.append(
                MetricResult(metric_name=metric_name, score=score, details=details)
            )

        results.append(
            TextPairResult(
                pair_index=idx,
                reference=text_pair.reference,
                candidate=text_pair.candidate,
                metrics=metrics,
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

        # Generate placeholder results
        results = generate_placeholder_results(request)

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
