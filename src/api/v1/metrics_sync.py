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
    MetricsConfigRequest,
    MetricsConfigResponse,
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


@router.post("/configure", response_model=MetricsConfigResponse)
async def configure_metrics(
    request: MetricsConfigRequest, _authenticated: bool = Depends(verify_api_key)
) -> MetricsConfigResponse:
    """
    Configure specific metrics to be used in future evaluations.

    This endpoint allows you to set configuration parameters for individual metrics,
    such as model paths, thresholds, or other metric-specific settings.
    Each metric can have its own unique configuration.
    """

    try:
        # Get the metric instances from the registry
        metric_names = [metric.value for metric in request.configs.keys()]
        metrics = metric_registry.get_metrics(metric_names)

        configured_metric_names = []
        failed_metrics = {}

        # Configure each metric instance with its specific config
        for metric_type, config in request.configs.items():
            metric_name = metric_type.value
            try:
                if metric_name in metrics:
                    metrics[metric_name].configure(config)
                    configured_metric_names.append(metric_name)
                else:
                    failed_metrics[metric_name] = (
                        f"Metric '{metric_name}' not found in registry"
                    )
            except NotImplementedError as e:
                failed_metrics[metric_name] = str(e)
            except Exception as e:
                failed_metrics[metric_name] = f"Configuration failed: {str(e)}"

        success = len(configured_metric_names) > 0

        if success and not failed_metrics:
            message = f"Successfully configured {len(configured_metric_names)} metrics"
        elif success and failed_metrics:
            message = (
                f"Configured {len(configured_metric_names)} metrics, "
                f"{len(failed_metrics)} failed"
            )
        else:
            message = "Failed to configure any metrics"

        return MetricsConfigResponse(
            success=success,
            message=message,
            configured_metrics=(
                configured_metric_names if configured_metric_names else None
            ),
            failed_metrics=failed_metrics if failed_metrics else None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while configuring metrics: {str(e)}",
        ) from e
