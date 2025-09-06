"""Core computation logic for metrics processing."""

from typing import List

from src.models.schemas import MetricsRequest, TextPair, TextPairResult
from src.core.metrics.registry import metric_registry


def compute_metrics_core(request: MetricsRequest) -> List[TextPairResult]:
    """
    Core metrics computation logic shared between sync and async processing.

    This function contains the main business logic for computing metrics
    on text pairs, isolated from API/task-specific concerns.

    Args:
        request: The MetricsRequest object containing text pairs and requested metrics

    Returns:
        List of TextPairResult objects with computed metrics
    """
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
        for _, metric_instance in metrics.items():
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
