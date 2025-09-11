"""
Simple progress tracker observer.
Outputs print statements at various intervals
"""

from src.core.metrics.base import MetricObserver


class ProgressTracker(MetricObserver):
    """Progress Tracker observer"""

    def on_metric_start(self, metric_name: str, total_pairs: int):
        """Notify on metrics start"""
        print(f"Starting {metric_name} for {total_pairs} pairs")

    def on_pair_processed(self, metric_name: str, pair_index: int, result):
        """Notify on pair processed"""
        print(f"{metric_name}: processed pair {pair_index}")

    def on_metric_complete(self, metric_name: str, results):
        """Notify on metric complete"""
        print(f"{metric_name}: completed with {len(results)} results")

    def on_metric_error(self, metric_name: str, error):
        """Notify on metric error"""
        print(f"{metric_name}: error occurred: {error}")
