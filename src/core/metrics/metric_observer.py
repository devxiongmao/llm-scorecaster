from abc import ABC, abstractmethod
from typing import List

from src.models.schemas import MetricResult


class MetricObserver(ABC):
    """Observer interface for tracking metric computation progress."""

    @abstractmethod
    def on_metric_start(self, metric_name: str, total_pairs: int) -> None:
        """Called when metric computation starts."""

    @abstractmethod
    def on_pair_processed(
        self, metric_name: str, pair_index: int, result: MetricResult
    ) -> None:
        """Called when a text pair is processed."""

    @abstractmethod
    def on_metric_complete(self, metric_name: str, results: List[MetricResult]) -> None:
        """Called when metric computation completes."""

    @abstractmethod
    def on_metric_error(self, metric_name: str, error: Exception) -> None:
        """Called when metric computation fails."""
