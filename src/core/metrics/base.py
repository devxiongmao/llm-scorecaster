from abc import ABC, abstractmethod
from typing import List
from src.core.metrics.metric_observer import MetricObserver
from src.models.schemas import MetricType, TextPair, MetricResult


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.

    This class defines the interface that all metric implementations must follow,
    ensuring consistency across different evaluation metrics.
    """

    def __init__(self):
        self._observers: List[MetricObserver] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name identifier."""

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Return the metric type enum."""

    @property
    def description(self) -> str:
        """Return a description of what this metric measures."""
        return f"{self.name} evaluation metric"

    @property
    def requires_model_download(self) -> bool:
        """Whether this metric requires downloading models/data."""
        return False

    @abstractmethod
    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        """
        Compute the metric for a single text pair.

        Args:
            reference: The reference/ground truth text
            candidate: The candidate text to evaluate

        Returns:
            MetricResult: The computed metric result
        """

    def compute_batch(
        self,
        text_pairs: List[TextPair],
        batch_size: int = 32,  # pylint: disable=unused-argument
    ) -> List[MetricResult]:
        """
        Compute the metric for multiple text pairs.

        Default implementation processes pairs individually. Subclasses can
        override this for more efficient batch processing.

        Args:
            text_pairs: List of text pairs to evaluate
            batch_size: Size of batches for processing (unused in default implementation)

        Returns:
            List[MetricResult]: List of metric results
        """
        results = []

        # Notify observers that computation is starting
        self._notify_start(len(text_pairs))

        try:
            for i, pair in enumerate(text_pairs):
                try:
                    result = self.compute_single(pair.reference, pair.candidate)
                    results.append(result)

                    # Notify observers of progress
                    self._notify_pair_processed(i, result)

                except Exception as e:
                    error_result = MetricResult(
                        metric_name=self.name, score=0.0, error=str(e)
                    )
                    results.append(error_result)
                    self._notify_pair_processed(i, error_result)

            # Notify observers of completion
            self._notify_complete(results)
            return results

        except Exception as e:
            self._notify_error(e)
            raise

    def add_observer(self, observer: MetricObserver) -> None:
        """Add an observer to track metric computation progress."""
        self._observers.append(observer)

    def remove_observer(self, observer: MetricObserver) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_start(self, total_pairs: int) -> None:
        """Notify observers that computation is starting."""
        for observer in self._observers:
            try:
                observer.on_metric_start(self.name, total_pairs)
            except Exception:
                # Don't let observer errors break metric computation
                pass

    def _notify_pair_processed(self, pair_index: int, result: MetricResult) -> None:
        """Notify observers that a pair was processed."""
        for observer in self._observers:
            try:
                observer.on_pair_processed(self.name, pair_index, result)
            except Exception:
                pass

    def _notify_complete(self, results: List[MetricResult]) -> None:
        """Notify observers that computation completed."""
        for observer in self._observers:
            try:
                observer.on_metric_complete(self.name, results)
            except Exception:
                pass

    def _notify_error(self, error: Exception) -> None:
        """Notify observers of computation error."""
        for observer in self._observers:
            try:
                observer.on_metric_error(self.name, error)
            except Exception:
                pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return self.__str__()
