import pytest
from typing import List

from src.core.metrics.metric_observer import MetricObserver
from src.models.schemas import MetricResult


class ConcreteObserver(MetricObserver):
    """Concrete implementation of MetricObserver for testing."""

    def __init__(self):
        self.events = []
        self.metric_starts = []
        self.pairs_processed = []
        self.metric_completes = []
        self.metric_errors = []

    def on_metric_start(self, metric_name: str, total_pairs: int) -> None:
        event = {
            "type": "start",
            "metric_name": metric_name,
            "total_pairs": total_pairs,
        }
        self.events.append(event)
        self.metric_starts.append((metric_name, total_pairs))

    def on_pair_processed(
        self, metric_name: str, pair_index: int, result: MetricResult
    ) -> None:
        event = {
            "type": "pair_processed",
            "metric_name": metric_name,
            "pair_index": pair_index,
            "result": result,
        }
        self.events.append(event)
        self.pairs_processed.append((metric_name, pair_index, result))

    def on_metric_complete(self, metric_name: str, results: List[MetricResult]) -> None:
        event = {"type": "complete", "metric_name": metric_name, "results": results}
        self.events.append(event)
        self.metric_completes.append((metric_name, results))

    def on_metric_error(self, metric_name: str, error: Exception) -> None:
        event = {"type": "error", "metric_name": metric_name, "error": error}
        self.events.append(event)
        self.metric_errors.append((metric_name, error))


# Fixtures


@pytest.fixture
def observer():
    """Basic concrete observer for testing."""
    return ConcreteObserver()


@pytest.fixture
def sample_result():
    """Single metric result for testing."""
    return MetricResult(metric_name="test", score=0.85)


@pytest.fixture
def sample_results():
    """List of metric results for testing."""
    return [
        MetricResult(metric_name="test", score=0.85, details={"precision": 0.9}),
        MetricResult(metric_name="test", score=0.75, details={"precision": 0.8}),
        MetricResult(metric_name="test", score=0.95, details={"precision": 0.95}),
    ]


@pytest.fixture
def test_error():
    """Sample exception for testing."""
    return ValueError("Test error message")


# Abstract class tests


def test_cannot_instantiate_abstract_class():
    """MetricObserver cannot be instantiated directly."""
    with pytest.raises(TypeError):
        MetricObserver()  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated


def test_concrete_implementation_works(observer):
    """Concrete implementations can be instantiated and used."""
    observer.on_metric_start("test_metric", 5)
    assert len(observer.events) == 1
    assert observer.events[0]["type"] == "start"
    assert observer.events[0]["metric_name"] == "test_metric"
    assert observer.events[0]["total_pairs"] == 5


# Method-specific tests


def test_on_metric_start(observer):
    """on_metric_start records metric initialization."""
    observer.on_metric_start("bert_score", 10)

    assert len(observer.metric_starts) == 1
    assert observer.metric_starts[0] == ("bert_score", 10)

    # Test with different values
    observer.on_metric_start("bleu_score", 0)
    assert len(observer.metric_starts) == 2
    assert observer.metric_starts[1] == ("bleu_score", 0)


def test_on_pair_processed(observer, sample_result):
    """on_pair_processed records individual pair processing."""
    observer.on_pair_processed("rouge", 3, sample_result)

    assert len(observer.pairs_processed) == 1
    metric_name, pair_index, stored_result = observer.pairs_processed[0]
    assert metric_name == "rouge"
    assert pair_index == 3
    assert stored_result == sample_result


def test_on_metric_complete(observer, sample_results):
    """on_metric_complete records completion with results."""
    observer.on_metric_complete("bert_score", sample_results)

    assert len(observer.metric_completes) == 1
    metric_name, stored_results = observer.metric_completes[0]
    assert metric_name == "bert_score"
    assert stored_results == sample_results
    assert len(stored_results) == 3


def test_on_metric_error(observer, test_error):
    """on_metric_error records error conditions."""
    observer.on_metric_error("rouge", test_error)

    assert len(observer.metric_errors) == 1
    metric_name, stored_error = observer.metric_errors[0]
    assert metric_name == "rouge"
    assert stored_error == test_error


# Workflow tests


def test_event_ordering(observer):
    """Events are recorded in the correct order."""
    # Simulate a typical metric computation sequence
    observer.on_metric_start("test_metric", 2)

    result1 = MetricResult(metric_name="test_metric", score=0.8)
    observer.on_pair_processed("test_metric", 0, result1)

    result2 = MetricResult(metric_name="test_metric", score=0.9)
    observer.on_pair_processed("test_metric", 1, result2)

    observer.on_metric_complete("test_metric", [result1, result2])

    # Check event order
    assert len(observer.events) == 4
    assert observer.events[0]["type"] == "start"
    assert observer.events[1]["type"] == "pair_processed"
    assert observer.events[1]["pair_index"] == 0
    assert observer.events[2]["type"] == "pair_processed"
    assert observer.events[2]["pair_index"] == 1
    assert observer.events[3]["type"] == "complete"


def test_multiple_metrics_tracking(observer):
    """Same observer can track multiple metrics."""
    # First metric
    observer.on_metric_start("bert_score", 1)
    result1 = MetricResult(metric_name="bert_score", score=0.8)
    observer.on_pair_processed("bert_score", 0, result1)
    observer.on_metric_complete("bert_score", [result1])

    # Second metric
    observer.on_metric_start("bleu_score", 1)
    result2 = MetricResult(metric_name="bleu_score", score=0.7)
    observer.on_pair_processed("bleu_score", 0, result2)
    observer.on_metric_complete("bleu_score", [result2])

    # Check that both metrics were tracked
    assert len(observer.metric_starts) == 2
    assert observer.metric_starts[0][0] == "bert_score"
    assert observer.metric_starts[1][0] == "bleu_score"

    assert len(observer.pairs_processed) == 2
    assert observer.pairs_processed[0][0] == "bert_score"
    assert observer.pairs_processed[1][0] == "bleu_score"


def test_error_handling_workflow(observer):
    """Error handling workflow works correctly."""
    observer.on_metric_start("failing_metric", 1)

    # Simulate an error occurring
    test_error = RuntimeError("Computation failed")
    observer.on_metric_error("failing_metric", test_error)

    assert len(observer.metric_starts) == 1
    assert len(observer.metric_errors) == 1
    assert observer.metric_errors[0][1] == test_error


# Abstract method requirement tests


def test_missing_on_metric_start():
    """Subclass must implement on_metric_start."""
    with pytest.raises(TypeError):

        class IncompleteObserver1(MetricObserver):
            def on_pair_processed(
                self, metric_name: str, pair_index: int, result: MetricResult
            ) -> None:
                pass

            def on_metric_complete(
                self, metric_name: str, results: List[MetricResult]
            ) -> None:
                pass

            def on_metric_error(self, metric_name: str, error: Exception) -> None:
                pass

        IncompleteObserver1()  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated


def test_missing_on_pair_processed():
    """Subclass must implement on_pair_processed."""
    with pytest.raises(TypeError):

        class IncompleteObserver2(MetricObserver):
            def on_metric_start(self, metric_name: str, total_pairs: int) -> None:
                pass

            def on_metric_complete(
                self, metric_name: str, results: List[MetricResult]
            ) -> None:
                pass

            def on_metric_error(self, metric_name: str, error: Exception) -> None:
                pass

        IncompleteObserver2()  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated


def test_missing_on_metric_complete():
    """Subclass must implement on_metric_complete."""
    with pytest.raises(TypeError):

        class IncompleteObserver3(MetricObserver):
            def on_metric_start(self, metric_name: str, total_pairs: int) -> None:
                pass

            def on_pair_processed(
                self, metric_name: str, pair_index: int, result: MetricResult
            ) -> None:
                pass

            def on_metric_error(self, metric_name: str, error: Exception) -> None:
                pass

        IncompleteObserver3()  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated


def test_missing_on_metric_error():
    """Subclass must implement on_metric_error."""
    with pytest.raises(TypeError):

        class IncompleteObserver4(MetricObserver):
            def on_metric_start(self, metric_name: str, total_pairs: int) -> None:
                pass

            def on_pair_processed(
                self, metric_name: str, pair_index: int, result: MetricResult
            ) -> None:
                pass

            def on_metric_complete(
                self, metric_name: str, results: List[MetricResult]
            ) -> None:
                pass

        IncompleteObserver4()  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated


def test_observer_state_isolation():
    """Multiple observer instances maintain separate state."""
    observer1 = ConcreteObserver()
    observer2 = ConcreteObserver()

    observer1.on_metric_start("metric1", 5)
    observer2.on_metric_start("metric2", 10)

    assert len(observer1.metric_starts) == 1
    assert len(observer2.metric_starts) == 1
    assert observer1.metric_starts[0][0] == "metric1"
    assert observer2.metric_starts[0][0] == "metric2"


# Parameterized tests


@pytest.mark.parametrize(
    "metric_name,total_pairs",
    [
        ("bert_score", 1),
        ("bleu_score", 10),
        ("rouge", 100),
        ("rouge", 0),  # Edge case: zero pairs
    ],
)
def test_on_metric_start_parameterized(observer, metric_name, total_pairs):
    """on_metric_start works with different parameters."""
    observer.on_metric_start(metric_name, total_pairs)

    assert len(observer.metric_starts) == 1
    assert observer.metric_starts[0] == (metric_name, total_pairs)


@pytest.mark.parametrize(
    "pair_index,score",
    [
        (0, 0.0),  # First pair, minimum score
        (5, 0.5),  # Middle pair, medium score
        (99, 1.0),  # High index, maximum score
    ],
)
def test_on_pair_processed_parameterized(observer, pair_index, score):
    """on_pair_processed works with different parameters."""
    result = MetricResult(metric_name="param_test", score=score)
    observer.on_pair_processed("param_test", pair_index, result)

    assert len(observer.pairs_processed) == 1
    metric_name, stored_index, stored_result = observer.pairs_processed[0]
    assert metric_name == "param_test"
    assert stored_index == pair_index
    assert stored_result.score == score


@pytest.mark.parametrize(
    "error_type,error_message",
    [
        (ValueError, "Invalid value"),
        (RuntimeError, "Runtime failure"),
        (TypeError, "Type mismatch"),
        (Exception, "Generic exception"),
    ],
)
def test_on_metric_error_different_exceptions(observer, error_type, error_message):
    """on_metric_error handles different exception types."""
    error = error_type(error_message)
    observer.on_metric_error("error_test", error)

    assert len(observer.metric_errors) == 1
    metric_name, stored_error = observer.metric_errors[0]
    assert metric_name == "error_test"
    assert isinstance(stored_error, error_type)
    assert str(stored_error) == error_message


def test_observer_with_complex_results(observer, sample_results):
    """Observer handles complex results with details."""
    observer.on_metric_complete("complex_test", sample_results)

    assert len(observer.metric_completes) == 1
    metric_name, results = observer.metric_completes[0]
    assert metric_name == "complex_test"
    assert len(results) == 3
    assert all(r.details is not None for r in results)
    assert all("precision" in r.details for r in results)
