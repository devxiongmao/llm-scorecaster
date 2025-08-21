import pytest
from unittest.mock import Mock

from src.core.metrics.base import BaseMetric
from src.core.metrics.metric_observer import MetricObserver
from src.models.schemas import MetricType, TextPair, MetricResult


class ConcreteMetric(BaseMetric):
    """Concrete implementation of BaseMetric for testing."""

    def __init__(
        self, name: str = "test_metric", metric_type: MetricType = MetricType.BERT_SCORE
    ):
        super().__init__()
        self._name = name
        self._metric_type = metric_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def metric_type(self) -> MetricType:
        return self._metric_type

    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        # Simple mock implementation - return score based on string length similarity
        score = 1.0 if len(reference) == len(candidate) else 0.5
        details = {
            "reference_length": len(reference),
            "candidate_length": len(candidate),
        }
        return MetricResult(metric_name=self.name, score=score, details=details)


class FailingMetric(BaseMetric):
    """Concrete implementation that fails for testing error handling."""

    @property
    def name(self) -> str:
        return "failing_metric"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.BLEU

    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        raise ValueError("Test error")


class DetailedMetric(BaseMetric):
    """Metric that returns detailed results for testing."""

    @property
    def name(self) -> str:
        return "detailed_metric"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.ROUGE_L

    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        score = 0.75
        details = {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "word_overlap": 5,
            "reference_tokens": reference.split(),
            "candidate_tokens": candidate.split(),
        }
        return MetricResult(metric_name=self.name, score=score, details=details)


class TestBaseMetric:
    """Test suite for BaseMetric abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric()  # type: ignore

    def test_concrete_implementation_properties(self):
        """Test that concrete implementations work correctly."""
        metric = ConcreteMetric("my_metric", MetricType.ROUGE_1)

        assert metric.name == "my_metric"
        assert metric.metric_type == MetricType.ROUGE_1
        assert metric.description == "my_metric evaluation metric"
        assert metric.requires_model_download is False

    def test_default_description(self):
        """Test that default description uses metric name."""
        metric = ConcreteMetric("custom_name")
        assert metric.description == "custom_name evaluation metric"

    def test_compute_single_basic(self):
        """Test single computation works correctly."""
        metric = ConcreteMetric()
        result = metric.compute_single("hello", "world")

        assert isinstance(result, MetricResult)
        assert result.metric_name == "test_metric"
        assert isinstance(result.score, float)
        assert result.error is None
        assert result.details is not None
        assert "reference_length" in result.details
        assert "candidate_length" in result.details

    def test_compute_single_with_details(self):
        """Test that compute_single can return detailed results."""
        metric = DetailedMetric()
        result = metric.compute_single("hello world", "hello there")

        assert result.metric_name == "detailed_metric"
        assert result.score == 0.75
        assert result.details is not None
        assert result.details["precision"] == 0.8
        assert result.details["recall"] == 0.7
        assert result.details["f1"] == 0.75
        assert len(result.details["reference_tokens"]) == 2
        assert len(result.details["candidate_tokens"]) == 2

    def test_metric_result_validation(self):
        """Test that MetricResult validates correctly with Pydantic."""
        # Valid MetricResult
        result = MetricResult(metric_name="test", score=0.85)
        assert result.metric_name == "test"
        assert result.score == 0.85
        assert result.details is None
        assert result.error is None

        # MetricResult with details
        details = {"precision": 0.9, "recall": 0.8}
        result_with_details = MetricResult(
            metric_name="detailed_test", score=0.85, details=details
        )
        assert result_with_details.details == details

    def test_text_pair_validation(self):
        """Test TextPair Pydantic model validation."""
        # Valid TextPair
        pair = TextPair(reference="reference text", candidate="candidate text")
        assert pair.reference == "reference text"
        assert pair.candidate == "candidate text"

        # Test that empty strings are allowed
        empty_pair = TextPair(reference="", candidate="")
        assert empty_pair.reference == ""
        assert empty_pair.candidate == ""

    def test_compute_batch_empty_list(self):
        """Test batch computation with empty input."""
        metric = ConcreteMetric()
        results = metric.compute_batch([])

        assert results == []

    def test_compute_batch_single_pair(self):
        """Test batch computation with single pair."""
        metric = ConcreteMetric()
        text_pairs = [TextPair(reference="hello", candidate="world")]
        results = metric.compute_batch(text_pairs)

        assert len(results) == 1
        assert results[0].metric_name == "test_metric"
        assert isinstance(results[0].score, float)
        assert results[0].details is not None

    def test_compute_batch_multiple_pairs(self):
        """Test batch computation with multiple pairs."""
        metric = ConcreteMetric()
        text_pairs = [
            TextPair(reference="hello", candidate="world"),
            TextPair(reference="foo", candidate="bar"),
            TextPair(reference="test", candidate="case"),
        ]
        results = metric.compute_batch(text_pairs)

        assert len(results) == 3
        for result in results:
            assert result.metric_name == "test_metric"
            assert isinstance(result.score, float)
            assert result.details is not None

    def test_compute_batch_with_error(self):
        """Test batch computation handles individual pair errors."""
        metric = FailingMetric()
        text_pairs = [
            TextPair(reference="hello", candidate="world"),
            TextPair(reference="foo", candidate="bar"),
        ]
        results = metric.compute_batch(text_pairs)

        assert len(results) == 2
        for result in results:
            assert result.metric_name == "failing_metric"
            assert result.score == 0.0
            assert result.error == "Test error"
            assert result.details is None

    def test_observer_management(self):
        """Test adding and removing observers."""
        metric = ConcreteMetric()
        observer1 = Mock(spec=MetricObserver)
        observer2 = Mock(spec=MetricObserver)

        # Test adding observers
        metric.add_observer(observer1)
        metric.add_observer(observer2)
        assert len(metric._observers) == 2

        # Test removing observer
        metric.remove_observer(observer1)
        assert len(metric._observers) == 1
        assert observer2 in metric._observers

        # Test removing non-existent observer (should not raise error)
        metric.remove_observer(observer1)
        assert len(metric._observers) == 1

    def test_observer_notifications_successful_batch(self):
        """Test that observers are notified correctly during successful batch computation."""
        metric = ConcreteMetric()
        observer = Mock(spec=MetricObserver)
        metric.add_observer(observer)

        text_pairs = [
            TextPair(reference="hello", candidate="world"),
            TextPair(reference="foo", candidate="bar"),
        ]

        results = metric.compute_batch(text_pairs)

        # Verify observer method calls
        observer.on_metric_start.assert_called_once_with("test_metric", 2)
        assert observer.on_pair_processed.call_count == 2
        observer.on_metric_complete.assert_called_once_with("test_metric", results)
        observer.on_metric_error.assert_not_called()

    def test_observer_notifications_with_pair_errors(self):
        """Test observer notifications when individual pairs fail."""
        metric = FailingMetric()
        observer = Mock(spec=MetricObserver)
        metric.add_observer(observer)

        text_pairs = [TextPair(reference="hello", candidate="world")]
        results = metric.compute_batch(text_pairs)

        # Should still notify start and completion, even with pair errors
        observer.on_metric_start.assert_called_once_with("failing_metric", 1)
        observer.on_pair_processed.assert_called_once()
        observer.on_metric_complete.assert_called_once_with("failing_metric", results)
        observer.on_metric_error.assert_not_called()

    def test_observer_error_during_notification(self):
        """Test that observer errors don't break metric computation."""
        metric = ConcreteMetric()

        # Create observer that raises exception
        faulty_observer = Mock(spec=MetricObserver)
        faulty_observer.on_metric_start.side_effect = Exception("Observer error")

        good_observer = Mock(spec=MetricObserver)

        metric.add_observer(faulty_observer)
        metric.add_observer(good_observer)

        text_pairs = [TextPair(reference="hello", candidate="world")]
        results = metric.compute_batch(text_pairs)

        # Should complete successfully despite observer error
        assert len(results) == 1
        assert results[0].score is not None

        # Good observer should still be notified
        good_observer.on_metric_start.assert_called_once()

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        metric = ConcreteMetric("string_test")

        str_repr = str(metric)
        assert "ConcreteMetric" in str_repr
        assert "string_test" in str_repr

        repr_repr = repr(metric)
        assert repr_repr == str_repr

    def test_batch_size_parameter(self):
        """Test that batch_size parameter is accepted (even if not used in default implementation)."""
        metric = ConcreteMetric()
        text_pairs = [TextPair(reference="hello", candidate="world")]

        # Should work with any batch_size value
        results = metric.compute_batch(text_pairs, batch_size=10)
        assert len(results) == 1

    def test_metric_result_properties(self):
        """Test that metric results have correct properties."""
        metric = ConcreteMetric("property_test")
        result = metric.compute_single("test", "test")  # Same length = score 1.0

        assert result.metric_name == "property_test"
        assert result.score == 1.0
        assert result.error is None
        assert result.details is not None


class TestMetricSubclassRequirements:
    """Test that subclasses must implement required abstract methods."""

    def test_missing_name_property(self):
        """Test that subclass must implement name property."""
        with pytest.raises(TypeError):

            class IncompleteMetric1(BaseMetric):
                @property
                def metric_type(self) -> MetricType:
                    return MetricType.BERT_SCORE

                def compute_single(
                    self, reference: str, candidate: str
                ) -> MetricResult:
                    return MetricResult(metric_name="test", score=0.0)

            IncompleteMetric1()  # type: ignore

    def test_missing_metric_type_property(self):
        """Test that subclass must implement metric_type property."""
        with pytest.raises(TypeError):

            class IncompleteMetric2(BaseMetric):
                @property
                def name(self) -> str:
                    return "test"

                def compute_single(
                    self, reference: str, candidate: str
                ) -> MetricResult:
                    return MetricResult(metric_name="test", score=0.0)

            IncompleteMetric2()  # type: ignore

    def test_missing_compute_single_method(self):
        """Test that subclass must implement compute_single method."""
        with pytest.raises(TypeError):

            class IncompleteMetric3(BaseMetric):
                @property
                def name(self) -> str:
                    return "test"

                @property
                def metric_type(self) -> MetricType:
                    return MetricType.BERT_SCORE

            IncompleteMetric3()  # type: ignore


class TestMetricTypes:
    """Test the MetricType enum values."""

    def test_metric_type_values(self):
        """Test that all metric types are correctly defined."""
        expected_values = {
            "bert_score",
            "bleu",
            "rouge_l",
            "rouge_1",
            "rouge_2",
            "align_score",
        }
        actual_values = {mt.value for mt in MetricType}
        assert actual_values == expected_values

    def test_metric_type_usage_in_concrete_implementations(self):
        """Test that different metric types can be used."""
        bert_metric = ConcreteMetric("bert_test", MetricType.BERT_SCORE)
        bleu_metric = ConcreteMetric("bleu_test", MetricType.BLEU)
        rouge_metric = ConcreteMetric("rouge_test", MetricType.ROUGE_L)

        assert bert_metric.metric_type == MetricType.BERT_SCORE
        assert bleu_metric.metric_type == MetricType.BLEU
        assert rouge_metric.metric_type == MetricType.ROUGE_L


# Additional fixtures and parameterized tests


@pytest.fixture
def sample_metric():
    """Fixture providing a sample metric instance."""
    return ConcreteMetric("sample", MetricType.ROUGE_1)


@pytest.fixture
def sample_text_pairs():
    """Fixture providing sample text pairs."""
    return [
        TextPair(
            reference="The quick brown fox jumps over the lazy dog",
            candidate="The fast brown fox leaps over the sleepy dog",
        ),
        TextPair(reference="Hello world", candidate="Hi world"),
        TextPair(
            reference="Python testing is important",
            candidate="Python testing is important",
        ),  # Exact match
    ]


@pytest.fixture
def complex_text_pairs():
    """Fixture providing more complex text pairs for testing."""
    return [
        TextPair(reference="", candidate=""),  # Empty strings
        TextPair(reference="Single word", candidate="Different"),
        TextPair(
            reference="Multi-line text\nwith line breaks",
            candidate="Multi-line text\nwith different breaks",
        ),
        TextPair(
            reference="Special chars: !@#$%^&*()", candidate="Special chars: !@#$%^&*()"
        ),
        TextPair(
            reference="Numbers 123 and symbols", candidate="Numbers 456 and symbols"
        ),
    ]


@pytest.mark.parametrize(
    "metric_name,metric_type,expected_description",
    [
        ("bert_accuracy", MetricType.BERT_SCORE, "bert_accuracy evaluation metric"),
        ("bleu_4", MetricType.BLEU, "bleu_4 evaluation metric"),
        ("rouge_score", MetricType.ROUGE_L, "rouge_score evaluation metric"),
    ],
)
def test_metric_descriptions(metric_name, metric_type, expected_description):
    """Test that metric descriptions are generated correctly for different types."""
    metric = ConcreteMetric(metric_name, metric_type)
    assert metric.description == expected_description
    assert metric.metric_type == metric_type


def test_compute_batch_with_fixture(sample_metric, sample_text_pairs):
    """Test batch computation using fixtures."""
    results = sample_metric.compute_batch(sample_text_pairs)

    assert len(results) == 3
    assert all(r.metric_name == "sample" for r in results)
    assert all(isinstance(r.score, float) for r in results)
    assert all(r.details is not None for r in results)

    # Last pair should have score 1.0 (exact length match)
    assert results[2].score == 1.0


def test_compute_batch_with_complex_pairs(sample_metric, complex_text_pairs):
    """Test batch computation with complex text pairs."""
    results = sample_metric.compute_batch(complex_text_pairs)

    assert len(results) == 5

    # Test empty strings case
    assert results[0].score == 1.0  # Both empty, same length
    assert results[0].details["reference_length"] == 0
    assert results[0].details["candidate_length"] == 0

    # Test special characters case
    assert results[3].score == 1.0  # Same special chars string

    # All results should have proper structure
    for result in results:
        assert result.metric_name == "sample"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert result.details is not None
        assert result.error is None


@pytest.mark.parametrize(
    "reference,candidate,expected_score",
    [
        ("hello", "world", 1.0),  # Same length
        ("hello", "hi", 0.5),  # Different length
        ("", "", 1.0),  # Both empty
        ("test", "testing", 0.5),  # Different length
    ],
)
def test_concrete_metric_scoring_logic(reference, candidate, expected_score):
    """Test the scoring logic of ConcreteMetric."""
    metric = ConcreteMetric()
    result = metric.compute_single(reference, candidate)
    assert result.score == expected_score


def test_metric_result_serialization():
    """Test that MetricResult can be serialized/deserialized properly."""
    original_result = MetricResult(
        metric_name="test_metric",
        score=0.85,
        details={"precision": 0.9, "recall": 0.8},
        error=None,
    )

    # Test dict conversion (useful for JSON serialization)
    result_dict = original_result.model_dump()
    assert result_dict["metric_name"] == "test_metric"
    assert result_dict["score"] == 0.85
    assert result_dict["details"]["precision"] == 0.9

    # Test reconstruction from dict
    reconstructed = MetricResult(**result_dict)
    assert reconstructed == original_result
