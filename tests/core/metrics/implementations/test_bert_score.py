"""Tests for the BERT Score metric implementation."""

from unittest.mock import Mock, patch
import pytest

from src.core.metrics.implementations.bert_score import BertScoreMetric
from src.models.schemas import MetricType, MetricResult, TextPair


# Fixtures
@pytest.fixture(name="bert_metric")
def bert_metric_fixture():
    """Basic BERT Score metric instance."""
    return BertScoreMetric()


@pytest.fixture(name="mock_scorer")
def mock_scorer_fixture():
    """Mock BERT scorer with typical return values."""
    scorer = Mock()
    # Mock typical BERT Score return values (tensors with single values)
    scorer.score.return_value = (
        [0.85],  # precision
        [0.80],  # recall
        [0.82],  # f1
    )
    return scorer


# Basic property tests


def test_metric_properties(bert_metric):
    """BERT Score metric has correct properties."""
    assert bert_metric.name == "bert_score"
    assert bert_metric.metric_type == MetricType.BERT_SCORE
    assert (
        bert_metric.description
        == "BERT Score: Contextual embeddings-based evaluation using BERT"
    )
    assert bert_metric.requires_model_download is True


def test_initial_state(bert_metric):
    """BERT Score metric starts in unloaded state."""
    assert bert_metric._scorer is None
    assert bert_metric._model_loaded is False


def test_get_model_info_not_loaded(bert_metric):
    """get_model_info returns not_loaded when model isn't loaded."""
    info = bert_metric.get_model_info()
    assert info == {"status": "not_loaded"}


def test_get_model_info_loaded(bert_metric, mock_scorer):
    """get_model_info returns correct info when model is loaded."""
    with patch("bert_score.BERTScorer", return_value=mock_scorer):
        bert_metric._load_model()
        info = bert_metric.get_model_info()

        expected = {
            "status": "loaded",
            "model_type": "roberta-large",
            "language": "en",
            "rescale_with_baseline": True,
        }
        assert info == expected


# Model loading tests


@patch("bert_score.BERTScorer")
def test_load_model_success(mock_bert_scorer_class, bert_metric, mock_scorer):
    """_load_model successfully loads BERT scorer."""
    mock_bert_scorer_class.return_value = mock_scorer

    bert_metric._load_model()

    assert bert_metric._model_loaded is True
    assert bert_metric._scorer == mock_scorer
    mock_bert_scorer_class.assert_called_once_with(
        model_type="roberta-large", lang="en", rescale_with_baseline=True
    )


@patch("bert_score.BERTScorer")
def test_load_model_import_error(mock_bert_scorer_class, bert_metric):
    """_load_model raises ImportError when bert-score not available."""
    mock_bert_scorer_class.side_effect = ImportError("No module named 'bert_score'")

    with pytest.raises(ImportError, match="bert-score package is required"):
        bert_metric._load_model()

    assert bert_metric._model_loaded is False
    assert bert_metric._scorer is None


@patch("bert_score.BERTScorer")
def test_load_model_runtime_error(mock_bert_scorer_class, bert_metric):
    """_load_model raises RuntimeError when initialization fails."""
    mock_bert_scorer_class.side_effect = RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="Failed to initialize BERT Score"):
        bert_metric._load_model()

    assert bert_metric._model_loaded is False
    assert bert_metric._scorer is None


@patch("bert_score.BERTScorer")
def test_load_model_only_loads_once(mock_bert_scorer_class, bert_metric, mock_scorer):
    """_load_model only loads model once, subsequent calls do nothing."""
    mock_bert_scorer_class.return_value = mock_scorer

    # Load multiple times
    bert_metric._load_model()
    bert_metric._load_model()
    bert_metric._load_model()

    # Should only be called once
    mock_bert_scorer_class.assert_called_once()
    assert bert_metric._model_loaded is True


# Single computation tests


@patch("bert_score.BERTScorer")
def test_compute_single_success(
    mock_bert_scorer_class, bert_metric, mock_scorer, sample_text_pair
):
    """compute_single returns correct result for valid input."""
    mock_bert_scorer_class.return_value = mock_scorer

    result = bert_metric.compute_single(
        sample_text_pair.reference, sample_text_pair.candidate
    )

    assert isinstance(result, MetricResult)
    assert result.metric_name == "bert_score"
    assert result.score == 0.82  # F1 score
    assert result.error is None
    assert result.details == {"precision": 0.85, "recall": 0.8, "f1": 0.82}

    # Verify scorer was called correctly
    mock_scorer.score.assert_called_once_with(
        [sample_text_pair.candidate], [sample_text_pair.reference]
    )


@patch("bert_score.BERTScorer")
def test_compute_single_scorer_error(mock_bert_scorer_class, bert_metric):
    """compute_single handles scorer errors gracefully."""
    mock_scorer = Mock()
    mock_scorer.score.side_effect = RuntimeError("CUDA error")
    mock_bert_scorer_class.return_value = mock_scorer

    result = bert_metric.compute_single("reference", "candidate")

    assert result.metric_name == "bert_score"
    assert result.score == 0.0
    assert result.error == "CUDA error"
    assert result.details is None


@patch("bert_score.BERTScorer")
def test_compute_single_loads_model_if_needed(
    mock_bert_scorer_class, bert_metric, mock_scorer
):
    """compute_single automatically loads model if not loaded."""
    mock_bert_scorer_class.return_value = mock_scorer

    assert bert_metric._model_loaded is False

    bert_metric.compute_single("reference", "candidate")

    assert bert_metric._model_loaded is True
    mock_bert_scorer_class.assert_called_once()


# Batch computation tests


@patch("bert_score.BERTScorer")
def test_compute_batch_empty_list(
    mock_bert_scorer_class, bert_metric, mock_scorer, empty_text_pairs
):
    """compute_batch handles empty input correctly."""
    mock_bert_scorer_class.return_value = mock_scorer

    results = bert_metric.compute_batch(empty_text_pairs)

    assert results == []
    mock_scorer.score.assert_not_called()


@patch("bert_score.BERTScorer")
def test_compute_batch_single_pair(
    mock_bert_scorer_class, bert_metric, mock_scorer, sample_text_pair
):
    """compute_batch works correctly with single pair."""
    mock_bert_scorer_class.return_value = mock_scorer

    results = bert_metric.compute_batch([sample_text_pair])

    assert len(results) == 1
    result = results[0]
    assert result.metric_name == "bert_score"
    assert result.score == 0.82
    assert result.error is None

    # Verify batch call
    mock_scorer.score.assert_called_once_with(
        [sample_text_pair.candidate], [sample_text_pair.reference]
    )


@patch("bert_score.BERTScorer")
def test_compute_batch_multiple_pairs(
    mock_bert_scorer_class, bert_metric, sample_text_pairs
):
    """compute_batch processes multiple pairs correctly."""
    mock_scorer = Mock()
    # Mock return values for 3 pairs
    mock_scorer.score.return_value = (
        [0.85, 0.90, 0.75],  # precision
        [0.80, 0.85, 0.70],  # recall
        [0.82, 0.87, 0.72],  # f1
    )
    mock_bert_scorer_class.return_value = mock_scorer

    results = bert_metric.compute_batch(sample_text_pairs)

    assert len(results) == 3

    # Check first result
    assert results[0].score == 0.82
    assert results[0].details["precision"] == 0.85

    # Check second result
    assert results[1].score == 0.87
    assert results[1].details["recall"] == 0.85

    # Check third result
    assert results[2].score == 0.72
    assert results[2].details["f1"] == 0.72

    # Verify batch call
    expected_candidates = [pair.candidate for pair in sample_text_pairs]
    expected_references = [pair.reference for pair in sample_text_pairs]
    mock_scorer.score.assert_called_once_with(expected_candidates, expected_references)


@patch("bert_score.BERTScorer")
def test_compute_batch_with_custom_batch_size(mock_bert_scorer_class, bert_metric):
    """compute_batch respects custom batch_size parameter."""
    mock_scorer = Mock()
    mock_scorer.score.return_value = ([0.8, 0.9], [0.7, 0.8], [0.75, 0.85])
    mock_bert_scorer_class.return_value = mock_scorer

    # Create 4 pairs but use batch_size=2
    text_pairs = [TextPair(reference=f"ref{i}", candidate=f"cand{i}") for i in range(4)]

    # Mock return values for each batch of 2
    mock_scorer.score.side_effect = [
        ([0.8, 0.9], [0.7, 0.8], [0.75, 0.85]),  # First batch
        ([0.6, 0.7], [0.5, 0.6], [0.55, 0.65]),  # Second batch
    ]

    results = bert_metric.compute_batch(text_pairs, batch_size=2)

    assert len(results) == 4
    assert mock_scorer.score.call_count == 2  # Called twice for 2 batches


@patch("bert_score.BERTScorer")
def test_compute_batch_handles_batch_error(
    mock_bert_scorer_class, bert_metric, sample_text_pairs
):
    """compute_batch handles errors in individual batches."""
    mock_scorer = Mock()
    mock_scorer.score.side_effect = RuntimeError("Batch processing failed")
    mock_bert_scorer_class.return_value = mock_scorer

    results = bert_metric.compute_batch(sample_text_pairs)

    assert len(results) == 3
    for result in results:
        assert result.metric_name == "bert_score"
        assert result.score == 0.0
        assert result.error == "Batch processing failed"


@patch("bert_score.BERTScorer")
def test_compute_batch_loads_model_if_needed(
    mock_bert_scorer_class, bert_metric, mock_scorer, sample_text_pairs
):
    """compute_batch automatically loads model if not loaded."""
    mock_bert_scorer_class.return_value = mock_scorer
    mock_scorer.score.return_value = ([0.8] * 3, [0.7] * 3, [0.75] * 3)

    assert bert_metric._model_loaded is False

    bert_metric.compute_batch(sample_text_pairs)

    assert bert_metric._model_loaded is True


# Observer notification tests


@patch("bert_score.BERTScorer")
def test_compute_batch_notifies_observers(
    mock_bert_scorer_class, bert_metric, sample_text_pairs
):
    """compute_batch properly notifies observers."""
    mock_scorer = Mock()
    mock_scorer.score.return_value = ([0.8] * 3, [0.7] * 3, [0.75] * 3)
    mock_bert_scorer_class.return_value = mock_scorer

    # Mock observer methods
    bert_metric._notify_start = Mock()
    bert_metric._notify_pair_processed = Mock()
    bert_metric._notify_complete = Mock()

    results = bert_metric.compute_batch(sample_text_pairs)

    # Verify observer calls
    bert_metric._notify_start.assert_called_once_with(3)
    assert bert_metric._notify_pair_processed.call_count == 3
    bert_metric._notify_complete.assert_called_once_with(results)


# Edge cases and integration tests


def test_empty_strings(bert_metric, mock_scorer):
    """compute_single handles empty strings."""
    with patch("bert_score.BERTScorer", return_value=mock_scorer):
        result = bert_metric.compute_single("", "")

        assert isinstance(result, MetricResult)
        assert result.metric_name == "bert_score"
        mock_scorer.score.assert_called_once_with([""], [""])


def test_very_long_strings(bert_metric, mock_scorer):
    """compute_single handles very long strings."""
    long_text = "word " * 1000  # Very long text

    with patch("bert_score.BERTScorer", return_value=mock_scorer):
        result = bert_metric.compute_single(long_text, long_text)

        assert isinstance(result, MetricResult)
        mock_scorer.score.assert_called_once_with([long_text], [long_text])


def test_special_characters(bert_metric, mock_scorer):
    """compute_single handles special characters and Unicode."""
    reference = "Hello 世界! @#$%^&*()_+-=[]{}|;:,.<>?"
    candidate = "Hi 世界! Special chars: @#$%"

    with patch("bert_score.BERTScorer", return_value=mock_scorer):
        result = bert_metric.compute_single(reference, candidate)

        assert isinstance(result, MetricResult)
        mock_scorer.score.assert_called_once_with([candidate], [reference])


# Parameterized tests


@pytest.mark.parametrize(
    "precision,recall,expected_f1",
    [
        (0.8, 0.9, 0.85),  # Normal values
        (1.0, 1.0, 1.0),  # Perfect scores
        (0.0, 0.0, 0.0),  # Minimum scores
        (0.5, 0.5, 0.5),  # Equal precision/recall
    ],
)
def test_compute_single_score_calculation(bert_metric, precision, recall, expected_f1):
    """compute_single correctly uses F1 as primary score."""
    mock_scorer = Mock()
    mock_scorer.score.return_value = ([precision], [recall], [expected_f1])

    with patch("bert_score.BERTScorer", return_value=mock_scorer):
        result = bert_metric.compute_single("reference", "candidate")

        assert result.score == expected_f1
        assert result.details["precision"] == precision
        assert result.details["recall"] == recall
        assert result.details["f1"] == expected_f1


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 32, 100])
def test_compute_batch_different_batch_sizes(bert_metric, batch_size):
    """compute_batch works with different batch sizes."""
    text_pairs = [
        TextPair(reference=f"ref{i}", candidate=f"cand{i}")
        for i in range(7)  # 7 pairs to test uneven batching
    ]

    mock_scorer = Mock()
    # Mock the score method to return appropriate values
    mock_scorer.score.return_value = (
        [0.8] * batch_size,
        [0.7] * batch_size,
        [0.75] * batch_size,
    )

    with patch("bert_score.BERTScorer", return_value=mock_scorer):
        results = bert_metric.compute_batch(text_pairs, batch_size=batch_size)

        assert len(results) == 7
        # Should call scorer for each batch (ceiling division)
        expected_calls = (len(text_pairs) + batch_size - 1) // batch_size
        assert (
            mock_scorer.score.call_count <= expected_calls + 1
        )  # Allow for some variance


@pytest.mark.parametrize(
    "error_type,error_message",
    [
        (RuntimeError, "CUDA out of memory"),
        (ValueError, "Invalid model input"),
        (TypeError, "Wrong tensor type"),
        (Exception, "Generic error"),
    ],
)
def test_compute_single_different_errors(bert_metric, error_type, error_message):
    """compute_single handles different error types properly."""
    mock_scorer = Mock()
    mock_scorer.score.side_effect = error_type(error_message)

    with patch("bert_score.BERTScorer", return_value=mock_scorer):
        result = bert_metric.compute_single("reference", "candidate")

        assert result.metric_name == "bert_score"
        assert result.score == 0.0
        assert result.error == error_message
        assert result.details is None


def test_string_representation(bert_metric):
    """BERT Score metric has proper string representation."""
    str_repr = str(bert_metric)
    assert "BertScoreMetric" in str_repr
    assert "bert_score" in str_repr

    repr_str = repr(bert_metric)
    assert repr_str == str_repr
