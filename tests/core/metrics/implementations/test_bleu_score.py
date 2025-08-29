from unittest.mock import Mock, patch
import pytest

from src.core.metrics.implementations.bleu_score import BleuMetric, BleuConfig
from src.models.schemas import MetricType, MetricResult, TextPair


# Fixtures
@pytest.fixture
def bleu_metric():
    """Basic BLEU Score metric instance."""
    return BleuMetric()


@pytest.fixture
def custom_bleu_metric():
    """BLEU Score metric with custom configuration."""
    return BleuMetric(
        config=BleuConfig(
            max_n=2,
            smooth_method="add-k",
            smooth_value=1.0,
            tokenize="intl",
            lowercase=True,
        )
    )


@pytest.fixture
def mock_bleu_scorer():
    """Mock SacreBLEU scorer with typical return values."""
    scorer = Mock()

    # Mock typical BLEU Score object with sentence_score method
    mock_score_result = Mock()
    mock_score_result.score = 75.5  # BLEU score (0-100 scale)
    mock_score_result.precisions = [0.85, 0.75, 0.65, 0.55]  # N-gram precisions
    mock_score_result.bp = 0.95  # Brevity penalty
    mock_score_result.sys_len = 8  # System (candidate) length
    mock_score_result.ref_len = 9  # Reference length

    scorer.sentence_score.return_value = mock_score_result
    return scorer


@pytest.fixture
def sample_text_pair():
    """Single text pair for testing."""
    return TextPair(
        reference="The quick brown fox jumps over the lazy dog",
        candidate="A fast brown fox leaps over a sleepy dog",
    )


@pytest.fixture
def sample_text_pairs():
    """Multiple text pairs for testing."""
    return [
        TextPair(reference="Hello world", candidate="Hi world"),
        TextPair(reference="Python is great", candidate="Python is awesome"),
        TextPair(reference="Testing code", candidate="Code testing"),
    ]


@pytest.fixture
def empty_text_pairs():
    """Empty list of text pairs."""
    return []


# Basic property tests


def test_metric_properties(bleu_metric):
    """BLEU Score metric has correct properties."""
    assert bleu_metric.name == "bleu_score"
    assert bleu_metric.metric_type == MetricType.BLEU
    assert (
        "BLEU Score: N-gram based evaluation (BLEU-4) using SacreBLEU"
        in bleu_metric.description
    )
    assert bleu_metric.requires_model_download is False


def test_custom_metric_properties(custom_bleu_metric):
    """Custom BLEU Score metric has correct properties."""
    assert custom_bleu_metric.name == "bleu_score"
    assert "BLEU-2" in custom_bleu_metric.description
    assert custom_bleu_metric.config.max_n == 2


def test_initial_state(bleu_metric):
    """BLEU Score metric starts in unloaded state."""
    assert bleu_metric._sacrebleu_loaded is False
    assert bleu_metric.config.max_n == 4
    assert bleu_metric.config.smooth_method == "exp"
    assert bleu_metric.config.tokenize == "13a"
    assert bleu_metric.config.lowercase is False


def test_get_model_info_not_loaded(bleu_metric):
    """get_model_info returns not_loaded when SacreBLEU isn't loaded."""
    info = bleu_metric.get_model_info()
    expected = {
        "status": "not_loaded",
        "library": "sacrebleu",
        "max_n": 4,
        "smooth_method": "exp",
        "smooth_value": 0.0,
        "tokenize": "13a",
        "lowercase": False,
        "requires_download": False,
    }
    assert info == expected


def test_get_model_info_loaded(bleu_metric, mock_bleu_scorer):
    """get_model_info returns correct info when SacreBLEU is loaded."""
    with patch("sacrebleu.BLEU", return_value=mock_bleu_scorer):
        bleu_metric._ensure_sacrebleu()
        info = bleu_metric.get_model_info()

        expected = {
            "status": "loaded",
            "library": "sacrebleu",
            "max_n": 4,
            "smooth_method": "exp",
            "smooth_value": 0.0,
            "tokenize": "13a",
            "lowercase": False,
            "requires_download": False,
        }
        assert info == expected


# SacreBLEU loading tests


@patch("sacrebleu.BLEU")
def test_ensure_sacrebleu_success(mock_sacrebleu_class, bleu_metric, mock_bleu_scorer):
    """_ensure_sacrebleu successfully loads SacreBLEU."""
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    bleu_metric._ensure_sacrebleu()

    assert bleu_metric._sacrebleu_loaded is True


def test_ensure_sacrebleu_import_error(bleu_metric):
    """_ensure_sacrebleu raises ImportError when sacrebleu not available."""
    with patch(
        "builtins.__import__", side_effect=ImportError("No module named 'sacrebleu'")
    ):
        with pytest.raises(ImportError, match="sacrebleu package is required"):
            bleu_metric._ensure_sacrebleu()

        assert bleu_metric._sacrebleu_loaded is False


def test_ensure_sacrebleu_runtime_error(bleu_metric):
    """_ensure_sacrebleu raises RuntimeError when initialization fails."""
    # Simulate a generic exception during sacrebleu import (not BLEU instantiation)
    with patch(
        "builtins.__import__", side_effect=RuntimeError("Module initialization failed")
    ):
        with pytest.raises(RuntimeError, match="Failed to initialize BLEU Score"):
            bleu_metric._ensure_sacrebleu()

        assert bleu_metric._sacrebleu_loaded is False


@patch("sacrebleu.BLEU")
def test_ensure_sacrebleu_only_loads_once(mock_sacrebleu_class, bleu_metric):
    """_ensure_sacrebleu only loads SacreBLEU once, subsequent calls do nothing."""

    assert bleu_metric._sacrebleu_loaded is False

    # Load multiple times
    bleu_metric._ensure_sacrebleu()
    bleu_metric._ensure_sacrebleu()
    bleu_metric._ensure_sacrebleu()

    assert bleu_metric._sacrebleu_loaded is True


# Single computation tests


@patch("sacrebleu.BLEU")
def test_compute_single_success(
    mock_sacrebleu_class, bleu_metric, mock_bleu_scorer, sample_text_pair
):
    """compute_single returns correct result for valid input."""
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    result = bleu_metric.compute_single(
        sample_text_pair.reference, sample_text_pair.candidate
    )

    assert isinstance(result, MetricResult)
    assert result.metric_name == "bleu_score"
    assert result.score == 0.7550  # 75.5 / 100
    assert result.error is None

    expected_details = {
        "bleu_score": 0.7550,
        "bleu_score_100": 75.50,
        "max_n": 4,
        "bleu_1": 0.85,
        "bleu_2": 0.75,
        "bleu_3": 0.65,
        "bleu_4": 0.55,
        "brevity_penalty": 0.95,
        "length_ratio": 0.8889,  # 8/9 rounded
        "reference_length": 9,
        "candidate_length": 8,
        "tokenization": "13a",
        "smoothing": "exp",
    }
    assert result.details == expected_details

    # Verify scorer was called correctly
    mock_bleu_scorer.sentence_score.assert_called_once_with(
        sample_text_pair.candidate, [sample_text_pair.reference]
    )


@patch("sacrebleu.BLEU")
def test_compute_single_with_custom_config(
    mock_sacrebleu_class, custom_bleu_metric, mock_bleu_scorer, sample_text_pair
):
    """compute_single works with custom configuration."""
    # Mock for custom config (BLEU-2)
    mock_score_result = Mock()
    mock_score_result.score = 80.0
    mock_score_result.precisions = [0.90, 0.80]
    mock_score_result.bp = 1.0
    mock_score_result.sys_len = 5
    mock_score_result.ref_len = 5
    mock_bleu_scorer.sentence_score.return_value = mock_score_result

    mock_sacrebleu_class.return_value = mock_bleu_scorer

    result = custom_bleu_metric.compute_single(
        sample_text_pair.reference, sample_text_pair.candidate
    )

    assert result.score == 0.8000
    assert result.details["max_n"] == 2
    assert result.details["bleu_1"] == 0.90
    assert result.details["bleu_2"] == 0.80
    assert "bleu_3" not in result.details or result.details["bleu_3"] == 0.0
    assert result.details["tokenization"] == "intl"
    assert result.details["smoothing"] == "add-k"

    # Verify BLEU object created with custom config
    mock_sacrebleu_class.assert_called_once_with(
        max_ngram_order=2,
        smooth_method="add-k",
        smooth_value=1.0,
        tokenize="intl",
        lowercase=True,
    )


@patch("sacrebleu.BLEU")
def test_compute_single_scorer_error(mock_sacrebleu_class, bleu_metric):
    """compute_single handles scorer errors gracefully."""
    mock_bleu_scorer = Mock()
    mock_bleu_scorer.sentence_score.side_effect = RuntimeError("Processing error")
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    result = bleu_metric.compute_single("reference", "candidate")

    assert result.metric_name == "bleu_score"
    assert result.score == 0.0
    assert result.error == "Processing error"
    assert result.details is None


@patch("sacrebleu.BLEU")
def test_compute_single_loads_sacrebleu_if_needed(
    mock_sacrebleu_class, bleu_metric, mock_bleu_scorer
):
    """compute_single automatically loads SacreBLEU if not loaded."""
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    assert bleu_metric._sacrebleu_loaded is False

    bleu_metric.compute_single("reference", "candidate")

    assert bleu_metric._sacrebleu_loaded is True


# Batch computation tests


@patch("sacrebleu.BLEU")
def test_compute_batch_empty_list(
    mock_sacrebleu_class, bleu_metric, mock_bleu_scorer, empty_text_pairs
):
    """compute_batch handles empty input correctly."""
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    results = bleu_metric.compute_batch(empty_text_pairs)

    assert results == []
    mock_bleu_scorer.sentence_score.assert_not_called()


@patch("sacrebleu.BLEU")
def test_compute_batch_single_pair(
    mock_sacrebleu_class, bleu_metric, mock_bleu_scorer, sample_text_pair
):
    """compute_batch works correctly with single pair."""
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    results = bleu_metric.compute_batch([sample_text_pair])

    assert len(results) == 1
    result = results[0]
    assert result.metric_name == "bleu_score"
    assert result.score == 0.7550
    assert result.error is None

    # Verify scorer call
    mock_bleu_scorer.sentence_score.assert_called_once_with(
        sample_text_pair.candidate, [sample_text_pair.reference]
    )


@patch("sacrebleu.BLEU")
def test_compute_batch_multiple_pairs(
    mock_sacrebleu_class, bleu_metric, sample_text_pairs
):
    """compute_batch processes multiple pairs correctly."""
    mock_bleu_scorer = Mock()

    # Mock different return values for each pair
    mock_results = [
        Mock(score=85.0, precisions=[0.9, 0.8, 0.7, 0.6], bp=1.0, sys_len=5, ref_len=5),
        Mock(
            score=75.0,
            precisions=[0.85, 0.75, 0.65, 0.55],
            bp=0.95,
            sys_len=6,
            ref_len=7,
        ),
        Mock(score=65.0, precisions=[0.8, 0.7, 0.6, 0.5], bp=0.9, sys_len=4, ref_len=6),
    ]
    mock_bleu_scorer.sentence_score.side_effect = mock_results
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    results = bleu_metric.compute_batch(sample_text_pairs)

    assert len(results) == 3

    # Check first result
    assert results[0].score == 0.8500
    assert results[0].details["bleu_1"] == 0.9

    # Check second result
    assert results[1].score == 0.7500
    assert results[1].details["brevity_penalty"] == 0.95

    # Check third result
    assert results[2].score == 0.6500
    assert results[2].details["bleu_4"] == 0.5

    # Verify all scorer calls
    assert mock_bleu_scorer.sentence_score.call_count == 3


@patch("sacrebleu.BLEU")
def test_compute_batch_handles_individual_errors(
    mock_sacrebleu_class, bleu_metric, sample_text_pairs
):
    """compute_batch handles errors in individual pairs."""
    mock_bleu_scorer = Mock()

    # First pair succeeds, second fails, third succeeds
    mock_success = Mock(
        score=80.0, precisions=[0.8, 0.7, 0.6, 0.5], bp=1.0, sys_len=5, ref_len=5
    )
    mock_bleu_scorer.sentence_score.side_effect = [
        mock_success,
        RuntimeError("Processing failed"),
        mock_success,
    ]
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    results = bleu_metric.compute_batch(sample_text_pairs)

    assert len(results) == 3

    # First result should succeed
    assert results[0].score == 0.8000
    assert results[0].error is None

    # Second result should have error
    assert results[1].score == 0.0
    assert results[1].error == "Processing failed"

    # Third result should succeed
    assert results[2].score == 0.8000
    assert results[2].error is None


@patch("sacrebleu.BLEU")
def test_compute_batch_loads_sacrebleu_if_needed(
    mock_sacrebleu_class, bleu_metric, mock_bleu_scorer, sample_text_pairs
):
    """compute_batch automatically loads SacreBLEU if not loaded."""
    mock_bleu_scorer.sentence_score.return_value = Mock(
        score=75.0, precisions=[0.8, 0.7, 0.6, 0.5], bp=1.0, sys_len=5, ref_len=5
    )
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    assert bleu_metric._sacrebleu_loaded is False

    bleu_metric.compute_batch(sample_text_pairs)

    assert bleu_metric._sacrebleu_loaded is True


# Configuration tests


def test_configure_method(bleu_metric):
    """configure method updates settings correctly."""
    bleu_metric.configure(
        config=BleuConfig(
            max_n=3,
            smooth_method="floor",
            smooth_value=0.5,
            tokenize="intl",
            lowercase=True,
        )
    )

    assert bleu_metric.config.max_n == 3
    assert bleu_metric.config.smooth_method == "floor"
    assert bleu_metric.config.smooth_value == 0.5
    assert bleu_metric.config.tokenize == "intl"
    assert bleu_metric.config.lowercase is True


def test_configure_partial_update(bleu_metric):
    """configure method allows partial updates."""
    original_smooth = bleu_metric.config.smooth_method
    original_tokenize = bleu_metric.config.tokenize

    bleu_metric.configure(config=BleuConfig(max_n=2, lowercase=True))

    assert bleu_metric.config.max_n == 2
    assert bleu_metric.config.lowercase is True
    # Other settings unchanged
    assert bleu_metric.config.smooth_method == original_smooth
    assert bleu_metric.config.tokenize == original_tokenize


def test_configure_none_values(bleu_metric):
    """configure method ignores None values."""
    original_config = {
        "max_n": bleu_metric.config.max_n,
        "smooth_method": bleu_metric.config.smooth_method,
        "smooth_value": bleu_metric.config.smooth_value,
        "tokenize": bleu_metric.config.tokenize,
        "lowercase": bleu_metric.config.lowercase,
    }

    bleu_metric.configure()

    # All settings should remain unchanged
    assert bleu_metric.config.max_n == original_config["max_n"]
    assert bleu_metric.config.smooth_method == original_config["smooth_method"]
    assert bleu_metric.config.smooth_value == original_config["smooth_value"]
    assert bleu_metric.config.tokenize == original_config["tokenize"]
    assert bleu_metric.config.lowercase == original_config["lowercase"]


# Observer notification tests


@patch("sacrebleu.BLEU")
def test_compute_batch_notifies_observers(
    mock_sacrebleu_class, bleu_metric, sample_text_pairs
):
    """compute_batch properly notifies observers."""
    mock_bleu_scorer = Mock()
    mock_bleu_scorer.sentence_score.return_value = Mock(
        score=75.0, precisions=[0.8, 0.7, 0.6, 0.5], bp=1.0, sys_len=5, ref_len=5
    )
    mock_sacrebleu_class.return_value = mock_bleu_scorer

    # Mock observer methods
    bleu_metric._notify_start = Mock()
    bleu_metric._notify_pair_processed = Mock()
    bleu_metric._notify_complete = Mock()

    results = bleu_metric.compute_batch(sample_text_pairs)

    # Verify observer calls
    bleu_metric._notify_start.assert_called_once_with(3)
    assert bleu_metric._notify_pair_processed.call_count == 3
    bleu_metric._notify_complete.assert_called_once_with(results)


# Edge cases and integration tests


def test_empty_strings(bleu_metric, mock_bleu_scorer):
    """compute_single handles empty strings."""
    mock_bleu_scorer.sentence_score.return_value = Mock(
        score=0.0, precisions=[0.0, 0.0, 0.0, 0.0], bp=0.0, sys_len=0, ref_len=0
    )

    with patch("sacrebleu.BLEU", return_value=mock_bleu_scorer):
        result = bleu_metric.compute_single("", "")

        assert isinstance(result, MetricResult)
        assert result.metric_name == "bleu_score"
        mock_bleu_scorer.sentence_score.assert_called_once_with("", [""])


def test_very_long_strings(bleu_metric, mock_bleu_scorer):
    """compute_single handles very long strings."""
    long_text = "word " * 1000  # Very long text

    with patch("sacrebleu.BLEU", return_value=mock_bleu_scorer):
        result = bleu_metric.compute_single(long_text, long_text)

        assert isinstance(result, MetricResult)
        mock_bleu_scorer.sentence_score.assert_called_once_with(long_text, [long_text])


def test_special_characters(bleu_metric, mock_bleu_scorer):
    """compute_single handles special characters and Unicode."""
    reference = "Hello 世界! @#$%^&*()_+-=[]{}|;:,.<>?"
    candidate = "Hi 世界! Special chars: @#$%"

    with patch("sacrebleu.BLEU", return_value=mock_bleu_scorer):
        result = bleu_metric.compute_single(reference, candidate)

        assert isinstance(result, MetricResult)
        mock_bleu_scorer.sentence_score.assert_called_once_with(candidate, [reference])


# Parameterized tests


@pytest.mark.parametrize(
    "bleu_score_100,expected_normalized",
    [
        (100.0, 1.0),  # Perfect score
        (75.5, 0.7550),  # Normal score
        (0.0, 0.0),  # Minimum score
        (50.0, 0.5),  # Half score
        (25.25, 0.2525),  # Decimal score
    ],
)
def test_score_normalization(bleu_metric, bleu_score_100, expected_normalized):
    """compute_single correctly normalizes scores from 0-100 to 0-1."""
    mock_bleu_scorer = Mock()
    mock_score_result = Mock()
    mock_score_result.score = bleu_score_100
    mock_score_result.precisions = [0.8, 0.7, 0.6, 0.5]
    mock_score_result.bp = 1.0
    mock_score_result.sys_len = 5
    mock_score_result.ref_len = 5
    mock_bleu_scorer.sentence_score.return_value = mock_score_result

    with patch("sacrebleu.BLEU", return_value=mock_bleu_scorer):
        result = bleu_metric.compute_single("reference", "candidate")

        assert result.score == expected_normalized
        assert result.details["bleu_score"] == expected_normalized
        assert result.details["bleu_score_100"] == bleu_score_100


@pytest.mark.parametrize("max_n", [1, 2, 3, 4, 5])
def test_different_max_n_values(max_n):
    """BLEU metric works with different max_n values."""
    bleu_metric = BleuMetric(config=BleuConfig(max_n=max_n))

    mock_bleu_scorer = Mock()
    mock_score_result = Mock()
    mock_score_result.score = 75.0
    mock_score_result.precisions = [0.9, 0.8, 0.7, 0.6, 0.5][
        :max_n
    ]  # Truncate to max_n
    mock_score_result.bp = 1.0
    mock_score_result.sys_len = 5
    mock_score_result.ref_len = 5
    mock_bleu_scorer.sentence_score.return_value = mock_score_result

    with patch("sacrebleu.BLEU", return_value=mock_bleu_scorer):
        result = bleu_metric.compute_single("reference", "candidate")

        assert result.details is not None
        assert result.details["max_n"] == max_n

        # Check that we have scores for 1 to max_n
        for n in range(1, max_n + 1):
            assert result.details.get(f"bleu_{n}") is not None
            assert result.details[f"bleu_{n}"] > 0

        # Check that we don't have scores beyond max_n
        for n in range(max_n + 1, 6):
            if result.details.get(f"bleu_{n}") is not None:
                assert result.details.get(f"bleu_{n}") == 0.0


@pytest.mark.parametrize("smooth_method", ["exp", "floor", "add-k", "none"])
def test_different_smoothing_methods(smooth_method):
    """BLEU metric works with different smoothing methods."""
    bleu_metric = BleuMetric(config=BleuConfig(smooth_method=smooth_method))

    with patch("sacrebleu.BLEU") as mock_sacrebleu_class:
        mock_bleu_scorer = Mock()
        mock_sacrebleu_class.return_value = mock_bleu_scorer

        bleu_metric.compute_single("reference", "candidate")

        # Verify BLEU was created with correct smoothing method
        mock_sacrebleu_class.assert_called_once()
        call_args = mock_sacrebleu_class.call_args[1]  # Get keyword arguments
        assert call_args["smooth_method"] == smooth_method


@pytest.mark.parametrize("tokenize_method", ["13a", "intl", "zh", "ja-mecab", "none"])
def test_different_tokenize_methods(tokenize_method):
    """BLEU metric works with different tokenization methods."""
    bleu_metric = BleuMetric(config=BleuConfig(tokenize=tokenize_method))

    with patch("sacrebleu.BLEU") as mock_sacrebleu_class:
        mock_bleu_scorer = Mock()
        mock_sacrebleu_class.return_value = mock_bleu_scorer

        bleu_metric.compute_single("reference", "candidate")

        # Verify BLEU was created with correct tokenization method
        mock_sacrebleu_class.assert_called_once()
        call_args = mock_sacrebleu_class.call_args[1]  # Get keyword arguments
        assert call_args["tokenize"] == tokenize_method


@pytest.mark.parametrize(
    "error_type,error_message",
    [
        (RuntimeError, "SacreBLEU processing error"),
        (ValueError, "Invalid input format"),
        (TypeError, "Wrong argument type"),
        (Exception, "Generic error"),
    ],
)
def test_compute_single_different_errors(bleu_metric, error_type, error_message):
    """compute_single handles different error types properly."""
    mock_bleu_scorer = Mock()
    mock_bleu_scorer.sentence_score.side_effect = error_type(error_message)

    with patch("sacrebleu.BLEU", return_value=mock_bleu_scorer):
        result = bleu_metric.compute_single("reference", "candidate")

        assert result.metric_name == "bleu_score"
        assert result.score == 0.0
        assert result.error == error_message
        assert result.details is None


def test_string_representation(bleu_metric):
    """BLEU Score metric has proper string representation."""
    str_repr = str(bleu_metric)
    assert "BleuMetric" in str_repr
    assert "bleu_score" in str_repr

    repr_str = repr(bleu_metric)
    assert repr_str == str_repr


# Integration tests with actual computation logic


def test_zero_precision_handling(bleu_metric):
    """Test handling of zero n-gram precisions."""
    mock_bleu_scorer = Mock()
    mock_score_result = Mock()
    mock_score_result.score = 0.0
    mock_score_result.precisions = [0.0, 0.0]  # Only 2 precisions, less than max_n=4
    mock_score_result.bp = 0.0
    mock_score_result.sys_len = 1
    mock_score_result.ref_len = 5
    mock_bleu_scorer.sentence_score.return_value = mock_score_result

    with patch("sacrebleu.BLEU", return_value=mock_bleu_scorer):
        result = bleu_metric.compute_single("reference", "candidate")

        assert result.details["bleu_1"] == 0.0
        assert result.details["bleu_2"] == 0.0
        assert result.details["bleu_3"] == 0.0  # Filled with 0.0
        assert result.details["bleu_4"] == 0.0  # Filled with 0.0
