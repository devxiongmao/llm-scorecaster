from unittest.mock import Mock, patch
import pytest

from src.core.metrics.implementations.rouge_score import RougeMetric
from src.models.schemas import MetricType, MetricResult, TextPair


# Fixtures
@pytest.fixture
def rouge_metric():
    """Basic ROUGE metric instance."""
    return RougeMetric()


@pytest.fixture
def custom_rouge_metric():
    """ROUGE metric with custom configuration."""
    return RougeMetric(rouge_types=["rouge1", "rouge2"], use_stemmer=False)


@pytest.fixture
def rouge1_only_metric():
    """ROUGE metric with only ROUGE-1."""
    return RougeMetric(rouge_types=["rouge1"])


@pytest.fixture
def mock_rouge_scorer():
    """Mock rouge-score scorer with typical return values."""
    scorer = Mock()

    # Mock typical ROUGE score results
    mock_rouge1 = Mock()
    mock_rouge1.precision = 0.8
    mock_rouge1.recall = 0.75
    mock_rouge1.fmeasure = 0.7742

    mock_rouge2 = Mock()
    mock_rouge2.precision = 0.6
    mock_rouge2.recall = 0.65
    mock_rouge2.fmeasure = 0.6244

    mock_rouge_l = Mock()
    mock_rouge_l.precision = 0.7
    mock_rouge_l.recall = 0.72
    mock_rouge_l.fmeasure = 0.7099

    mock_rouge_l_sum = Mock()
    mock_rouge_l_sum.precision = 0.68
    mock_rouge_l_sum.recall = 0.7
    mock_rouge_l_sum.fmeasure = 0.6898

    scorer.score.return_value = {
        "rouge1": mock_rouge1,
        "rouge2": mock_rouge2,
        "rougeL": mock_rouge_l,
        "rougeLsum": mock_rouge_l_sum,
    }
    return scorer


@pytest.fixture
def sample_text_pair():
    """Single text pair for testing."""
    return TextPair(
        reference="The quick brown fox jumps over the lazy dog.",
        candidate="A fast brown fox leaps over a sleepy dog.",
    )


@pytest.fixture
def sample_text_pairs():
    """Multiple text pairs for testing."""
    return [
        TextPair(reference="Hello world.", candidate="Hi world."),
        TextPair(
            reference="Python is great for data science.",
            candidate="Python is awesome for ML.",
        ),
        TextPair(
            reference="Testing code is important.", candidate="Code testing matters."
        ),
    ]


@pytest.fixture
def empty_text_pairs():
    """Empty list of text pairs."""
    return []


# Basic property tests


def test_metric_properties(rouge_metric):
    """ROUGE metric has correct properties."""
    assert rouge_metric.name == "rouge_score"
    assert rouge_metric.metric_type == MetricType.ROUGE
    assert "ROUGE Score: N-gram overlap evaluation" in rouge_metric.description
    assert "rouge1, rouge2, rougeL, rougeLsum" in rouge_metric.description
    assert "with stemmer" in rouge_metric.description
    assert rouge_metric.requires_model_download is True


def test_custom_metric_properties(custom_rouge_metric):
    """Custom ROUGE metric has correct properties."""
    assert custom_rouge_metric.name == "rouge_score"
    assert "rouge1, rouge2" in custom_rouge_metric.description
    assert "without stemmer" in custom_rouge_metric.description
    assert custom_rouge_metric.requires_model_download is False


def test_initial_state(rouge_metric):
    """ROUGE metric starts in unloaded state."""
    assert rouge_metric._rouge_score_loaded is False
    assert rouge_metric.rouge_types == ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    assert rouge_metric.use_stemmer is True
    assert rouge_metric._scorer is None


def test_get_model_info_not_loaded(rouge_metric):
    """get_model_info returns not_loaded when rouge-score isn't loaded."""
    info = rouge_metric.get_model_info()
    expected = {
        "status": "not_loaded",
        "library": "rouge-score",
        "rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        "use_stemmer": True,
        "requires_download": True,
        "supported_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    }
    assert info == expected


def test_get_model_info_loaded(rouge_metric, mock_rouge_scorer):
    """get_model_info returns correct info when rouge-score is loaded."""
    with patch("rouge_score.rouge_scorer.RougeScorer", return_value=mock_rouge_scorer):
        rouge_metric._ensure_rouge_score()
        info = rouge_metric.get_model_info()

        expected = {
            "status": "loaded",
            "library": "rouge-score",
            "rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            "use_stemmer": True,
            "requires_download": True,
            "supported_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        }
        assert info == expected


# Validation tests


def test_valid_rouge_types_initialization():
    """ROUGE metric accepts valid rouge types."""
    valid_combinations = [
        ["rouge1"],
        ["rouge2"],
        ["rougeL"],
        ["rougeLsum"],
        ["rouge1", "rouge2"],
        ["rouge1", "rougeL"],
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    ]

    for rouge_types in valid_combinations:
        metric = RougeMetric(rouge_types=rouge_types)
        assert metric.rouge_types == rouge_types


def test_invalid_rouge_types_initialization():
    """ROUGE metric rejects invalid rouge types."""
    invalid_combinations = [
        ["rouge3"],  # Doesn't exist
        ["rouge1", "rouge3"],  # Mix of valid and invalid
        ["bleu"],  # Wrong metric type
        ["rouge_1"],  # Wrong format
        [],  # Empty list
    ]

    for rouge_types in invalid_combinations:
        with pytest.raises(ValueError, match="Invalid ROUGE types"):
            RougeMetric(rouge_types=rouge_types)


def test_get_supported_rouge_types(rouge_metric):
    """get_supported_rouge_types returns correct set."""
    supported = rouge_metric.get_supported_rouge_types()
    expected = {"rouge1", "rouge2", "rougeL", "rougeLsum"}
    assert supported == expected


# rouge-score loading tests


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_ensure_rouge_score_success(mock_scorer_class, rouge_metric, mock_rouge_scorer):
    """_ensure_rouge_score successfully loads rouge-score."""
    mock_scorer_class.return_value = mock_rouge_scorer

    rouge_metric._ensure_rouge_score()

    assert rouge_metric._rouge_score_loaded is True
    mock_scorer_class.assert_called_once_with(
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )


def test_ensure_rouge_score_import_error(rouge_metric):
    """_ensure_rouge_score raises ImportError when rouge-score not available."""
    with patch(
        "builtins.__import__", side_effect=ImportError("No module named 'rouge_score'")
    ):
        with pytest.raises(ImportError, match="rouge-score package is required"):
            rouge_metric._ensure_rouge_score()

        assert rouge_metric._rouge_score_loaded is False


def test_ensure_rouge_score_runtime_error(rouge_metric):
    """_ensure_rouge_score raises RuntimeError when initialization fails."""
    with patch(
        "rouge_score.rouge_scorer.RougeScorer",
        side_effect=RuntimeError("Scorer initialization failed"),
    ):
        with pytest.raises(RuntimeError, match="Failed to initialize ROUGE scorer"):
            rouge_metric._ensure_rouge_score()

        assert rouge_metric._rouge_score_loaded is False


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_ensure_rouge_score_only_loads_once(mock_scorer_class, rouge_metric):
    """_ensure_rouge_score only loads rouge-score once."""
    assert rouge_metric._rouge_score_loaded is False

    # Load multiple times
    rouge_metric._ensure_rouge_score()
    rouge_metric._ensure_rouge_score()
    rouge_metric._ensure_rouge_score()

    assert rouge_metric._rouge_score_loaded is True
    # Should only be called once
    mock_scorer_class.assert_called_once()


# Single computation tests


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_single_success(
    mock_scorer_class, rouge_metric, mock_rouge_scorer, sample_text_pair
):
    """compute_single returns correct result for valid input."""
    mock_scorer_class.return_value = mock_rouge_scorer

    result = rouge_metric.compute_single(
        sample_text_pair.reference, sample_text_pair.candidate
    )

    assert isinstance(result, MetricResult)
    assert result.metric_name == "rouge_score"
    assert result.score == 0.7099  # ROUGE-L F1 (primary score)
    assert result.error is None

    expected_details = {
        "rouge1": {
            "precision": 0.8,
            "recall": 0.75,
            "f1": 0.7742,
        },
        "rouge2": {
            "precision": 0.6,
            "recall": 0.65,
            "f1": 0.6244,
        },
        "rougeL": {
            "precision": 0.7,
            "recall": 0.72,
            "f1": 0.7099,
        },
        "rougeLsum": {
            "precision": 0.68,
            "recall": 0.7,
            "f1": 0.6898,
        },
        "rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        "use_stemmer": True,
        "library": "rouge-score",
    }
    assert result.details == expected_details

    # Verify scorer was called correctly
    mock_rouge_scorer.score.assert_called_once_with(
        sample_text_pair.reference, sample_text_pair.candidate
    )


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_single_with_custom_config(
    mock_scorer_class, custom_rouge_metric, sample_text_pair
):
    """compute_single works with custom configuration."""
    # Mock for custom config (rouge1, rouge2 only)
    mock_scorer = Mock()
    mock_rouge1 = Mock()
    mock_rouge1.precision = 0.85
    mock_rouge1.recall = 0.8
    mock_rouge1.fmeasure = 0.8246

    mock_rouge2 = Mock()
    mock_rouge2.precision = 0.7
    mock_rouge2.recall = 0.65
    mock_rouge2.fmeasure = 0.6739

    mock_scorer.score.return_value = {
        "rouge1": mock_rouge1,
        "rouge2": mock_rouge2,
    }
    mock_scorer_class.return_value = mock_scorer

    result = custom_rouge_metric.compute_single(
        sample_text_pair.reference, sample_text_pair.candidate
    )

    # Should use first available F1 since no ROUGE-L
    assert result.score == 0.8246  # rouge1 F1
    assert "rouge1" in result.details
    assert "rouge2" in result.details
    assert "rougeL" not in result.details
    assert result.details["use_stemmer"] is False

    # Verify scorer created with custom config
    mock_scorer_class.assert_called_once_with(
        rouge_types=["rouge1", "rouge2"], use_stemmer=False
    )


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_single_rouge1_only_uses_as_primary(
    mock_scorer_class, rouge1_only_metric, sample_text_pair
):
    """compute_single uses rouge1 F1 as primary when only rouge1 configured."""
    mock_scorer = Mock()
    mock_rouge1 = Mock()
    mock_rouge1.precision = 0.9
    mock_rouge1.recall = 0.85
    mock_rouge1.fmeasure = 0.8739

    mock_scorer.score.return_value = {"rouge1": mock_rouge1}
    mock_scorer_class.return_value = mock_scorer

    result = rouge1_only_metric.compute_single(
        sample_text_pair.reference, sample_text_pair.candidate
    )

    assert result.score == 0.8739  # rouge1 F1
    assert len(result.details) == 4  # rouge1 + config info


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_single_scorer_error(mock_scorer_class, rouge_metric):
    """compute_single handles scorer errors gracefully."""
    mock_scorer = Mock()
    mock_scorer.score.side_effect = RuntimeError("ROUGE processing error")
    mock_scorer_class.return_value = mock_scorer

    result = rouge_metric.compute_single("reference", "candidate")

    assert result.metric_name == "rouge_score"
    assert result.score == 0.0
    assert result.error == "ROUGE processing error"
    assert result.details is None


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_single_loads_rouge_score_if_needed(
    mock_scorer_class, rouge_metric, mock_rouge_scorer
):
    """compute_single automatically loads rouge-score if not loaded."""
    mock_scorer_class.return_value = mock_rouge_scorer

    assert rouge_metric._rouge_score_loaded is False

    rouge_metric.compute_single("reference", "candidate")

    assert rouge_metric._rouge_score_loaded is True


# Batch computation tests


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_batch_empty_list(
    mock_scorer_class, rouge_metric, mock_rouge_scorer, empty_text_pairs
):
    """compute_batch handles empty input correctly."""
    mock_scorer_class.return_value = mock_rouge_scorer

    results = rouge_metric.compute_batch(empty_text_pairs)

    assert results == []
    mock_rouge_scorer.score.assert_not_called()


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_batch_single_pair(
    mock_scorer_class, rouge_metric, mock_rouge_scorer, sample_text_pair
):
    """compute_batch works correctly with single pair."""
    mock_scorer_class.return_value = mock_rouge_scorer

    results = rouge_metric.compute_batch([sample_text_pair])

    assert len(results) == 1
    result = results[0]
    assert result.metric_name == "rouge_score"
    assert result.score == 0.7099  # ROUGE-L F1
    assert result.error is None

    # Verify scorer call
    mock_rouge_scorer.score.assert_called_once_with(
        sample_text_pair.reference, sample_text_pair.candidate
    )


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_batch_multiple_pairs(
    mock_scorer_class, rouge_metric, sample_text_pairs
):
    """compute_batch processes multiple pairs correctly."""
    mock_scorer = Mock()

    # Mock different return values for each pair
    def mock_score_side_effect(ref, cand):
        if "Hello" in ref:
            return {
                "rouge1": Mock(precision=0.9, recall=0.8, fmeasure=0.8421),
                "rouge2": Mock(precision=0.7, recall=0.6, fmeasure=0.6462),
                "rougeL": Mock(precision=0.85, recall=0.75, fmeasure=0.7955),
                "rougeLsum": Mock(precision=0.8, recall=0.7, fmeasure=0.7467),
            }
        if "Python" in ref:
            return {
                "rouge1": Mock(precision=0.8, recall=0.75, fmeasure=0.7742),
                "rouge2": Mock(precision=0.6, recall=0.65, fmeasure=0.6244),
                "rougeL": Mock(precision=0.7, recall=0.72, fmeasure=0.7099),
                "rougeLsum": Mock(precision=0.68, recall=0.7, fmeasure=0.6898),
            }

        return {
            "rouge1": Mock(precision=0.75, recall=0.7, fmeasure=0.7246),
            "rouge2": Mock(precision=0.5, recall=0.55, fmeasure=0.5238),
            "rougeL": Mock(precision=0.65, recall=0.68, fmeasure=0.6646),
            "rougeLsum": Mock(precision=0.6, recall=0.65, fmeasure=0.6244),
        }

    mock_scorer.score.side_effect = mock_score_side_effect
    mock_scorer_class.return_value = mock_scorer

    results = rouge_metric.compute_batch(sample_text_pairs)

    assert len(results) == 3

    # Check first result (Hello world)
    assert results[0].score == 0.7955  # ROUGE-L F1
    assert results[0].details["rouge1"]["f1"] == 0.8421

    # Check second result (Python)
    assert results[1].score == 0.7099  # ROUGE-L F1
    assert results[1].details["rouge2"]["f1"] == 0.6244

    # Check third result (Testing)
    assert results[2].score == 0.6646  # ROUGE-L F1
    assert results[2].details["rougeLsum"]["f1"] == 0.6244

    # Verify all scorer calls
    assert mock_scorer.score.call_count == 3


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_batch_handles_individual_errors(
    mock_scorer_class, rouge_metric, sample_text_pairs
):
    """compute_batch handles errors in individual pairs."""
    mock_scorer = Mock()

    # First pair succeeds, second fails, third succeeds
    def mock_score_side_effect(ref, cand):
        if "Hello" in ref:
            return {
                "rouge1": Mock(precision=0.8, recall=0.75, fmeasure=0.7742),
                "rougeL": Mock(precision=0.7, recall=0.72, fmeasure=0.7099),
            }
        if "Python" in ref:
            raise RuntimeError("Processing failed for Python text")

        return {
            "rouge1": Mock(precision=0.75, recall=0.7, fmeasure=0.7246),
            "rougeL": Mock(precision=0.65, recall=0.68, fmeasure=0.6646),
        }

    mock_scorer.score.side_effect = mock_score_side_effect
    mock_scorer_class.return_value = mock_scorer

    results = rouge_metric.compute_batch(sample_text_pairs)

    assert len(results) == 3

    # First result should succeed
    assert results[0].score == 0.7099
    assert results[0].error is None

    # Second result should have error
    assert results[1].score == 0.0
    assert results[1].error == "Processing failed for Python text"

    # Third result should succeed
    assert results[2].score == 0.6646
    assert results[2].error is None


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_batch_loads_rouge_score_if_needed(
    mock_scorer_class, rouge_metric, mock_rouge_scorer, sample_text_pairs
):
    """compute_batch automatically loads rouge-score if not loaded."""
    mock_scorer_class.return_value = mock_rouge_scorer

    assert rouge_metric._rouge_score_loaded is False

    rouge_metric.compute_batch(sample_text_pairs)

    assert rouge_metric._rouge_score_loaded is True


# Configuration tests


def test_configure_method(rouge_metric):
    """configure method updates settings correctly."""
    rouge_metric.configure(
        rouge_types=["rouge1", "rougeL"],
        use_stemmer=False,
    )

    assert rouge_metric.rouge_types == ["rouge1", "rougeL"]
    assert rouge_metric.use_stemmer is False
    # Should reset loader state
    assert rouge_metric._rouge_score_loaded is False
    assert rouge_metric._scorer is None


def test_configure_partial_update(rouge_metric):
    """configure method allows partial updates."""
    original_stemmer = rouge_metric.use_stemmer

    rouge_metric.configure(rouge_types=["rouge2"])

    assert rouge_metric.rouge_types == ["rouge2"]
    assert rouge_metric.use_stemmer == original_stemmer


def test_configure_none_values(rouge_metric):
    """configure method ignores None values."""
    original_config = {
        "rouge_types": rouge_metric.rouge_types.copy(),
        "use_stemmer": rouge_metric.use_stemmer,
    }

    rouge_metric.configure(rouge_types=None, use_stemmer=None)

    # All settings should remain unchanged
    assert rouge_metric.rouge_types == original_config["rouge_types"]
    assert rouge_metric.use_stemmer == original_config["use_stemmer"]


def test_configure_invalid_rouge_types(rouge_metric):
    """configure method validates rouge types."""
    with pytest.raises(ValueError, match="Invalid ROUGE types"):
        rouge_metric.configure(rouge_types=["rouge3", "invalid"])

    # Original config should be unchanged
    assert rouge_metric.rouge_types == ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def test_configure_no_change_doesnt_reset_loader(rouge_metric):
    """configure method doesn't reset loader if no changes made."""
    # Simulate loaded state
    rouge_metric._rouge_score_loaded = True
    rouge_metric._scorer = Mock()

    # Configure with same values
    rouge_metric.configure(
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )

    # Should not reset loader
    assert rouge_metric._rouge_score_loaded is True
    assert rouge_metric._scorer is not None


# Observer notification tests


@patch("rouge_score.rouge_scorer.RougeScorer")
def test_compute_batch_notifies_observers(
    mock_scorer_class, rouge_metric, sample_text_pairs
):
    """compute_batch properly notifies observers."""
    mock_scorer = Mock()
    mock_scorer.score.return_value = {
        "rouge1": Mock(precision=0.8, recall=0.75, fmeasure=0.7742),
        "rougeL": Mock(precision=0.7, recall=0.72, fmeasure=0.7099),
    }
    mock_scorer_class.return_value = mock_scorer

    # Mock observer methods
    rouge_metric._notify_start = Mock()
    rouge_metric._notify_pair_processed = Mock()
    rouge_metric._notify_complete = Mock()

    results = rouge_metric.compute_batch(sample_text_pairs)

    # Verify observer calls
    rouge_metric._notify_start.assert_called_once_with(3)
    assert rouge_metric._notify_pair_processed.call_count == 3
    rouge_metric._notify_complete.assert_called_once_with(results)


# Edge cases and integration tests


def test_empty_strings(rouge_metric, mock_rouge_scorer):
    """compute_single handles empty strings."""
    mock_rouge_scorer.score.return_value = {
        "rouge1": Mock(precision=0.0, recall=0.0, fmeasure=0.0),
        "rougeL": Mock(precision=0.0, recall=0.0, fmeasure=0.0),
    }

    with patch("rouge_score.rouge_scorer.RougeScorer", return_value=mock_rouge_scorer):
        result = rouge_metric.compute_single("", "")

        assert isinstance(result, MetricResult)
        assert result.metric_name == "rouge_score"
        mock_rouge_scorer.score.assert_called_once_with("", "")


def test_very_long_strings(rouge_metric, mock_rouge_scorer):
    """compute_single handles very long strings."""
    long_text = "word " * 1000  # Very long text

    with patch("rouge_score.rouge_scorer.RougeScorer", return_value=mock_rouge_scorer):
        result = rouge_metric.compute_single(long_text, long_text)

        assert isinstance(result, MetricResult)
        mock_rouge_scorer.score.assert_called_once_with(long_text, long_text)


def test_special_characters(rouge_metric, mock_rouge_scorer):
    """compute_single handles special characters and Unicode."""
    reference = "Hello 世界! @#$%^&*()_+-=[]{}|;:,.<>?"
    candidate = "Hi 世界! Special chars: @#$%"

    with patch("rouge_score.rouge_scorer.RougeScorer", return_value=mock_rouge_scorer):
        result = rouge_metric.compute_single(reference, candidate)

        assert isinstance(result, MetricResult)
        mock_rouge_scorer.score.assert_called_once_with(reference, candidate)


# Parameterized tests


@pytest.mark.parametrize(
    "rouge_types",
    [
        ["rouge1"],
        ["rouge2"],
        ["rougeL"],
        ["rougeLsum"],
        ["rouge1", "rouge2"],
        ["rouge1", "rougeL"],
        ["rouge2", "rougeLsum"],
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    ],
)
def test_different_rouge_type_combinations(rouge_types):
    """ROUGE metric works with different rouge type combinations."""
    rouge_metric = RougeMetric(rouge_types=rouge_types)

    mock_scorer = Mock()
    mock_results = {}
    for rouge_type in rouge_types:
        mock_results[rouge_type] = Mock(precision=0.7, recall=0.75, fmeasure=0.7246)
    mock_scorer.score.return_value = mock_results

    with patch("rouge_score.rouge_scorer.RougeScorer", return_value=mock_scorer):
        result = rouge_metric.compute_single("reference", "candidate")

        assert result.details is not None
        assert result.details["rouge_types"] == rouge_types

        # Check that we have scores for all specified types
        for rouge_type in rouge_types:
            assert result.details.get(rouge_type) is not None
            assert result.details[rouge_type]["f1"] == 0.7246

        # Primary score should be ROUGE-L if available, otherwise first
        if "rougeL" in rouge_types:
            assert result.score == 0.7246  # ROUGE-L F1
        else:
            assert result.score == 0.7246  # First available F1


@pytest.mark.parametrize("use_stemmer", [True, False])
def test_different_stemmer_settings(use_stemmer):
    """ROUGE metric works with different stemmer settings."""
    rouge_metric = RougeMetric(use_stemmer=use_stemmer)

    with patch("rouge_score.rouge_scorer.RougeScorer") as mock_scorer_class:
        mock_scorer = Mock()
        mock_scorer_class.return_value = mock_scorer

        rouge_metric.compute_single("reference", "candidate")

        # Verify RougeScorer was created with correct stemmer setting
        mock_scorer_class.assert_called_once()
        call_args = mock_scorer_class.call_args[1]  # Get keyword arguments
        assert call_args["use_stemmer"] == use_stemmer


@pytest.mark.parametrize(
    "error_type,error_message",
    [
        (RuntimeError, "ROUGE processing error"),
        (ValueError, "Invalid input format"),
        (TypeError, "Wrong argument type"),
        (Exception, "Generic error"),
    ],
)
def test_compute_single_different_errors(rouge_metric, error_type, error_message):
    """compute_single handles different error types properly."""
    mock_scorer = Mock()
    mock_scorer.score.side_effect = error_type(error_message)

    with patch("rouge_score.rouge_scorer.RougeScorer", return_value=mock_scorer):
        result = rouge_metric.compute_single("reference", "candidate")

        assert result.metric_name == "rouge_score"
        assert result.score == 0.0
        assert result.error == error_message
        assert result.details is None


def test_string_representation(rouge_metric):
    """ROUGE metric has proper string representation."""
    str_repr = str(rouge_metric)
    assert "RougeMetric" in str_repr
    assert "rouge_score" in str_repr
    assert "rouge1,rouge2,rougeL,rougeLsum" in str_repr

    repr_str = repr(rouge_metric)
    assert repr_str == str_repr


def test_string_representation_custom_config(custom_rouge_metric):
    """ROUGE metric string representation reflects custom config."""
    str_repr = str(custom_rouge_metric)
    assert "RougeMetric" in str_repr
    assert "rouge1,rouge2" in str_repr
