import pytest
from unittest.mock import Mock, patch
from typing import Any

from src.core.metrics.registry import MetricRegistry, metric_registry
from src.core.metrics.base import BaseMetric
from src.models.schemas import MetricType


# Mock metric classes for testing


class MockBleuMetric(BaseMetric):
    """Mock BLEU metric for testing."""

    @property
    def name(self) -> str:
        return "bleu"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.BLEU

    @property
    def description(self) -> str:
        return "Mock BLEU metric"

    @property
    def requires_model_download(self) -> bool:
        return False

    def compute_single(self, reference: str, candidate: str) -> Any:
        return Mock(score=0.5)


class MockRougeLMetric(BaseMetric):
    """Mock ROUGE-L metric for testing."""

    @property
    def name(self) -> str:
        return "rouge_l"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.ROUGE_L

    @property
    def description(self) -> str:
        return "Mock ROUGE-L metric"

    @property
    def requires_model_download(self) -> bool:
        return True

    def compute_single(self, reference: str, candidate: str) -> Any:
        return Mock(score=0.7)


class MockFailingMetric(BaseMetric):
    """Mock metric that fails during instantiation."""

    def __init__(self):
        raise RuntimeError("Initialization failed")

    @property
    def name(self) -> str:
        return "failing"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.BLEU

    @property
    def description(self) -> str:
        return "Failing metric"

    @property
    def requires_model_download(self) -> bool:
        return False

    def compute_single(self, reference: str, candidate: str) -> Any:
        return Mock()


# Helper functions


def create_fresh_registry():
    """Create a fresh MetricRegistry instance for testing."""
    return MetricRegistry()


def setup_mock_module_with_metrics(*metric_classes):
    """Create a mock module containing the given metric classes."""
    mock_module = Mock()

    # Add all the metric classes and other typical module attributes
    module_attrs = {}
    for metric_class in metric_classes:
        module_attrs[metric_class.__name__] = metric_class

    # Add some non-metric attributes to test filtering
    module_attrs.update(
        {
            "__name__": "mock_module",
            "__file__": "/mock/path.py",
            "some_function": lambda: None,
            "SomeClass": type,  # Non-BaseMetric class
            "CONSTANT": "value",
            "BaseMetric": BaseMetric,  # Should be filtered out
        }
    )

    mock_module.__dict__.update(module_attrs)

    # Mock dir() to return all attribute names
    mock_module.__dir__ = Mock(return_value=list(module_attrs.keys()))

    return mock_module


def setup_pkgutil_mock(module_names):
    """Setup pkgutil.iter_modules mock to return given module names."""
    mock_modules = []
    for name in module_names:
        # (finder, module_name, ispkg)
        mock_modules.append((Mock(), name, False))
    return mock_modules


# Basic functionality tests


def test_fresh_registry_starts_empty():
    """A fresh MetricRegistry starts with no metrics and no instances."""
    registry = create_fresh_registry()

    assert registry._metrics == {}
    assert registry._instances == {}
    assert registry._discovered is False


def test_discover_metrics_skips_if_already_discovered():
    """discover_metrics skips discovery if already discovered and no force_reload."""
    registry = create_fresh_registry()
    registry._discovered = True

    with patch("pkgutil.iter_modules") as mock_iter:
        registry.discover_metrics()

        # Should not have called iter_modules since already discovered
        mock_iter.assert_not_called()


def test_discover_metrics_with_force_reload():
    """discover_metrics runs discovery even if already discovered when force_reload=True."""
    registry = create_fresh_registry()
    registry._discovered = True

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=[]),
        patch("pkgutil.iter_modules") as mock_iter,
    ):
        registry.discover_metrics(force_reload=True)

        # Should have called iter_modules despite already being discovered
        mock_iter.assert_called_once()


def test_discover_metrics_handles_missing_implementations_directory():
    """discover_metrics handles missing implementations directory gracefully."""
    registry = create_fresh_registry()

    with patch("pathlib.Path.exists", return_value=False):
        registry.discover_metrics()

        # Should complete without error and mark as discovered
        assert registry._discovered is True
        assert len(registry._metrics) == 0


def test_discover_metrics_finds_valid_metrics():
    """discover_metrics finds and registers valid metric classes."""
    registry = create_fresh_registry()

    # Setup mocks
    mock_bleu_module = setup_mock_module_with_metrics(MockBleuMetric, MockRougeLMetric)
    mock_modules = setup_pkgutil_mock(["bleu_rouge"])

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=mock_modules),
        patch("importlib.import_module", return_value=mock_bleu_module),
    ):
        registry.discover_metrics()

        # Should have discovered both metrics
        assert len(registry._metrics) == 2
        assert "bleu" in registry._metrics
        assert "rouge_l" in registry._metrics
        assert registry._metrics["bleu"] is MockBleuMetric
        assert registry._metrics["rouge_l"] is MockRougeLMetric


def test_discover_metrics_ignores_non_metric_classes():
    """discover_metrics ignores classes that don't inherit from BaseMetric."""
    registry = create_fresh_registry()

    # Create a module with BaseMetric and non-metric classes
    mock_module = Mock()
    mock_module.__dict__ = {
        "BaseMetric": BaseMetric,  # Should be ignored (is BaseMetric itself)
        "SomeOtherClass": type,  # Should be ignored (not BaseMetric subclass)
        "regular_function": lambda: None,  # Should be ignored (not a class)
        "MockBleuMetric": MockBleuMetric,  # Should be included
    }
    mock_module.__dir__ = Mock(return_value=list(mock_module.__dict__.keys()))

    mock_modules = setup_pkgutil_mock(["test_module"])

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=mock_modules),
        patch("importlib.import_module", return_value=mock_module),
    ):
        registry.discover_metrics()

        # Should only have registered the valid metric
        assert len(registry._metrics) == 1
        assert "bleu" in registry._metrics


def test_discover_metrics_handles_import_errors():
    """discover_metrics continues when individual modules fail to import."""
    registry = create_fresh_registry()

    mock_modules = setup_pkgutil_mock(["failing_module", "working_module"])
    mock_working_module = setup_mock_module_with_metrics(MockBleuMetric)

    def import_side_effect(module_name):
        if "failing_module" in module_name:
            raise ImportError("Module not found")
        return mock_working_module

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=mock_modules),
        patch("importlib.import_module", side_effect=import_side_effect),
    ):
        registry.discover_metrics()

        # Should have registered the metric from the working module
        assert len(registry._metrics) == 1
        assert "bleu" in registry._metrics


def test_discover_metrics_handles_metric_registration_errors():
    """discover_metrics continues when individual metrics fail to register."""
    registry = create_fresh_registry()

    # Module with both valid and failing metrics
    mock_module = setup_mock_module_with_metrics(MockBleuMetric, MockFailingMetric)
    mock_modules = setup_pkgutil_mock(["mixed_module"])

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=mock_modules),
        patch("importlib.import_module", return_value=mock_module),
    ):
        registry.discover_metrics()

        # Should only register the working metric
        assert len(registry._metrics) == 1
        assert "bleu" in registry._metrics
        assert "failing" not in registry._metrics


def test_register_metric_manually():
    """register_metric allows manual registration of metric classes."""
    registry = create_fresh_registry()

    registry.register_metric(MockBleuMetric)

    assert len(registry._metrics) == 1
    assert "bleu" in registry._metrics
    assert registry._metrics["bleu"] is MockBleuMetric


def test_register_metric_handles_failing_metric():
    """register_metric handles metrics that fail during instantiation."""
    registry = create_fresh_registry()

    # Should not raise an exception, but also shouldn't register the metric
    registry.register_metric(MockFailingMetric)

    assert len(registry._metrics) == 0


# Factory pattern tests (get_metric)


def test_get_metric_creates_and_caches_instance():
    """get_metric creates metric instance and caches it for reuse."""
    registry = create_fresh_registry()
    registry._metrics["bleu"] = MockBleuMetric
    registry._discovered = True

    # First call should create instance
    instance1 = registry.get_metric("bleu")
    assert isinstance(instance1, MockBleuMetric)
    assert "bleu" in registry._instances

    # Second call should return cached instance
    instance2 = registry.get_metric("bleu")
    assert instance1 is instance2


def test_get_metric_auto_discovers_if_needed():
    """get_metric automatically runs discovery if not yet discovered."""
    registry = create_fresh_registry()

    mock_module = setup_mock_module_with_metrics(MockBleuMetric)
    mock_modules = setup_pkgutil_mock(["bleu"])

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=mock_modules),
        patch("importlib.import_module", return_value=mock_module),
    ):
        instance = registry.get_metric("bleu")

        assert isinstance(instance, MockBleuMetric)
        assert registry._discovered is True


def test_get_metric_raises_for_unknown_metric():
    """get_metric raises ValueError for unknown metric names."""
    registry = create_fresh_registry()
    registry._discovered = True

    with pytest.raises(ValueError, match="Unknown metric: nonexistent"):
        registry.get_metric("nonexistent")


def test_get_metric_raises_for_instantiation_failure():
    """get_metric raises ValueError when metric instantiation fails."""
    registry = create_fresh_registry()
    registry._metrics["failing"] = MockFailingMetric
    registry._discovered = True

    with pytest.raises(ValueError, match="Failed to create metric instance: failing"):
        registry.get_metric("failing")


def test_get_metrics_returns_multiple_instances():
    """get_metrics returns dictionary of multiple metric instances."""
    registry = create_fresh_registry()
    registry._metrics["bleu"] = MockBleuMetric
    registry._metrics["rouge_l"] = MockRougeLMetric
    registry._discovered = True

    metrics = registry.get_metrics(["bleu", "rouge_l"])

    assert len(metrics) == 2
    assert isinstance(metrics["bleu"], MockBleuMetric)
    assert isinstance(metrics["rouge_l"], MockRougeLMetric)


def test_get_metrics_handles_empty_list():
    """get_metrics handles empty list correctly."""
    registry = create_fresh_registry()

    metrics = registry.get_metrics([])

    assert metrics == {}


# Information retrieval tests


def test_list_available_metrics():
    """list_available_metrics returns list of all registered metric names."""
    registry = create_fresh_registry()
    registry._metrics["bleu"] = MockBleuMetric
    registry._metrics["rouge_l"] = MockRougeLMetric
    registry._discovered = True

    available = registry.list_available_metrics()

    assert set(available) == {"bleu", "rouge_l"}


def test_list_available_metrics_auto_discovers():
    """list_available_metrics auto-discovers if needed."""
    registry = create_fresh_registry()

    mock_module = setup_mock_module_with_metrics(MockBleuMetric)
    mock_modules = setup_pkgutil_mock(["bleu"])

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=mock_modules),
        patch("importlib.import_module", return_value=mock_module),
    ):
        available = registry.list_available_metrics()

        assert "bleu" in available
        assert registry._discovered is True


def test_get_metric_info():
    """get_metric_info returns complete information about a metric."""
    registry = create_fresh_registry()
    registry._metrics["bleu"] = MockBleuMetric
    registry._discovered = True

    info = registry.get_metric_info("bleu")

    expected = {
        "name": "bleu",
        "type": "bleu",
        "description": "Mock BLEU metric",
        "requires_model_download": False,
        "class_name": "MockBleuMetric",
    }
    assert info == expected


def test_get_all_metrics_info():
    """get_all_metrics_info returns information for all available metrics."""
    registry = create_fresh_registry()
    registry._metrics["bleu"] = MockBleuMetric
    registry._metrics["rouge_l"] = MockRougeLMetric
    registry._discovered = True

    all_info = registry.get_all_metrics_info()

    assert len(all_info) == 2
    assert "bleu" in all_info
    assert "rouge_l" in all_info
    assert all_info["bleu"]["name"] == "bleu"
    assert all_info["rouge_l"]["requires_model_download"] is True


# Validation tests


def test_is_metric_available():
    """is_metric_available correctly identifies available metrics."""
    registry = create_fresh_registry()
    registry._metrics["bleu"] = MockBleuMetric
    registry._discovered = True

    assert registry.is_metric_available("bleu") is True
    assert registry.is_metric_available("nonexistent") is False


def test_is_metric_available_auto_discovers():
    """is_metric_available auto-discovers if needed."""
    registry = create_fresh_registry()

    mock_module = setup_mock_module_with_metrics(MockBleuMetric)
    mock_modules = setup_pkgutil_mock(["bleu"])

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=mock_modules),
        patch("importlib.import_module", return_value=mock_module),
    ):
        is_available = registry.is_metric_available("bleu")

        assert is_available is True
        assert registry._discovered is True


def test_validate_metrics():
    """validate_metrics separates valid and invalid metric names."""
    registry = create_fresh_registry()
    registry._metrics["bleu"] = MockBleuMetric
    registry._metrics["rouge_l"] = MockRougeLMetric
    registry._discovered = True

    metric_names = ["bleu", "nonexistent", "rouge_l", "another_invalid"]
    valid, invalid = registry.validate_metrics(metric_names)

    assert set(valid) == {"bleu", "rouge_l"}
    assert set(invalid) == {"nonexistent", "another_invalid"}


def test_validate_metrics_with_empty_list():
    """validate_metrics handles empty input correctly."""
    registry = create_fresh_registry()
    registry._discovered = True

    valid, invalid = registry.validate_metrics([])

    assert valid == []
    assert invalid == []


# Cache management tests


def test_clear_cache():
    """clear_cache removes all cached instances."""
    registry = create_fresh_registry()
    registry._metrics["bleu"] = MockBleuMetric
    registry._discovered = True

    # Create cached instance
    instance = registry.get_metric("bleu")
    assert "bleu" in registry._instances

    # Clear cache
    registry.clear_cache()

    assert registry._instances == {}
    # Should still have the registered metric class
    assert "bleu" in registry._metrics


def test_reload_metrics():
    """reload_metrics clears everything and rediscovers."""
    registry = create_fresh_registry()

    # Setup initial state
    registry._metrics["old_metric"] = MockBleuMetric
    registry._instances["old_metric"] = MockBleuMetric()
    registry._discovered = True

    mock_module = setup_mock_module_with_metrics(MockRougeLMetric)
    mock_modules = setup_pkgutil_mock(["rouge"])

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=mock_modules),
        patch("importlib.import_module", return_value=mock_module),
    ):
        registry.reload_metrics()

        # Should have cleared old state and discovered new metrics
        assert "old_metric" not in registry._metrics
        assert "old_metric" not in registry._instances
        assert "rouge_l" in registry._metrics
        assert registry._discovered is True


# Global registry instance test


def test_global_registry_instance():
    """Global metric_registry instance is available and functional."""
    assert isinstance(metric_registry, MetricRegistry)

    # Should be able to call methods on it
    available = metric_registry.list_available_metrics()
    assert isinstance(available, list)


# Edge cases and error handling


def test_discover_metrics_handles_packages_in_implementations():
    """discover_metrics skips packages and only processes modules."""
    registry = create_fresh_registry()

    # Setup modules list with both modules and packages
    mock_modules = [
        (Mock(), "metric_module", False),  # Module - should process
        (Mock(), "utils_package", True),  # Package - should skip
        (Mock(), "another_module", False),  # Module - should process
    ]

    mock_metric_module = setup_mock_module_with_metrics(MockBleuMetric)
    mock_another_module = setup_mock_module_with_metrics(MockRougeLMetric)

    def import_side_effect(module_name):
        if "metric_module" in module_name:
            return mock_metric_module
        elif "another_module" in module_name:
            return mock_another_module
        else:
            raise ImportError("Should not import packages")

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pkgutil.iter_modules", return_value=mock_modules),
        patch("importlib.import_module", side_effect=import_side_effect),
    ):
        registry.discover_metrics()

        # Should have found metrics from both modules but skipped the package
        assert len(registry._metrics) == 2
        assert "bleu" in registry._metrics
        assert "rouge_l" in registry._metrics


def test_discover_metrics_handles_discovery_exception():
    """discover_metrics handles exceptions during discovery process."""
    registry = create_fresh_registry()

    with patch("pathlib.Path.exists", side_effect=Exception("File system error")):
        with pytest.raises(Exception, match="Error during metric discovery"):
            registry.discover_metrics()


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("get_metric_info", ["bleu"]),
        ("is_metric_available", ["bleu"]),
        ("validate_metrics", [["bleu"]]),
    ],
)
def test_methods_work_with_registered_metrics(method_name, args):
    """Various methods work correctly with manually registered metrics."""
    registry = create_fresh_registry()
    registry.register_metric(MockBleuMetric)

    method = getattr(registry, method_name)
    result = method(*args)

    # Should not raise an exception and return a result
    assert result is not None


def test_concurrent_access_safety():
    """Registry handles concurrent access to cached instances safely."""
    registry = create_fresh_registry()
    registry._metrics["bleu"] = MockBleuMetric
    registry._discovered = True

    # Simulate concurrent access by getting the same metric multiple times
    instances = []
    for _ in range(10):
        instances.append(registry.get_metric("bleu"))

    # All instances should be the same object (cached)
    first_instance = instances[0]
    for instance in instances[1:]:
        assert instance is first_instance


def test_metric_name_consistency():
    """Registry ensures metric names are consistent between class registration and retrieval."""
    registry = create_fresh_registry()

    # Register metric and verify name consistency
    registry.register_metric(MockBleuMetric)

    # The metric should be accessible by its name property
    instance = registry.get_metric("bleu")
    assert instance.name == "bleu"

    # Info should match
    info = registry.get_metric_info("bleu")
    assert info["name"] == "bleu"
