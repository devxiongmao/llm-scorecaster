"""Registry for discovering and managing metric implementations."""

from typing import Any, Dict, List, Tuple, Type
import importlib
import pkgutil
from pathlib import Path
import logging
from src.core.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class MetricRegistry:
    """
    Registry for discovering and managing metric implementations.

    Uses the Registry Pattern to automatically discover metric classes
    and the Factory Pattern to instantiate them dynamically.
    """

    def __init__(self):
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._instances: Dict[str, BaseMetric] = {}
        self._discovered = False

    def _get_implementations_path(self) -> Path:
        """Get the implementations directory path."""
        return Path(__file__).parent / "implementations"

    def _discover_module_metrics(
        self, module_name: str, implementations_pkg: str
    ) -> None:
        """Discover metrics in a single module."""
        full_module_name = f"{implementations_pkg}.{module_name}"

        try:
            module = importlib.import_module(full_module_name)
            self._register_metrics_from_module(module)
        except Exception as e:
            logger.error("Failed to import %s: %s", full_module_name, e)

    def _register_metrics_from_module(self, module) -> None:
        """Register all BaseMetric subclasses found in a module."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if (
                isinstance(attr, type)
                and issubclass(attr, BaseMetric)
                and attr is not BaseMetric
            ):
                self._register_metric_class(attr)
                logger.info("Discovered metric: %s", attr.__name__)

    def discover_metrics(self, force_reload: bool = False) -> None:
        """
        Automatically discover all metric implementations.

        Scans the implementations package for classes that inherit from BaseMetric
        and registers them automatically.

        Args:
            force_reload: Whether to force rediscovery of metrics
        """
        if self._discovered and not force_reload:
            return

        logger.info("Discovering metric implementations...")

        try:
            implementations_path = self._get_implementations_path()

            if not implementations_path.exists():
                logger.warning(
                    "Implementations directory not found: %s", implementations_path
                )
                self._discovered = True  # Mark as discovered even if directory missing
                return

            # Import the implementations package
            implementations_pkg = "src.core.metrics.implementations"

            # Iterate through all modules in the implementations package
            for _, module_name, ispkg in pkgutil.iter_modules(
                [str(implementations_path)]
            ):
                if not ispkg:  # Skip packages, only import modules
                    self._discover_module_metrics(module_name, implementations_pkg)

            self._discovered = True
            logger.info("Discovery complete. Found %d metrics.", len(self._metrics))

        except Exception as e:
            logger.error("Error during metric discovery: %s", e)
            raise

    def _register_metric_class(self, metric_class: Type[BaseMetric]) -> None:
        """Register a metric class."""
        # Create a temporary instance to get the metric name
        try:
            temp_instance = metric_class()
            metric_name = temp_instance.name
            self._metrics[metric_name] = metric_class
        except Exception as e:
            logger.error(
                "Failed to register metric class %s: %s", metric_class.__name__, e
            )

    def register_metric(self, metric_class: Type[BaseMetric]) -> None:
        """
        Manually register a metric class.

        Args:
            metric_class: The metric class to register
        """
        self._register_metric_class(metric_class)

    def get_metric(self, metric_name: str) -> BaseMetric:
        """
        Get a metric instance by name (Factory Pattern).

        Args:
            metric_name: Name of the metric to retrieve

        Returns:
            BaseMetric: Instance of the requested metric

        Raises:
            ValueError: If the metric is not found
        """
        if not self._discovered:
            self.discover_metrics()

        # Return cached instance if available
        if metric_name in self._instances:
            return self._instances[metric_name]

        # Create new instance if class is registered
        if metric_name in self._metrics:
            try:
                instance = self._metrics[metric_name]()
                self._instances[metric_name] = instance
                return instance
            except Exception as e:
                logger.error("Failed to instantiate metric %s: %s", metric_name, e)
                raise ValueError(
                    f"Failed to create metric instance: {metric_name}"
                ) from e

        raise ValueError(f"Unknown metric: {metric_name}")

    def get_metrics(self, metric_names: List[str]) -> Dict[str, BaseMetric]:
        """
        Get multiple metric instances by names.

        Args:
            metric_names: List of metric names to retrieve

        Returns:
            Dict[str, BaseMetric]: Dictionary mapping names to metric instances
        """
        return {name: self.get_metric(name) for name in metric_names}

    def list_available_metrics(self) -> List[str]:
        """
        Get list of all available metric names.

        Returns:
            List[str]: List of metric names
        """
        if not self._discovered:
            self.discover_metrics()

        return list(self._metrics.keys())

    def get_metric_info(
        self, metric_name: str
    ) -> Dict[str, Any]:  # Should not be Any, Fix this pyright
        """
        Get information about a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dict containing metric information
        """
        metric = self.get_metric(metric_name)
        return {
            "name": metric.name,
            "type": metric.metric_type.value,
            "description": metric.description,
            "requires_model_download": metric.requires_model_download,
            "class_name": metric.__class__.__name__,
        }

    def get_all_metrics_info(
        self,
    ) -> Dict[str, Dict[str, Any]]:  # Should not be Any, Fix this pyright
        """
        Get information about all available metrics.

        Returns:
            Dict mapping metric names to their information
        """
        return {
            name: self.get_metric_info(name) for name in self.list_available_metrics()
        }

    def is_metric_available(self, metric_name: str) -> bool:
        """
        Check if a metric is available.

        Args:
            metric_name: Name of the metric to check

        Returns:
            bool: True if metric is available, False otherwise
        """
        if not self._discovered:
            self.discover_metrics()

        return metric_name in self._metrics

    def validate_metrics(self, metric_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate a list of metric names.

        Args:
            metric_names: List of metric names to validate

        Returns:
            Tuple of (valid_metrics, invalid_metrics)
        """
        if not self._discovered:
            self.discover_metrics()

        valid = []
        invalid = []

        for name in metric_names:
            if self.is_metric_available(name):
                valid.append(name)
            else:
                invalid.append(name)

        return valid, invalid

    def clear_cache(self) -> None:
        """Clear the instance cache."""
        self._instances.clear()

    def reload_metrics(self) -> None:
        """Reload all metrics from scratch."""
        self._metrics.clear()
        self._instances.clear()
        self._discovered = False
        self.discover_metrics()

    def are_metrics_discovered(self) -> bool:
        """Public method letting users know
        if metrics have been discovered yet

        Returns:
            bool: True if metrics have been discovered
        """
        return self._discovered


# Global registry instance
metric_registry = MetricRegistry()
