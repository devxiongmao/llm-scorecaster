"""
Celery configuration settings.

This module contains all Celery-specific configuration in one place,
making it easy to modify settings without touching the core app setup.
"""

from typing import Dict, Any, Literal


def get_celery_config() -> Dict[str, Any]:
    """
    Get Celery configuration dictionary.

    Returns:
        Dictionary containing all Celery configuration settings
    """
    return {
        # Task settings
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        # Result settings
        "result_expires": 3600,  # Results expire after 1 hour
        "result_persistent": True,
        # Worker settings
        "worker_prefetch_multiplier": 1,
        "task_acks_late": True,
        "worker_max_tasks_per_child": 1000,
        # Retry settings
        "task_reject_on_worker_lost": True,
        "task_default_retry_delay": 60,
        "task_max_retries": 3,
    }


def get_development_config() -> Dict[str, Any]:
    """
    Get development-specific Celery configuration.

    Returns:
        Dictionary with development overrides
    """
    config = get_celery_config()
    config.update(
        {
            "task_always_eager": False,  # Set to True for synchronous testing
            "task_eager_propagates": True,
            "worker_log_level": "DEBUG",
        }
    )
    return config


def get_production_config() -> Dict[str, Any]:
    """
    Get production-specific Celery configuration.

    Returns:
        Dictionary with production overrides
    """
    config = get_celery_config()
    config.update(
        {
            "worker_prefetch_multiplier": 4,  # Higher for production
            "worker_max_tasks_per_child": 5000,  # More tasks per worker
            "task_soft_time_limit": 300,  # 5 minutes
            "task_time_limit": 600,  # 10 minutes hard limit
        }
    )
    return config


def get_config_for_environment(
    environment: Literal["development", "production"],
) -> Dict[str, Any]:
    """
    Get Celery configuration for the specified environment.

    Args:
        environment: The environment to get configuration for

    Returns:
        Dictionary containing environment-specific Celery configuration
    """
    if environment == "production":
        return get_production_config()

    return get_development_config()
