"""
Tests for Celery configuration module.
"""

from src.celery.config import (
    get_celery_config,
    get_config_for_environment,
    get_development_config,
    get_production_config,
)


def test_get_celery_config():
    """Test that get_celery_config returns expected configuration."""
    expected = {
        # Task settings
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        # Result settings
        "result_expires": 3600,
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

    assert get_celery_config() == expected


def test_get_development_config():
    """Test that get_development_config returns expected configuration."""
    expected = {
        # Base config
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        "result_expires": 3600,
        "result_persistent": True,
        "worker_prefetch_multiplier": 1,
        "task_acks_late": True,
        "worker_max_tasks_per_child": 1000,
        "task_reject_on_worker_lost": True,
        "task_default_retry_delay": 60,
        "task_max_retries": 3,
        # Development overrides
        "task_always_eager": False,
        "task_eager_propagates": True,
        "worker_log_level": "DEBUG",
    }

    assert get_development_config() == expected


def test_get_production_config():
    """Test that get_production_config returns expected configuration."""
    expected = {
        # Base config
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        "result_expires": 3600,
        "result_persistent": True,
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
        "task_default_retry_delay": 60,
        "task_max_retries": 3,
        # Production overrides
        "worker_prefetch_multiplier": 4,
        "worker_max_tasks_per_child": 5000,
        "task_soft_time_limit": 300,
        "task_time_limit": 600,
    }

    assert get_production_config() == expected


def test_get_config_for_environment_for_production():
    """Test that get_production_config returns expected configuration."""
    expected = {
        # Base config
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        "result_expires": 3600,
        "result_persistent": True,
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
        "task_default_retry_delay": 60,
        "task_max_retries": 3,
        # Production overrides
        "worker_prefetch_multiplier": 4,
        "worker_max_tasks_per_child": 5000,
        "task_soft_time_limit": 300,
        "task_time_limit": 600,
    }

    assert get_config_for_environment("production") == expected


def test_get_config_for_environment_for_development():
    """Test that get_development_config returns expected configuration."""
    expected = {
        # Base config
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        "result_expires": 3600,
        "result_persistent": True,
        "worker_prefetch_multiplier": 1,
        "task_acks_late": True,
        "worker_max_tasks_per_child": 1000,
        "task_reject_on_worker_lost": True,
        "task_default_retry_delay": 60,
        "task_max_retries": 3,
        # Development overrides
        "task_always_eager": False,
        "task_eager_propagates": True,
        "worker_log_level": "DEBUG",
    }

    assert get_config_for_environment("development") == expected
