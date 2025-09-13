"""Tests for the Health Celery Task"""

import pytest
from celery import states

from src.celery.tasks.health import _health_check_task, health_check_task

TASK_ID = "health_check_test"


@pytest.fixture(name="expected_response")
def expected_response_fixture():
    """Expected Response Fixture for Health Task"""
    return {"status": "healthy", "message": "Celery worker is running"}


def test_health_check_task_returns_expected_response(expected_response: dict):
    """Test that health check task returns correct response structure."""
    result = _health_check_task()

    assert result == expected_response
    assert isinstance(result, dict)
    assert "status" in result
    assert "message" in result


def test_health_check_task_is_successful_on_execution():
    """Test that health check task executes successfully through Celery."""
    task_result = health_check_task.apply(task_id=TASK_ID)

    assert task_result.state == states.SUCCESS
    assert task_result.result == {
        "status": "healthy",
        "message": "Celery worker is running",
    }


def test_health_check_task_has_correct_task_name():
    """Test that the task is registered with the correct name."""
    assert health_check_task.name == "health_check"


def test_health_check_task_is_idempotent():
    """Test that multiple calls return the same result."""
    result1 = _health_check_task()
    result2 = _health_check_task()

    assert result1 == result2


@pytest.mark.parametrize(
    "task_id",
    [
        pytest.param("test-1", id="with alphanumeric task_id"),
        pytest.param("health_check_monitoring", id="with descriptive task_id"),
        pytest.param(None, id="with default task_id"),
    ],
)
def test_health_check_task_works_with_different_task_ids(
    task_id: str, expected_response: dict
):
    """Test that health check works regardless of task ID."""
    if task_id:
        task_result = health_check_task.apply(task_id=task_id)
    else:
        task_result = health_check_task.apply()

    assert task_result.state == states.SUCCESS
    assert task_result.result == expected_response
