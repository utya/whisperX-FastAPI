"""Tests for the in-process task cancellation registry."""

import pytest

from app.core.task_cancellation import (
    clear_cancellation,
    is_cancelled,
    request_cancellation,
)


@pytest.mark.unit
def test_request_and_check_cancellation() -> None:
    """Cancellation can be registered and queried."""
    task_id = "task-cancel-1"
    try:
        assert is_cancelled(task_id) is False
        request_cancellation(task_id)
        assert is_cancelled(task_id) is True
    finally:
        clear_cancellation(task_id)


@pytest.mark.unit
def test_clear_cancellation() -> None:
    """Cleared tasks are no longer reported as cancelled."""
    task_id = "task-cancel-2"
    request_cancellation(task_id)
    clear_cancellation(task_id)
    assert is_cancelled(task_id) is False
