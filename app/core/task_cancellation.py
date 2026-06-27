"""In-process task cancellation registry for cooperative pipeline abort."""

import threading

_cancelled_tasks: set[str] = set()
_lock = threading.Lock()


def request_cancellation(task_id: str) -> None:
    """Register a task as cancelled for the current worker process."""
    with _lock:
        _cancelled_tasks.add(task_id)


def is_cancelled(task_id: str) -> bool:
    """Return whether cancellation was requested for the given task."""
    with _lock:
        return task_id in _cancelled_tasks


def clear_cancellation(task_id: str) -> None:
    """Remove a task from the in-process cancellation registry."""
    with _lock:
        _cancelled_tasks.discard(task_id)
