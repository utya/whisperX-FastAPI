"""Lightweight schema migrations for existing databases."""

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection


def ensure_task_schema(connection: Connection) -> None:
    """Add task columns introduced after initial deployments."""
    inspector = inspect(connection)
    if "tasks" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("tasks")}
    if "current_stage" not in columns:
        connection.execute(text("ALTER TABLE tasks ADD COLUMN current_stage VARCHAR"))
    if "partial_text" not in columns:
        connection.execute(text("ALTER TABLE tasks ADD COLUMN partial_text TEXT"))
