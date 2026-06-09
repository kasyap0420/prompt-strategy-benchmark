import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

from backend.config import BASE_DIR, settings


def _database_path() -> Path:
    parsed = urlparse(settings.database_url)
    if parsed.scheme != "sqlite":
        raise ValueError("Only SQLite DATABASE_URL values are supported in Phase 1.")

    raw_path = unquote(parsed.path or "")
    if parsed.netloc:
        raw_path = f"//{parsed.netloc}{raw_path}"

    if not raw_path:
        raise ValueError("SQLite DATABASE_URL must include a database file path.")

    if raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
        raw_path = raw_path[1:]
    elif raw_path.startswith("/") and not raw_path.startswith("//"):
        raw_path = raw_path[1:]

    path = Path(raw_path)
    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


DATABASE_PATH = _database_path()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Provide a SQLite connection with row access by column name."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Initialize the database file and keep schema creation intentionally minimal."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
