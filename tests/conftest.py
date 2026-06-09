from pathlib import Path

import pytest

from backend import database
from backend.main import limiter


@pytest.fixture(autouse=True)
def isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "benchmark-test.db")
    limiter.reset()
