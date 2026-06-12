import pytest
from fastapi.testclient import TestClient

from app import database
from app.config import settings
from app.main import app


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Isolated per-test SQLite database with all tables created."""
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "test.db"))
    database.init_db()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient backed by an isolated per-test SQLite database."""
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c
