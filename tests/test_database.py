"""
Tests for database.py: init_db, lightweight migrations.
"""
import sqlite3

import pytest

from app import database
from app.config import settings


def _column_names(db_path: str, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


# ── Normal init ───────────────────────────────────────────────────────────────

def test_init_db_creates_all_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "test.db"))
    database.init_db()
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"audit_runs", "usage_records", "reports", "replay_runs", "replay_results", "migration_simulations"} <= tables


def test_init_db_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "test.db"))
    database.init_db()
    database.init_db()  # second call must not raise


def test_migration_simulations_has_traceability_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "test.db"))
    database.init_db()
    cols = _column_names(str(tmp_path / "test.db"), "migration_simulations")
    assert "audit_run_id" in cols
    assert "replay_run_id" in cols


# ── Lightweight migration: columns already missing ────────────────────────────

def _create_old_migration_simulations(db_path: str) -> None:
    """Create migration_simulations as it existed before audit_run_id/replay_run_id were added."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE migration_simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            scenario_name TEXT NOT NULL,
            source_model TEXT NOT NULL,
            target_model TEXT NOT NULL,
            current_annualized_cost REAL NOT NULL,
            simulated_annualized_cost REAL NOT NULL,
            estimated_annual_savings REAL NOT NULL,
            estimated_savings_pct REAL NOT NULL,
            average_quality_delta REAL NOT NULL,
            confidence_score REAL NOT NULL,
            recommendation TEXT NOT NULL,
            rationale TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def test_lightweight_migration_adds_missing_columns(tmp_path, monkeypatch):
    db_path = str(tmp_path / "old.db")
    _create_old_migration_simulations(db_path)

    # Confirm columns are absent before migration
    assert "audit_run_id" not in _column_names(db_path, "migration_simulations")
    assert "replay_run_id" not in _column_names(db_path, "migration_simulations")

    monkeypatch.setattr(settings, "database_path", db_path)
    database.init_db()

    cols = _column_names(db_path, "migration_simulations")
    assert "audit_run_id" in cols
    assert "replay_run_id" in cols


def test_lightweight_migration_is_idempotent(tmp_path, monkeypatch):
    db_path = str(tmp_path / "old.db")
    _create_old_migration_simulations(db_path)

    monkeypatch.setattr(settings, "database_path", db_path)
    database.init_db()
    database.init_db()  # second call must not raise even though columns now exist


def test_lightweight_migration_preserves_existing_rows(tmp_path, monkeypatch):
    db_path = str(tmp_path / "old.db")
    _create_old_migration_simulations(db_path)

    # Insert a row using the old schema
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO migration_simulations
           (created_at, scenario_name, source_model, target_model,
            current_annualized_cost, simulated_annualized_cost,
            estimated_annual_savings, estimated_savings_pct,
            average_quality_delta, confidence_score, recommendation, rationale)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("2024-01-01T00:00:00+00:00", "legacy", "gpt-4o", "gpt-4o-mini",
         10000.0, 7000.0, 3000.0, 30.0, -0.01, 0.8, "proceed", "looks good"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(settings, "database_path", db_path)
    database.init_db()

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM migration_simulations WHERE scenario_name = 'legacy'").fetchall()
    conn.close()
    assert len(rows) == 1
