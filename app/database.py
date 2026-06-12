import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.config import settings


@contextmanager
def get_connection():
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Add `column` to `table` if it does not already exist."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _run_lightweight_migrations(conn: sqlite3.Connection) -> None:
    """Idempotent column additions for databases created before a schema change."""
    _ensure_column(conn, "migration_simulations", "audit_run_id", "TEXT")
    _ensure_column(conn, "migration_simulations", "replay_run_id", "TEXT")
    _ensure_column(conn, "replay_results", "quality_explanation", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "replay_results", "quality_confidence", "REAL NOT NULL DEFAULT 1.0")
    _ensure_column(conn, "replay_results", "quality_flags", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "migration_simulations", "avg_current_quality", "REAL NOT NULL DEFAULT 0.0")
    _ensure_column(conn, "migration_simulations", "avg_simulated_quality", "REAL NOT NULL DEFAULT 0.0")
    _ensure_column(conn, "migration_simulations", "avg_latency_delta_ms", "REAL NOT NULL DEFAULT 0.0")
    _ensure_column(conn, "migration_simulations", "records_analyzed", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "migration_simulations", "failed_replays", "INTEGER NOT NULL DEFAULT 0")


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                file_name TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_run_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                cost REAL NOT NULL,
                feedback TEXT,
                task_type TEXT,
                FOREIGN KEY (audit_run_id) REFERENCES audit_runs(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                audit_run_id TEXT NOT NULL UNIQUE,
                generated_at TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS replay_runs (
                replay_run_id TEXT PRIMARY KEY,
                audit_run_id TEXT,
                created_at TEXT NOT NULL,
                candidate_models TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                status TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS replay_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                replay_run_id TEXT NOT NULL,
                replay_id TEXT NOT NULL,
                original_record_id TEXT NOT NULL,
                candidate_provider TEXT NOT NULL,
                candidate_model TEXT NOT NULL,
                candidate_response TEXT NOT NULL,
                estimated_cost REAL NOT NULL,
                latency_ms REAL NOT NULL,
                quality_score REAL NOT NULL,
                quality_method TEXT NOT NULL,
                quality_explanation TEXT NOT NULL DEFAULT '',
                quality_confidence REAL NOT NULL DEFAULT 1.0,
                quality_flags TEXT NOT NULL DEFAULT '[]',
                error_message TEXT,
                FOREIGN KEY (replay_run_id) REFERENCES replay_runs(replay_run_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                audit_run_id TEXT,
                replay_run_id TEXT,
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
                rationale TEXT NOT NULL,
                avg_current_quality REAL NOT NULL DEFAULT 0.0,
                avg_simulated_quality REAL NOT NULL DEFAULT 0.0,
                avg_latency_delta_ms REAL NOT NULL DEFAULT 0.0,
                records_analyzed INTEGER NOT NULL DEFAULT 0,
                failed_replays INTEGER NOT NULL DEFAULT 0
            )
        """)
        _run_lightweight_migrations(conn)


def save_audit_run(run_id: str, file_name: str, record_count: int, status: str = "complete") -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audit_runs (id, created_at, record_count, status, file_name) VALUES (?, ?, ?, ?, ?)",
            (run_id, datetime.now(timezone.utc).isoformat(), record_count, status, file_name),
        )


def save_usage_records(audit_run_id: str, records: list[dict]) -> None:
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO usage_records
               (audit_run_id, prompt, response, timestamp, model, cost, feedback, task_type)
               VALUES (:audit_run_id, :prompt, :response, :timestamp, :model, :cost, :feedback, :task_type)""",
            [
                {
                    "audit_run_id": audit_run_id,
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "timestamp": r["timestamp"],
                    "model": r["model"],
                    "cost": r["cost"],
                    "feedback": r.get("feedback"),
                    "task_type": r.get("task_type"),
                }
                for r in records
            ],
        )


def audit_run_exists(audit_run_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM audit_runs WHERE id = ?",
            (audit_run_id,),
        ).fetchone()
        return row is not None


def get_usage_records(audit_run_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM usage_records WHERE audit_run_id = ?",
            (audit_run_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def save_report(report) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO reports (report_id, audit_run_id, generated_at, report_json)
               VALUES (?, ?, ?, ?)""",
            (report.report_id, report.audit_run_id, report.generated_at.isoformat(), report.model_dump_json()),
        )


def delete_migration_simulations_for_replay_run(replay_run_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM migration_simulations WHERE replay_run_id = ?",
            (replay_run_id,),
        )


def get_migration_simulations_for_replay_run(replay_run_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM migration_simulations WHERE replay_run_id = ? ORDER BY estimated_annual_savings DESC",
            (replay_run_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def count_replay_results(replay_run_id: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM replay_results WHERE replay_run_id = ?",
            (replay_run_id,),
        ).fetchone()
        return row[0] if row else 0


def get_replay_run(replay_run_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM replay_runs WHERE replay_run_id = ?",
            (replay_run_id,),
        ).fetchone()
        return dict(row) if row else None


def get_report(audit_run_id: str):
    from app.schemas import AuditReport  # local import avoids circular dependency at module load
    with get_connection() as conn:
        row = conn.execute(
            "SELECT report_json FROM reports WHERE audit_run_id = ?",
            (audit_run_id,),
        ).fetchone()
        if row is None:
            return None
        return AuditReport.model_validate_json(row["report_json"])
