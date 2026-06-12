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
