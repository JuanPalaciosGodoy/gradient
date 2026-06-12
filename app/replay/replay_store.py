"""
SQLite persistence for replay runs and results.

Tables are created by init_db() in app/database.py.
"""
import json
from datetime import datetime, timezone

from app.database import get_connection
from app.replay.replay_models import MigrationSimulationResult, ReplayResult


def save_replay_run(
    replay_run_id: str,
    audit_run_id: str | None,
    candidate_models: list[str],
    record_count: int,
    status: str = "complete",
) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO replay_runs
               (replay_run_id, audit_run_id, created_at, candidate_models, record_count, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                replay_run_id,
                audit_run_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(candidate_models),
                record_count,
                status,
            ),
        )


def save_replay_results(replay_run_id: str, results: list[ReplayResult]) -> None:
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO replay_results
               (replay_run_id, replay_id, original_record_id, candidate_provider,
                candidate_model, candidate_response, estimated_cost, latency_ms,
                quality_score, quality_method, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    replay_run_id,
                    r.replay_id,
                    r.original_record_id,
                    r.candidate_provider,
                    r.candidate_model,
                    r.candidate_response,
                    r.estimated_cost,
                    r.latency_ms,
                    r.quality_score,
                    r.quality_method,
                    r.error_message,
                )
                for r in results
            ],
        )


def get_migration_simulations(scenario_name: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM migration_simulations WHERE scenario_name = ?",
            (scenario_name,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_replay_results(replay_run_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM replay_results WHERE replay_run_id = ?",
            (replay_run_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def save_migration_simulation(
    result: MigrationSimulationResult,
    audit_run_id: str | None = None,
    replay_run_id: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO migration_simulations
               (created_at, audit_run_id, replay_run_id, scenario_name, source_model, target_model,
                current_annualized_cost, simulated_annualized_cost,
                estimated_annual_savings, estimated_savings_pct,
                average_quality_delta, confidence_score, recommendation, rationale)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                audit_run_id,
                replay_run_id,
                result.scenario_name,
                result.source_model,
                result.target_model,
                result.current_annualized_cost,
                result.simulated_annualized_cost,
                result.estimated_annual_savings,
                result.estimated_savings_pct,
                result.average_quality_delta,
                result.confidence_score,
                result.recommendation,
                result.rationale,
            ),
        )
