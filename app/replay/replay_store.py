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
                quality_score, quality_method, quality_explanation,
                quality_confidence, quality_flags, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    r.quality_explanation,
                    r.quality_confidence,
                    json.dumps(r.quality_flags),
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


def _decode_quality_flags(raw) -> list:
    """Safely decode a stored quality_flags value to a Python list.

    NULL          → []
    valid JSON list  → that list
    malformed JSON   → []
    valid JSON but not a list (e.g. a string or dict) → []
    """
    if raw is None:
        return []
    if not isinstance(raw, str):
        return []
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def get_replay_results(replay_run_id: str) -> list[dict]:
    """Return persisted replay results with quality_flags decoded to a Python list."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM replay_results WHERE replay_run_id = ?",
            (replay_run_id,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["quality_flags"] = _decode_quality_flags(d.get("quality_flags"))
            results.append(d)
        return results


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
