"""
SQLite persistence for replay runs and results.

Tables are created by init_db() in app/database.py.
"""
import json
from datetime import datetime, timezone

from app.database import (
    delete_migration_simulations_for_replay_run,
    get_connection,
    get_migration_simulations_for_replay_run,
    get_replay_run,
)
from app.replay.replay_models import MigrationSimulationResult, ReplayResult
from app.schemas import EvidenceLevel, ValidationStatus
from app.utils.evidence import adjust_confidence_for_evidence


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
                quality_confidence, quality_flags, error_message,
                evidence_level, evidence_summary, validation_status, limitations,
                confidence_score, input_tokens, output_tokens, cost_source, latency_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    r.evidence_level.value,
                    r.evidence_summary,
                    r.validation_status.value,
                    json.dumps(r.limitations),
                    r.confidence_score,
                    r.input_tokens,
                    r.output_tokens,
                    r.cost_source,
                    r.latency_source,
                )
                for r in results
            ],
        )


def get_simulations_for_replay_run(replay_run_id: str) -> list[MigrationSimulationResult]:
    """Load persisted simulation results for a replay run as model objects."""
    rows = get_migration_simulations_for_replay_run(replay_run_id)
    results = []
    for row in rows:
        level = EvidenceLevel(row.get("evidence_level") or "heuristic")
        status = ValidationStatus(row.get("validation_status") or "not_validated")
        stored_base = row.get("base_confidence_score")
        stored_conf = row["confidence_score"]
        if stored_base:
            # New row: both fields persisted — use stored values as authoritative.
            # Avoids re-running adjust_confidence_for_evidence() with potentially
            # updated factors after the row was originally written.
            base_conf = stored_base
            adj_conf = stored_conf
        else:
            # Legacy row: base_confidence_score column was added by migration and
            # defaulted to 0.0, so confidence_score holds the raw pre-Phase-2.5 value.
            base_conf = stored_conf
            adj_conf = adjust_confidence_for_evidence(stored_conf, level, status)
        results.append(MigrationSimulationResult(
            scenario_name=row["scenario_name"],
            source_model=row["source_model"],
            target_model=row["target_model"],
            current_annualized_cost=row["current_annualized_cost"],
            simulated_annualized_cost=row["simulated_annualized_cost"],
            estimated_annual_savings=row["estimated_annual_savings"],
            estimated_savings_pct=row["estimated_savings_pct"],
            average_quality_delta=row["average_quality_delta"],
            base_confidence_score=base_conf,
            confidence_score=adj_conf,
            recommendation=row["recommendation"],
            rationale=row["rationale"],
            avg_current_quality=row.get("avg_current_quality") or 0.0,
            avg_simulated_quality=row.get("avg_simulated_quality") or 0.0,
            avg_latency_delta_ms=row.get("avg_latency_delta_ms") or 0.0,
            records_analyzed=row.get("records_analyzed") or 0,
            failed_replays=row.get("failed_replays") or 0,
            evidence_level=level,
            evidence_summary=row.get("evidence_summary") or "",
            validation_status=status,
            limitations=_decode_json_list(row.get("limitations")),
            evidence_coverage_pct=row.get("evidence_coverage_pct") or 0.0,
            evidence_counts=_decode_json_dict(row.get("evidence_counts")),
        ))
    return results


def replace_migration_simulations(
    replay_run_id: str,
    audit_run_id: str | None,
    simulations: list[MigrationSimulationResult],
) -> None:
    """Atomically replace all simulation rows for a replay run.

    DELETE and INSERT run inside a single SQLite transaction.  If the INSERT
    fails for any reason the DELETE is rolled back, leaving the database in
    its previous state.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            created_at,
            audit_run_id,
            replay_run_id,
            s.scenario_name,
            s.source_model,
            s.target_model,
            s.current_annualized_cost,
            s.simulated_annualized_cost,
            s.estimated_annual_savings,
            s.estimated_savings_pct,
            s.average_quality_delta,
            s.base_confidence_score,
            s.confidence_score,
            s.recommendation,
            s.rationale,
            s.avg_current_quality,
            s.avg_simulated_quality,
            s.avg_latency_delta_ms,
            s.records_analyzed,
            s.failed_replays,
            s.evidence_level.value,
            s.evidence_summary,
            s.validation_status.value,
            json.dumps(s.limitations),
            s.evidence_coverage_pct,
            json.dumps(s.evidence_counts),
        )
        for s in simulations
    ]
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM migration_simulations WHERE replay_run_id = ?",
            (replay_run_id,),
        )
        conn.executemany(
            """INSERT INTO migration_simulations
               (created_at, audit_run_id, replay_run_id, scenario_name, source_model, target_model,
                current_annualized_cost, simulated_annualized_cost,
                estimated_annual_savings, estimated_savings_pct,
                average_quality_delta, base_confidence_score, confidence_score,
                recommendation, rationale,
                avg_current_quality, avg_simulated_quality, avg_latency_delta_ms,
                records_analyzed, failed_replays,
                evidence_level, evidence_summary, validation_status, limitations,
                evidence_coverage_pct, evidence_counts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def get_migration_simulations(scenario_name: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM migration_simulations WHERE scenario_name = ?",
            (scenario_name,),
        ).fetchall()
        return [dict(row) for row in rows]


def _decode_json_list(raw) -> list:
    """Safely decode a stored JSON list column to a Python list."""
    if raw is None:
        return []
    if not isinstance(raw, str):
        return []
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def _decode_json_dict(raw) -> dict:
    """Safely decode a stored JSON dict column to a Python dict."""
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return {}
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


# Backward-compatible alias used by existing tests.
_decode_quality_flags = _decode_json_list


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
            d["quality_flags"] = _decode_json_list(d.get("quality_flags"))
            d["limitations"] = _decode_json_list(d.get("limitations"))
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
                average_quality_delta, base_confidence_score, confidence_score,
                recommendation, rationale,
                avg_current_quality, avg_simulated_quality, avg_latency_delta_ms,
                records_analyzed, failed_replays,
                evidence_level, evidence_summary, validation_status, limitations,
                evidence_coverage_pct, evidence_counts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                result.base_confidence_score,
                result.confidence_score,
                result.recommendation,
                result.rationale,
                result.avg_current_quality,
                result.avg_simulated_quality,
                result.avg_latency_delta_ms,
                result.records_analyzed,
                result.failed_replays,
                result.evidence_level.value,
                result.evidence_summary,
                result.validation_status.value,
                json.dumps(result.limitations),
                result.evidence_coverage_pct,
                json.dumps(result.evidence_counts),
            ),
        )
