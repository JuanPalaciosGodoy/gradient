"""Tests for replace_migration_simulations atomicity and DB persistence defaults."""
import json
import sqlite3
import uuid

import pytest

from app.config import settings
from app.database import get_connection, get_migration_simulations_for_replay_run, init_db
from app.replay.replay_models import MigrationSimulationResult
from app.replay.replay_store import (
    get_replay_results,
    get_simulations_for_replay_run,
    replace_migration_simulations,
    save_migration_simulation,
    save_replay_results,
    save_replay_run,
)
from app.schemas import EvidenceLevel, ValidationStatus


def _sim(scenario_name: str = "gpt-4o→gpt-4o-mini:summarization") -> MigrationSimulationResult:
    return MigrationSimulationResult(
        scenario_name=scenario_name,
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=100.0,
        simulated_annualized_cost=50.0,
        estimated_annual_savings=50.0,
        estimated_savings_pct=50.0,
        average_quality_delta=-0.01,
        base_confidence_score=0.75,
        confidence_score=0.52,  # adjust_confidence_for_evidence(0.75, HEURISTIC, NOT_VALIDATED)
        recommendation="migrate",
        rationale="test",
    )


def test_replace_simulations_replaces_prior_rows(db):
    save_migration_simulation(_sim("old"), replay_run_id="run-A")
    assert len(get_migration_simulations_for_replay_run("run-A")) == 1

    replace_migration_simulations("run-A", "audit-A", [_sim("new")])

    rows = get_migration_simulations_for_replay_run("run-A")
    assert len(rows) == 1
    assert rows[0]["scenario_name"] == "new"


def test_replace_simulations_idempotent_no_duplicates(db):
    replace_migration_simulations("run-B", "audit-B", [_sim()])
    count_1 = len(get_migration_simulations_for_replay_run("run-B"))

    replace_migration_simulations("run-B", "audit-B", [_sim()])
    count_2 = len(get_migration_simulations_for_replay_run("run-B"))

    assert count_1 == count_2 == 1


def test_replace_simulations_empty_clears_prior_rows(db):
    save_migration_simulation(_sim(), replay_run_id="run-C")
    assert len(get_migration_simulations_for_replay_run("run-C")) == 1

    replace_migration_simulations("run-C", "audit-C", [])
    assert len(get_migration_simulations_for_replay_run("run-C")) == 0


def test_replace_simulations_rollback_preserves_old_rows_on_failure(db, monkeypatch):
    """If the INSERT phase raises, the DELETE must be rolled back atomically."""
    from contextlib import contextmanager
    from app.config import settings
    import app.replay.replay_store as store_module

    save_migration_simulation(_sim("original"), replay_run_id="run-D")
    assert len(get_migration_simulations_for_replay_run("run-D")) == 1

    # Patch get_connection in replay_store's namespace with a wrapper whose
    # executemany always raises, while execute (used for DELETE) works normally.
    @contextmanager
    def failing_get_connection():
        real = sqlite3.connect(settings.database_path)
        real.row_factory = sqlite3.Row

        class BoomProxy:
            def execute(self, sql, params=None):
                return real.execute(sql, params) if params is not None else real.execute(sql)

            def executemany(self, sql, params):
                raise sqlite3.OperationalError("forced insert failure")

        try:
            yield BoomProxy()
            real.commit()
        except Exception:
            real.rollback()
            raise
        finally:
            real.close()

    monkeypatch.setattr(store_module, "get_connection", failing_get_connection)

    with pytest.raises(Exception):
        replace_migration_simulations("run-D", "audit-D", [_sim("replacement")])

    rows = get_migration_simulations_for_replay_run("run-D")
    assert len(rows) == 1, "Rollback should have restored the original row"
    assert rows[0]["scenario_name"] == "original"


def test_replace_simulations_multiple_rows_inserted(db):
    sims = [_sim(f"gpt-4o→gpt-4o-mini:{t}") for t in ("summarization", "classification", "coding")]
    replace_migration_simulations("run-E", "audit-E", sims)
    rows = get_migration_simulations_for_replay_run("run-E")
    assert len(rows) == 3
    names = {row["scenario_name"] for row in rows}
    assert names == {s.scenario_name for s in sims}


# ── DB persistence defaults ───────────────────────────────────────────────────

def _legacy_insert_replay_result(conn, replay_run_id: str, record_id: int, error: bool = False) -> None:
    """Insert a replay_results row WITHOUT evidence columns (simulates a pre-2.5 insert)."""
    # Use raw SQL that omits evidence_level / validation_status to exercise DB defaults.
    conn.execute(
        """INSERT INTO replay_results
           (replay_run_id, replay_id, original_record_id, candidate_provider,
            candidate_model, candidate_response, estimated_cost, latency_ms,
            quality_score, quality_method, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            replay_run_id,
            str(uuid.uuid4()),
            str(record_id),
            "openai",
            "gpt-4o-mini",
            "" if error else "Category: billing",
            0.002,
            250.0,
            0.0 if error else 0.92,
            "heuristic",
            "Provider error" if error else None,
        ),
    )


def test_legacy_replay_result_gets_conservative_evidence_defaults(db):
    """Rows inserted before evidence columns were added must default to heuristic/not_validated."""
    run_id = "legacy-run-1"
    save_replay_run(run_id, "audit-X", ["gpt-4o-mini"], record_count=1)

    with get_connection() as conn:
        _legacy_insert_replay_result(conn, run_id, record_id=1)

    rows = get_replay_results(run_id)
    assert len(rows) == 1
    row = rows[0]
    # Conservative defaults — not overclaiming observed_replay or evaluator_scored
    assert row["evidence_level"] == "heuristic"
    assert row["validation_status"] == "not_validated"


def test_legacy_error_replay_result_gets_conservative_evidence_defaults(db):
    """Error rows inserted without evidence columns must also default to heuristic/not_validated."""
    run_id = "legacy-run-2"
    save_replay_run(run_id, "audit-Y", ["gpt-4o-mini"], record_count=1)

    with get_connection() as conn:
        _legacy_insert_replay_result(conn, run_id, record_id=1, error=True)

    rows = get_replay_results(run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["evidence_level"] == "heuristic"
    assert row["validation_status"] == "not_validated"


def test_legacy_simulation_gets_conservative_evidence_defaults(db):
    """Simulations inserted before evidence columns must default to heuristic/not_validated."""
    run_id = "legacy-sim-run-1"
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO migration_simulations
               (created_at, audit_run_id, replay_run_id, scenario_name,
                source_model, target_model,
                current_annualized_cost, simulated_annualized_cost,
                estimated_annual_savings, estimated_savings_pct,
                average_quality_delta, confidence_score, recommendation, rationale)
               VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("audit-Z", run_id, "gpt-4o→gpt-4o-mini:classification",
             "gpt-4o", "gpt-4o-mini", 1000.0, 300.0, 700.0, 70.0, -0.01, 0.80, "migrate", "test"),
        )

    sims = get_simulations_for_replay_run(run_id)
    assert sims
    sim = sims[0]
    assert sim.evidence_level == EvidenceLevel.HEURISTIC
    assert sim.validation_status == ValidationStatus.NOT_VALIDATED


def test_explicit_evidence_fields_are_persisted_and_roundtripped(db):
    """Evidence fields and base/adjusted confidence must survive a save → load roundtrip."""
    from app.schemas import EvidenceLevel, ValidationStatus
    from app.utils.evidence import adjust_confidence_for_evidence

    raw_base = 0.88
    adj = adjust_confidence_for_evidence(raw_base, EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED)

    sim_with_evidence = MigrationSimulationResult(
        scenario_name="gpt-4o→gpt-4o-mini:summarization",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=500.0,
        simulated_annualized_cost=150.0,
        estimated_annual_savings=350.0,
        estimated_savings_pct=70.0,
        average_quality_delta=-0.005,
        base_confidence_score=raw_base,
        confidence_score=adj,
        recommendation="migrate",
        rationale="Strong replay evidence.",
        evidence_level=EvidenceLevel.OBSERVED_REPLAY,
        evidence_summary="50 records replayed with 0.92 average quality.",
        validation_status=ValidationStatus.EVALUATOR_SCORED,
        limitations=["Heuristic evaluator only."],
    )

    replace_migration_simulations("run-RT", "audit-RT", [sim_with_evidence])
    loaded = get_simulations_for_replay_run("run-RT")

    assert len(loaded) == 1
    s = loaded[0]
    assert s.evidence_level == EvidenceLevel.OBSERVED_REPLAY
    assert s.validation_status == ValidationStatus.EVALUATOR_SCORED
    assert "50 records" in s.evidence_summary
    assert "Heuristic evaluator only." in s.limitations
    assert s.base_confidence_score == raw_base
    assert s.confidence_score == adj


def test_new_replay_result_evidence_fields_roundtrip(db):
    """Replay results saved with explicit evidence fields must load back correctly."""
    from app.replay.replay_models import ReplayResult

    run_id = "rr-rt-1"
    save_replay_run(run_id, "audit-RT2", ["gpt-4o-mini"], record_count=1)

    result = ReplayResult(
        replay_id=str(uuid.uuid4()),
        original_record_id="42",
        candidate_provider="openai",
        candidate_model="gpt-4o-mini",
        candidate_response="Summary here.",
        estimated_cost=0.002,
        latency_ms=200.0,
        quality_score=0.91,
        quality_method="heuristic",
        quality_explanation="Matches reference.",
        quality_confidence=0.87,
        quality_flags=["high_confidence"],
        error_message=None,
        evidence_level=EvidenceLevel.OBSERVED_REPLAY,
        evidence_summary="Replayed gpt-4o-mini with heuristic evaluator.",
        validation_status=ValidationStatus.EVALUATOR_SCORED,
        limitations=["Heuristic evaluator only."],
        confidence_score=0.73,
    )
    save_replay_results(run_id, [result])

    rows = get_replay_results(run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["evidence_level"] == "observed_replay"
    assert row["validation_status"] == "evaluator_scored"
    assert row["limitations"] == ["Heuristic evaluator only."]


# ── Persistence-stability: stored confidence_score is authoritative ────────────

def test_new_row_roundtrip_preserves_exact_stored_confidence_score(db):
    """When base_confidence_score > 0, loading must return the stored confidence_score
    verbatim — not a freshly recomputed value (which would change if factors update)."""
    from app.utils.evidence import adjust_confidence_for_evidence

    raw_base = 0.75
    stored_adj = adjust_confidence_for_evidence(
        raw_base, EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED
    )
    # Verify the stored value would differ if recomputed with wrong factors
    assert stored_adj != raw_base

    sim = MigrationSimulationResult(
        scenario_name="gpt-4o→gpt-4o-mini:classification",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=800.0,
        simulated_annualized_cost=240.0,
        estimated_annual_savings=560.0,
        estimated_savings_pct=70.0,
        average_quality_delta=-0.01,
        base_confidence_score=raw_base,
        confidence_score=stored_adj,
        recommendation="migrate",
        rationale="test",
        evidence_level=EvidenceLevel.OBSERVED_REPLAY,
        validation_status=ValidationStatus.EVALUATOR_SCORED,
    )
    replace_migration_simulations("run-stable-1", "audit-stable-1", [sim])
    loaded = get_simulations_for_replay_run("run-stable-1")

    assert len(loaded) == 1
    s = loaded[0]
    assert s.base_confidence_score == raw_base
    # Must be exactly the value that was persisted, not a recomputed approximation
    assert s.confidence_score == stored_adj


def test_legacy_row_gets_adjusted_confidence_on_load(db):
    """A row without base_confidence_score (pre-Phase-2.5) must have confidence_score
    recomputed from its raw stored value via adjust_confidence_for_evidence."""
    from app.utils.evidence import adjust_confidence_for_evidence

    run_id = "legacy-stable-1"
    raw_stored = 0.80  # the pre-Phase-2.5 raw value in the confidence_score column

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO migration_simulations
               (created_at, audit_run_id, replay_run_id, scenario_name,
                source_model, target_model,
                current_annualized_cost, simulated_annualized_cost,
                estimated_annual_savings, estimated_savings_pct,
                average_quality_delta, confidence_score, recommendation, rationale,
                evidence_level, validation_status)
               VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("audit-legacy", run_id, "gpt-4o→gpt-4o-mini:classification",
             "gpt-4o", "gpt-4o-mini", 1000.0, 300.0, 700.0, 70.0, -0.01,
             raw_stored, "migrate", "test",
             "observed_replay", "evaluator_scored"),
        )

    loaded = get_simulations_for_replay_run(run_id)
    assert loaded
    s = loaded[0]
    # base falls back to the old raw value
    assert s.base_confidence_score == raw_stored
    # confidence_score is recomputed (evidence-adjusted) since no base was persisted
    expected = adjust_confidence_for_evidence(
        raw_stored, EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED
    )
    assert s.confidence_score == expected
    assert s.confidence_score < raw_stored  # adjustment always discounts


def test_stored_confidence_score_survives_factor_change(db, monkeypatch):
    """If adjust_confidence_for_evidence factors change after a row is written,
    the loaded confidence_score must still be the originally stored value."""
    from app.utils.evidence import adjust_confidence_for_evidence
    import app.replay.replay_store as store_module

    raw_base = 0.70
    original_adj = adjust_confidence_for_evidence(
        raw_base, EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED
    )

    sim = MigrationSimulationResult(
        scenario_name="gpt-4o→gpt-4o-mini:summarization",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=600.0,
        simulated_annualized_cost=180.0,
        estimated_annual_savings=420.0,
        estimated_savings_pct=70.0,
        average_quality_delta=-0.01,
        base_confidence_score=raw_base,
        confidence_score=original_adj,
        recommendation="migrate",
        rationale="test",
        evidence_level=EvidenceLevel.OBSERVED_REPLAY,
        validation_status=ValidationStatus.EVALUATOR_SCORED,
    )
    replace_migration_simulations("run-factor-1", "audit-factor-1", [sim])

    # Simulate a future factor change by monkeypatching the function
    def new_adjust(base, level, status):
        return round(base * 0.50, 4)  # drastically different factors

    monkeypatch.setattr(store_module, "adjust_confidence_for_evidence", new_adjust)

    loaded = get_simulations_for_replay_run("run-factor-1")
    assert loaded
    s = loaded[0]
    # Must be the original stored value, not affected by the new factors
    assert s.confidence_score == original_adj
    assert s.confidence_score != new_adjust(raw_base, EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED)
