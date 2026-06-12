from datetime import datetime, timezone

import pytest

from app.replay.migration_simulator import simulate_migration
from app.replay.replay_models import MigrationScenario, MigrationSimulationResult
from app.replay.replay_store import get_migration_simulations, save_migration_simulation
from app.schemas import TaskType, UsageRecord


def _record(model: str, cost: float, task_type: TaskType = TaskType.SUMMARIZATION) -> UsageRecord:
    return UsageRecord(
        prompt="Summarize this document",
        response="Summary.",
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        model=model,
        cost=cost,
        task_type=task_type,
    )


@pytest.fixture
def gpt4o_records():
    return [_record("gpt-4o", 1.0) for _ in range(10)]


@pytest.fixture
def summarization_scenario():
    return MigrationScenario(
        scenario_name="gpt4o-to-haiku",
        source_model="gpt-4o",
        target_model="claude-3-haiku-20240307",
        task_types_included=["summarization"],
        migration_percentage=1.0,
    )


# ── Basic behavior ────────────────────────────────────────────────────────────

def test_basic_simulation_returns_result(gpt4o_records, summarization_scenario):
    result = simulate_migration(gpt4o_records, summarization_scenario, date_range_days=30)
    assert result.scenario_name == "gpt4o-to-haiku"
    assert result.source_model == "gpt-4o"
    assert result.target_model == "claude-3-haiku-20240307"


def test_full_migration_to_cheaper_model_produces_savings(gpt4o_records, summarization_scenario):
    result = simulate_migration(gpt4o_records, summarization_scenario, date_range_days=30)
    assert result.estimated_annual_savings > 0
    assert result.estimated_savings_pct > 0


def test_savings_pct_consistent_with_absolute_savings(gpt4o_records, summarization_scenario):
    result = simulate_migration(gpt4o_records, summarization_scenario, date_range_days=30)
    if result.current_annualized_cost > 0:
        expected_pct = result.estimated_annual_savings / result.current_annualized_cost * 100
        assert abs(result.estimated_savings_pct - expected_pct) < 0.1


def test_simulated_cost_less_than_current(gpt4o_records, summarization_scenario):
    result = simulate_migration(gpt4o_records, summarization_scenario, date_range_days=30)
    assert result.simulated_annualized_cost < result.current_annualized_cost


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_records_produce_zero_costs():
    scenario = MigrationScenario(
        scenario_name="empty",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=1.0,
    )
    result = simulate_migration([], scenario, date_range_days=30)
    assert result.current_annualized_cost == 0.0
    assert result.estimated_annual_savings == 0.0


def test_no_matching_records_produces_zero_savings():
    records = [_record("claude-3-5-sonnet-20241022", 1.0)]
    scenario = MigrationScenario(
        scenario_name="no-match",
        source_model="gpt-4o",  # different source
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=1.0,
    )
    result = simulate_migration(records, scenario, date_range_days=30)
    assert result.current_annualized_cost == 0.0


def test_partial_migration_saves_less_than_full():
    records = [_record("gpt-4o", 1.0) for _ in range(10)]
    full = MigrationScenario(
        scenario_name="full",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=1.0,
    )
    partial = MigrationScenario(
        scenario_name="partial",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=0.5,
    )
    full_result = simulate_migration(records, full, date_range_days=30)
    partial_result = simulate_migration(records, partial, date_range_days=30)
    assert full_result.estimated_annual_savings > partial_result.estimated_annual_savings


def test_zero_migration_percentage_saves_nothing():
    records = [_record("gpt-4o", 1.0)]
    scenario = MigrationScenario(
        scenario_name="zero",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=0.0,
    )
    result = simulate_migration(records, scenario, date_range_days=30)
    assert result.estimated_annual_savings == 0.0


def test_task_type_filter_restricts_matching():
    records = [
        _record("gpt-4o", 1.0, TaskType.SUMMARIZATION),
        _record("gpt-4o", 1.0, TaskType.CODING),  # should not match
    ]
    scenario = MigrationScenario(
        scenario_name="filtered",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=["summarization"],  # only summarization
        migration_percentage=1.0,
    )
    result = simulate_migration(records, scenario, date_range_days=30)
    # Only the summarization record contributes
    all_tasks_scenario = MigrationScenario(
        scenario_name="all-tasks",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=1.0,
    )
    all_tasks_result = simulate_migration(records, all_tasks_scenario, date_range_days=30)
    assert all_tasks_result.current_annualized_cost > result.current_annualized_cost


# ── Recommendation logic ──────────────────────────────────────────────────────

def test_recommendation_is_proceed_for_clear_savings():
    records = [_record("gpt-4o", 10.0) for _ in range(50)]  # many records → high confidence
    scenario = MigrationScenario(
        scenario_name="clear-savings",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=1.0,
    )
    result = simulate_migration(records, scenario, date_range_days=30)
    assert result.recommendation in {"proceed", "investigate"}


def test_confidence_score_in_range():
    records = [_record("gpt-4o", 1.0)]
    scenario = MigrationScenario(
        scenario_name="conf-test",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=1.0,
    )
    result = simulate_migration(records, scenario, date_range_days=30)
    assert 0.0 <= result.confidence_score <= 1.0


def test_rationale_mentions_replay():
    records = [_record("gpt-4o", 1.0)]
    scenario = MigrationScenario(
        scenario_name="rationale-test",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=1.0,
    )
    result = simulate_migration(records, scenario, date_range_days=30)
    assert "replay" in result.rationale.lower()


# ── Persistence ───────────────────────────────────────────────────────────────

def _sim_result(scenario_name: str = "test-scenario") -> MigrationSimulationResult:
    return MigrationSimulationResult(
        scenario_name=scenario_name,
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=10_000.0,
        simulated_annualized_cost=7_000.0,
        estimated_annual_savings=3_000.0,
        estimated_savings_pct=30.0,
        average_quality_delta=-0.01,
        confidence_score=0.80,
        recommendation="proceed",
        rationale="Run a replay to validate.",
    )


def test_save_migration_simulation_without_ids(db):
    save_migration_simulation(_sim_result("no-ids"))
    rows = get_migration_simulations("no-ids")
    assert len(rows) == 1
    assert rows[0]["audit_run_id"] is None
    assert rows[0]["replay_run_id"] is None


def test_save_migration_simulation_with_audit_run_id(db):
    save_migration_simulation(_sim_result("with-audit"), audit_run_id="audit-abc")
    rows = get_migration_simulations("with-audit")
    assert rows[0]["audit_run_id"] == "audit-abc"
    assert rows[0]["replay_run_id"] is None


def test_save_migration_simulation_with_both_ids(db):
    save_migration_simulation(
        _sim_result("with-both"),
        audit_run_id="audit-xyz",
        replay_run_id="replay-xyz",
    )
    rows = get_migration_simulations("with-both")
    assert rows[0]["audit_run_id"] == "audit-xyz"
    assert rows[0]["replay_run_id"] == "replay-xyz"


def test_save_migration_simulation_persists_core_fields(db):
    save_migration_simulation(_sim_result("core-fields"))
    rows = get_migration_simulations("core-fields")
    row = rows[0]
    assert row["source_model"] == "gpt-4o"
    assert row["target_model"] == "gpt-4o-mini"
    assert row["estimated_annual_savings"] == 3_000.0
    assert row["recommendation"] == "proceed"
