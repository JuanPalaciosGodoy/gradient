"""Tests for evidence-based migration simulation (simulate_from_replay_data)."""
from datetime import datetime, timezone

import pytest

from app.replay.migration_simulator import (
    _compute_replay_confidence,
    _recommend_from_replay,
    calculate_date_range_days,
    simulate_from_replay_data,
)
from app.replay.replay_models import MigrationSimulationResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_record(record_id: int, model: str = "gpt-4o", cost: float = 0.10,
                 task_type: str = "summarization") -> dict:
    return {
        "id": record_id,
        "prompt": f"prompt {record_id}",
        "response": f"response {record_id}",
        "timestamp": "2024-01-15 09:00:00",
        "model": model,
        "cost": cost,
        "task_type": task_type,
        "feedback": None,
    }


def _make_result(record_id: int, candidate_model: str = "gpt-4o-mini",
                 quality_score: float = 0.85, estimated_cost: float = 0.02,
                 latency_ms: float = 400.0, quality_confidence: float = 0.75,
                 error_message: str | None = None) -> dict:
    return {
        "original_record_id": str(record_id),
        "candidate_model": candidate_model,
        "candidate_provider": "openai",
        "candidate_response": "candidate response",
        "estimated_cost": estimated_cost,
        "latency_ms": latency_ms,
        "quality_score": quality_score,
        "quality_method": "heuristic",
        "quality_explanation": "test",
        "quality_confidence": quality_confidence,
        "quality_flags": [],
        "error_message": error_message,
    }


# ── simulate_from_replay_data ─────────────────────────────────────────────────

def test_simulate_empty_records_returns_empty():
    assert simulate_from_replay_data([], []) == []


def test_simulate_no_matching_records_returns_empty():
    records = [_make_record(1)]
    results = [_make_result(99)]  # mismatched id
    assert simulate_from_replay_data(records, results) == []


def test_simulate_groups_by_source_target_task():
    records = [
        _make_record(1, model="gpt-4o", task_type="summarization"),
        _make_record(2, model="gpt-4o", task_type="classification"),
    ]
    results = [
        _make_result(1, candidate_model="gpt-4o-mini"),
        _make_result(2, candidate_model="gpt-4o-mini"),
    ]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    keys = {(s.source_model, s.target_model, s.scenario_name.split(":")[-1]) for s in sims}
    assert ("gpt-4o", "gpt-4o-mini", "summarization") in keys
    assert ("gpt-4o", "gpt-4o-mini", "classification") in keys


def test_simulate_computes_annualized_cost():
    # 1 record, cost=0.10 over 30 days → annualized = 0.10 * (365/30) ≈ 1.217
    records = [_make_record(1, cost=0.10)]
    results = [_make_result(1, estimated_cost=0.01)]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    assert len(sims) == 1
    expected_current = round(0.10 * (365 / 30), 2)
    assert abs(sims[0].current_annualized_cost - expected_current) < 0.01


def test_simulate_excludes_failed_from_cost():
    records = [
        _make_record(1, cost=0.10),
        _make_record(2, cost=0.10),
    ]
    results = [
        _make_result(1, estimated_cost=0.02),
        _make_result(2, error_message="timeout"),  # failed — must not inflate savings
    ]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    assert len(sims) == 1
    # Only record 1 counted in costs
    expected_current = round(0.10 * (365 / 30), 2)
    expected_simulated = round(0.02 * (365 / 30), 2)
    assert abs(sims[0].current_annualized_cost - expected_current) < 0.01
    assert abs(sims[0].simulated_annualized_cost - expected_simulated) < 0.01


def test_simulate_records_analyzed_is_success_count():
    records = [_make_record(i, cost=0.05) for i in range(1, 6)]
    results = [
        _make_result(1),
        _make_result(2),
        _make_result(3, error_message="err"),
        _make_result(4),
        _make_result(5, error_message="err"),
    ]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    assert sims[0].records_analyzed == 3


def test_simulate_failed_replays_counted():
    records = [_make_record(i, cost=0.05) for i in range(1, 4)]
    results = [
        _make_result(1),
        _make_result(2, error_message="err"),
        _make_result(3, error_message="err"),
    ]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    assert sims[0].failed_replays == 2


def test_simulate_quality_delta_is_candidate_minus_one():
    records = [_make_record(1)]
    results = [_make_result(1, quality_score=0.80)]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    # avg_current = 1.0, avg_simulated = 0.80, delta = -0.20
    assert abs(sims[0].average_quality_delta - (-0.20)) < 0.001
    assert abs(sims[0].avg_simulated_quality - 0.80) < 0.001
    assert abs(sims[0].avg_current_quality - 1.0) < 0.001


def test_simulate_sorts_by_savings_descending():
    records = [
        _make_record(1, model="gpt-4o", cost=1.00, task_type="summarization"),
        _make_record(2, model="gpt-4o", cost=0.10, task_type="classification"),
    ]
    results = [
        _make_result(1, estimated_cost=0.05),   # big savings
        _make_result(2, estimated_cost=0.09),   # small savings
    ]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    assert sims[0].estimated_annual_savings >= sims[1].estimated_annual_savings


def test_simulate_all_failed_produces_investigate():
    records = [_make_record(1)]
    results = [_make_result(1, error_message="timeout")]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    assert sims[0].recommendation == "investigate"
    assert sims[0].records_analyzed == 0
    assert sims[0].failed_replays == 1


def test_simulate_scenario_name_format():
    records = [_make_record(1, model="gpt-4o", task_type="extraction")]
    results = [_make_result(1, candidate_model="claude-3-haiku-20240307")]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    assert sims[0].scenario_name == "gpt-4o→claude-3-haiku-20240307:extraction"


def test_simulate_latency_populated():
    records = [_make_record(1)]
    results = [_make_result(1, latency_ms=350.0)]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    assert sims[0].avg_latency_delta_ms == 350.0


# ── _recommend_from_replay ────────────────────────────────────────────────────

def test_recommend_migrate_low_loss_high_savings():
    action, rationale = _recommend_from_replay(
        quality_loss_pct=1.0, savings_pct=25.0, confidence=0.8,
        task_type="summarization", records_analyzed=20,
    )
    assert action == "migrate"
    assert "migrat" in rationale.lower()


def test_recommend_controlled_pilot_moderate_loss():
    action, _ = _recommend_from_replay(
        quality_loss_pct=3.0, savings_pct=35.0, confidence=0.7,
        task_type="other", records_analyzed=20,
    )
    assert action == "controlled_pilot"


def test_recommend_no_migration_high_loss():
    action, _ = _recommend_from_replay(
        quality_loss_pct=10.0, savings_pct=50.0, confidence=0.9,
        task_type="other", records_analyzed=20,
    )
    assert action == "no_migration"


def test_recommend_hold_when_no_savings():
    action, rationale = _recommend_from_replay(
        quality_loss_pct=0.0, savings_pct=-5.0, confidence=0.9,
        task_type="other", records_analyzed=20,
    )
    assert action == "hold"
    assert "costs more" in rationale.lower() or "savings" in rationale.lower()


def test_recommend_investigate_when_insufficient_data():
    action, _ = _recommend_from_replay(
        quality_loss_pct=1.0, savings_pct=25.0, confidence=0.8,
        task_type="other", records_analyzed=2,
    )
    assert action == "investigate"


def test_recommend_conservative_for_coding():
    # coding: max quality loss for migrate = 1%, min savings = 30%
    # 1.5% loss with 25% savings → should NOT be "migrate"
    action, _ = _recommend_from_replay(
        quality_loss_pct=1.5, savings_pct=25.0, confidence=0.8,
        task_type="coding", records_analyzed=20,
    )
    assert action != "migrate"


def test_recommend_aggressive_for_classification():
    # classification: max quality loss for migrate = 4%, min savings = 15%
    # 3% loss with 20% savings → should be "migrate"
    action, _ = _recommend_from_replay(
        quality_loss_pct=3.0, savings_pct=20.0, confidence=0.8,
        task_type="classification", records_analyzed=20,
    )
    assert action == "migrate"


def test_recommend_no_migration_for_research_moderate_loss():
    # research: pilot threshold = 3%, so 4% loss → no_migration
    action, _ = _recommend_from_replay(
        quality_loss_pct=4.0, savings_pct=50.0, confidence=0.8,
        task_type="research", records_analyzed=20,
    )
    assert action == "no_migration"


# ── _compute_replay_confidence ────────────────────────────────────────────────

def test_confidence_high_volume_increases_score():
    low = _compute_replay_confidence(5, 0.75, 0.0, 0.05, "other")
    high = _compute_replay_confidence(200, 0.75, 0.0, 0.05, "other")
    assert high > low


def test_confidence_high_failure_rate_decreases_score():
    good = _compute_replay_confidence(50, 0.75, 0.0, 0.05, "other")
    bad = _compute_replay_confidence(50, 0.75, 0.80, 0.05, "other")
    assert bad < good


def test_confidence_high_risk_task_decreases_score():
    base = _compute_replay_confidence(50, 0.75, 0.0, 0.05, "other")
    risky = _compute_replay_confidence(50, 0.75, 0.0, 0.05, "coding")
    assert risky < base


def test_confidence_lower_risk_task_increases_score():
    base = _compute_replay_confidence(50, 0.75, 0.0, 0.05, "other")
    safe = _compute_replay_confidence(50, 0.75, 0.0, 0.05, "classification")
    assert safe >= base


def test_confidence_is_clamped_to_unit_range():
    score = _compute_replay_confidence(0, 0.0, 1.0, 0.5, "research")
    assert 0.0 <= score <= 1.0
    score = _compute_replay_confidence(1000, 1.0, 0.0, 0.0, "classification")
    assert 0.0 <= score <= 1.0


def test_simulation_result_confidence_in_unit_range():
    records = [_make_record(i, cost=0.01) for i in range(1, 11)]
    results = [_make_result(i) for i in range(1, 11)]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    for sim in sims:
        assert 0.0 <= sim.confidence_score <= 1.0


# ── calculate_date_range_days ─────────────────────────────────────────────────

def test_date_range_empty_returns_one():
    assert calculate_date_range_days([]) == 1


def test_date_range_single_timestamp_is_one():
    ts = datetime(2024, 1, 15, tzinfo=timezone.utc)
    assert calculate_date_range_days([ts]) == 1


def test_date_range_same_day_timestamps_is_one():
    ts1 = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 15, 17, 0, tzinfo=timezone.utc)
    assert calculate_date_range_days([ts1, ts2]) == 1


def test_date_range_adjacent_days_is_two():
    ts1 = datetime(2024, 1, 15, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 16, tzinfo=timezone.utc)
    assert calculate_date_range_days([ts1, ts2]) == 2


def test_date_range_thirty_day_span_is_thirty_one():
    ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 31, tzinfo=timezone.utc)
    assert calculate_date_range_days([ts1, ts2]) == 31


def test_date_range_always_positive():
    ts1 = datetime(2024, 3, 1, tzinfo=timezone.utc)
    ts2 = datetime(2024, 3, 1, tzinfo=timezone.utc)
    assert calculate_date_range_days([ts1, ts2]) >= 1


def test_simulation_savings_pct_reasonable():
    records = [_make_record(1, cost=0.10)]
    results = [_make_result(1, estimated_cost=0.02)]
    sims = simulate_from_replay_data(records, results, date_range_days=30)
    # 80% cost reduction
    assert abs(sims[0].estimated_savings_pct - 80.0) < 1.0
