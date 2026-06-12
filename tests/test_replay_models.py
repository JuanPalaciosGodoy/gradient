from datetime import datetime, timezone

import pytest

from app.replay.replay_models import (
    MigrationScenario,
    MigrationSimulationResult,
    REPLAY_CANDIDATES,
    ReplayCandidate,
    ReplayRequest,
    ReplayResult,
    get_candidate,
)


# ── ReplayCandidate ───────────────────────────────────────────────────────────

def test_replay_candidate_creation():
    c = ReplayCandidate(
        provider="openai",
        model="gpt-4o-mini",
        model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    assert c.provider == "openai"
    assert c.model == "gpt-4o-mini"
    assert c.enabled is True


def test_replay_candidates_catalog_not_empty():
    assert len(REPLAY_CANDIDATES) > 0


def test_replay_candidates_have_required_fields():
    for c in REPLAY_CANDIDATES:
        assert c.provider
        assert c.model
        assert c.model_group in {"frontier", "balanced", "cheap", "open_source"}
        assert c.estimated_input_cost_per_1k_tokens >= 0
        assert c.estimated_output_cost_per_1k_tokens >= 0


def test_replay_candidate_rejects_negative_input_cost():
    with pytest.raises(Exception):
        ReplayCandidate(
            provider="openai", model="gpt-4o-mini", model_group="cheap",
            estimated_input_cost_per_1k_tokens=-0.001,
            estimated_output_cost_per_1k_tokens=0.0006,
        )


def test_replay_candidate_rejects_negative_output_cost():
    with pytest.raises(Exception):
        ReplayCandidate(
            provider="openai", model="gpt-4o-mini", model_group="cheap",
            estimated_input_cost_per_1k_tokens=0.00015,
            estimated_output_cost_per_1k_tokens=-0.001,
        )


def test_replay_candidate_rejects_blank_provider():
    with pytest.raises(Exception):
        ReplayCandidate(
            provider="   ", model="gpt-4o-mini", model_group="cheap",
            estimated_input_cost_per_1k_tokens=0.00015,
            estimated_output_cost_per_1k_tokens=0.0006,
        )


def test_replay_candidate_rejects_blank_model():
    with pytest.raises(Exception):
        ReplayCandidate(
            provider="openai", model="", model_group="cheap",
            estimated_input_cost_per_1k_tokens=0.00015,
            estimated_output_cost_per_1k_tokens=0.0006,
        )


def test_replay_candidate_rejects_blank_model_group():
    with pytest.raises(Exception):
        ReplayCandidate(
            provider="openai", model="gpt-4o-mini", model_group="",
            estimated_input_cost_per_1k_tokens=0.00015,
            estimated_output_cost_per_1k_tokens=0.0006,
        )


def test_replay_candidate_accepts_zero_cost():
    c = ReplayCandidate(
        provider="openai", model="gpt-4o-mini", model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.0,
        estimated_output_cost_per_1k_tokens=0.0,
    )
    assert c.estimated_input_cost_per_1k_tokens == 0.0


def test_catalog_covers_all_major_providers():
    providers = {c.provider for c in REPLAY_CANDIDATES}
    assert "openai" in providers
    assert "anthropic" in providers
    assert "google" in providers


def test_get_candidate_returns_correct_model():
    c = get_candidate("gpt-4o-mini")
    assert c is not None
    assert c.model == "gpt-4o-mini"


def test_get_candidate_returns_none_for_unknown():
    assert get_candidate("nonexistent-model-xyz") is None


# ── ReplayRequest ─────────────────────────────────────────────────────────────

def test_replay_request_creation():
    req = ReplayRequest(
        original_record_id="rec-001",
        prompt="Summarize this document",
        original_response="This document covers...",
        original_model="gpt-4o",
        original_cost=0.05,
        task_type="summarization",
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    assert req.original_record_id == "rec-001"
    assert req.original_model == "gpt-4o"
    assert req.task_type == "summarization"


def test_replay_request_task_type_is_optional():
    req = ReplayRequest(
        original_record_id="rec-002",
        prompt="Some prompt",
        original_response="Some response",
        original_model="gpt-4o",
        original_cost=0.01,
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    assert req.task_type is None


# ── ReplayResult ──────────────────────────────────────────────────────────────

def test_replay_result_creation():
    result = ReplayResult(
        replay_id="replay-001",
        original_record_id="rec-001",
        candidate_provider="anthropic",
        candidate_model="claude-3-haiku-20240307",
        candidate_response="A concise summary.",
        estimated_cost=0.001,
        latency_ms=450.0,
        quality_score=0.82,
        quality_method="heuristic",
    )
    assert result.error_message is None
    assert result.quality_score == 0.82


def test_replay_result_error_case():
    result = ReplayResult(
        replay_id="replay-002",
        original_record_id="rec-001",
        candidate_provider="openai",
        candidate_model="gpt-4o-mini",
        candidate_response="",
        estimated_cost=0.0,
        latency_ms=0.0,
        quality_score=0.0,
        quality_method="error",
        error_message="Connection timeout",
    )
    assert result.error_message == "Connection timeout"


# ── MigrationScenario ─────────────────────────────────────────────────────────

def test_migration_scenario_creation():
    scenario = MigrationScenario(
        scenario_name="summarization-downgrade",
        source_model="gpt-4o",
        target_model="claude-3-haiku-20240307",
        task_types_included=["summarization"],
        migration_percentage=0.80,
    )
    assert scenario.migration_percentage == 0.80


def test_migration_percentage_must_be_between_0_and_1():
    with pytest.raises(Exception):
        MigrationScenario(
            scenario_name="bad-scenario",
            source_model="gpt-4o",
            target_model="gpt-4o-mini",
            task_types_included=[],
            migration_percentage=1.5,
        )


def test_migration_percentage_zero_is_valid():
    s = MigrationScenario(
        scenario_name="no-migration",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=0.0,
    )
    assert s.migration_percentage == 0.0


def test_migration_percentage_one_is_valid():
    s = MigrationScenario(
        scenario_name="full-migration",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=[],
        migration_percentage=1.0,
    )
    assert s.migration_percentage == 1.0


# ── MigrationSimulationResult ─────────────────────────────────────────────────

def test_migration_simulation_result_creation():
    result = MigrationSimulationResult(
        scenario_name="test",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=100_000.0,
        simulated_annualized_cost=70_000.0,
        estimated_annual_savings=30_000.0,
        estimated_savings_pct=30.0,
        average_quality_delta=-0.01,
        confidence_score=0.80,
        recommendation="proceed",
        rationale="Savings look solid.",
    )
    assert result.estimated_annual_savings == 30_000.0
    assert result.recommendation == "proceed"
