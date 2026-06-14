"""Tests for replay report executive summary evidence description updates."""
import pytest

from app.replay.replay_models import MigrationSimulationResult
from app.replay.replay_report import (
    _evidence_method_description,
    build_executive_replay_report_from_simulations,
)
from app.schemas import EvidenceLevel, ValidationStatus


def _sim(
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVED_REPLAY,
    recommendation: str = "migrate",
    savings: float = 10000.0,
) -> MigrationSimulationResult:
    return MigrationSimulationResult(
        scenario_name="gpt-4o→gpt-4o-mini:summarization",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=50000.0,
        simulated_annualized_cost=40000.0,
        estimated_annual_savings=savings,
        estimated_savings_pct=20.0,
        average_quality_delta=-0.02,
        base_confidence_score=0.80,
        confidence_score=0.72,
        recommendation=recommendation,
        rationale="Test.",
        records_analyzed=50,
        evidence_level=evidence_level,
        validation_status=ValidationStatus.EVALUATOR_SCORED,
    )


# ── _evidence_method_description ──────────────────────────────────────────────

def test_heuristic_description():
    desc = _evidence_method_description([_sim(EvidenceLevel.HEURISTIC)])
    assert "heuristic" in desc.lower() or "replay" in desc.lower()


def test_observed_replay_description_mentions_evaluator():
    desc = _evidence_method_description([_sim(EvidenceLevel.OBSERVED_REPLAY)])
    assert "heuristic" in desc.lower()
    assert "replay" in desc.lower()


def test_llm_judge_description():
    desc = _evidence_method_description([_sim(EvidenceLevel.LLM_JUDGE)])
    assert "llm judge" in desc.lower() or "judge" in desc.lower()


def test_human_reviewed_description_is_highest():
    desc = _evidence_method_description([_sim(EvidenceLevel.HUMAN_REVIEWED)])
    assert "human" in desc.lower()
    assert "highest" in desc.lower() or "reviewer" in desc.lower()


def test_human_reviewed_takes_precedence_over_observed_replay():
    sims = [_sim(EvidenceLevel.OBSERVED_REPLAY), _sim(EvidenceLevel.HUMAN_REVIEWED)]
    desc = _evidence_method_description(sims)
    assert "human" in desc.lower()


def test_llm_judge_takes_precedence_over_observed_replay():
    sims = [_sim(EvidenceLevel.OBSERVED_REPLAY), _sim(EvidenceLevel.LLM_JUDGE)]
    desc = _evidence_method_description(sims)
    assert "judge" in desc.lower()


# ── Executive summary includes evidence description ───────────────────────────

def test_executive_summary_contains_evidence_method():
    report = build_executive_replay_report_from_simulations(
        replay_run_id="test-run",
        simulations=[_sim(EvidenceLevel.OBSERVED_REPLAY)],
        current_annualized_spend=50000.0,
        total_requests=100,
        total_results=50,
    )
    assert "heuristic" in report.executive_summary.lower() or "replay" in report.executive_summary.lower()


def test_executive_summary_for_human_reviewed_mentions_human():
    report = build_executive_replay_report_from_simulations(
        replay_run_id="test-run",
        simulations=[_sim(EvidenceLevel.HUMAN_REVIEWED)],
        current_annualized_spend=50000.0,
        total_requests=100,
        total_results=50,
    )
    assert "human" in report.executive_summary.lower()


def test_executive_summary_for_llm_judge_mentions_judge():
    report = build_executive_replay_report_from_simulations(
        replay_run_id="test-run",
        simulations=[_sim(EvidenceLevel.LLM_JUDGE)],
        current_annualized_spend=50000.0,
        total_requests=100,
        total_results=50,
    )
    assert "judge" in report.executive_summary.lower()


# ── Report stores evidence level per scenario ─────────────────────────────────

def test_report_scenario_evidence_level_preserved():
    report = build_executive_replay_report_from_simulations(
        replay_run_id="test-run",
        simulations=[_sim(EvidenceLevel.LLM_JUDGE)],
        current_annualized_spend=50000.0,
        total_requests=100,
        total_results=50,
    )
    assert report.recommended_migrations[0].evidence_level == EvidenceLevel.LLM_JUDGE
