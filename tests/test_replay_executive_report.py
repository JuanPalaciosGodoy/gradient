"""Tests for build_executive_replay_report and render_replay_markdown_report."""
import pytest

from app.replay.replay_models import MigrationSimulationResult
from app.replay.replay_report import (
    ExecutiveReplayReport,
    ScenarioOutcome,
    _portfolio_savings,
    build_executive_replay_report,
    build_executive_replay_report_from_simulations,
)
from app.reports.replay_markdown import render_replay_markdown_report


def _make_sim(
    source: str, target: str, task_type: str,
    savings: float, recommendation: str = "migrate",
) -> MigrationSimulationResult:
    return MigrationSimulationResult(
        scenario_name=f"{source}→{target}:{task_type}",
        source_model=source,
        target_model=target,
        current_annualized_cost=savings * 2,
        simulated_annualized_cost=savings,
        estimated_annual_savings=savings,
        estimated_savings_pct=50.0,
        average_quality_delta=-0.01,
        confidence_score=0.75,
        recommendation=recommendation,
        rationale="test",
        records_analyzed=10,
    )


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
                 error_message: str | None = None) -> dict:
    return {
        "original_record_id": str(record_id),
        "candidate_model": candidate_model,
        "candidate_provider": "openai",
        "candidate_response": "candidate",
        "estimated_cost": estimated_cost,
        "latency_ms": 400.0,
        "quality_score": quality_score,
        "quality_method": "heuristic",
        "quality_explanation": "test",
        "quality_confidence": 0.75,
        "quality_flags": [],
        "error_message": error_message,
    }


def _build_report(records=None, results=None, date_range_days=30):
    if records is None:
        records = [_make_record(i) for i in range(1, 6)]
    if results is None:
        results = [_make_result(i) for i in range(1, 6)]
    return build_executive_replay_report("run-001", records, results, date_range_days)


# ── ExecutiveReplayReport structure ──────────────────────────────────────────

def test_executive_report_returns_correct_type():
    report = _build_report()
    assert isinstance(report, ExecutiveReplayReport)


def test_executive_report_has_replay_run_id():
    report = _build_report()
    assert report.replay_run_id == "run-001"


def test_executive_report_has_generated_at():
    report = _build_report()
    assert report.generated_at is not None


def test_executive_report_total_requests():
    records = [_make_record(i) for i in range(1, 4)]
    results = [_make_result(i) for i in range(1, 4)]
    report = build_executive_replay_report("run-x", records, results)
    assert report.total_requests == 3


def test_executive_report_total_results():
    records = [_make_record(1)]
    results = [_make_result(1), _make_result(1, candidate_model="claude-3-haiku-20240307")]
    report = build_executive_replay_report("run-x", records, results)
    assert report.total_results == 2


def test_executive_report_current_annualized_spend_positive():
    report = _build_report()
    assert report.current_annualized_spend > 0


def test_executive_report_estimated_savings_non_negative():
    report = _build_report()
    assert report.estimated_annual_savings >= 0


def test_executive_summary_is_non_empty():
    report = _build_report()
    assert len(report.executive_summary) > 20


def test_risk_notes_is_list():
    report = _build_report()
    assert isinstance(report.risk_notes, list)


def test_next_actions_is_non_empty_list():
    report = _build_report()
    assert isinstance(report.next_actions, list)
    assert len(report.next_actions) > 0


def test_recommended_migrations_are_only_migrate_and_pilot():
    report = _build_report()
    for s in report.recommended_migrations:
        assert s.recommendation in ("migrate", "controlled_pilot")


def test_do_not_migrate_contains_only_no_migration():
    report = _build_report()
    for s in report.do_not_migrate:
        assert s.recommendation == "no_migration"


def test_empty_replay_results_builds_report():
    records = [_make_record(1)]
    report = build_executive_replay_report("run-empty", records, [], 30)
    assert report.total_results == 0
    assert report.estimated_annual_savings == 0.0


def test_scenario_outcome_fields_present():
    report = _build_report()
    all_outcomes = report.recommended_migrations + report.do_not_migrate
    if all_outcomes:
        s = all_outcomes[0]
        assert isinstance(s, ScenarioOutcome)
        assert isinstance(s.source_model, str)
        assert isinstance(s.target_model, str)
        assert isinstance(s.task_type, str)
        assert isinstance(s.confidence_pct, int)
        assert 0 <= s.confidence_pct <= 100


def test_low_quality_score_triggers_do_not_migrate():
    # quality_score = 0.01 → 99% quality loss → no_migration
    records = [_make_record(i) for i in range(1, 10)]
    results = [_make_result(i, quality_score=0.01, estimated_cost=0.001) for i in range(1, 10)]
    report = build_executive_replay_report("run-bad", records, results, 30)
    assert len(report.do_not_migrate) > 0


# ── Markdown renderer ─────────────────────────────────────────────────────────

def test_markdown_has_title():
    report = _build_report()
    md = render_replay_markdown_report(report)
    assert "# Gradient Replay Analysis" in md


def test_markdown_has_executive_summary_section():
    report = _build_report()
    md = render_replay_markdown_report(report)
    assert "## Executive Summary" in md


def test_markdown_has_recommended_migrations_section():
    report = _build_report()
    md = render_replay_markdown_report(report)
    assert "## Recommended Migrations" in md


def test_markdown_has_do_not_migrate_section():
    report = _build_report()
    md = render_replay_markdown_report(report)
    assert "## Do Not Migrate" in md


def test_markdown_has_risk_notes_section():
    report = _build_report()
    md = render_replay_markdown_report(report)
    assert "## Risk Notes" in md


def test_markdown_has_next_actions_section():
    report = _build_report()
    md = render_replay_markdown_report(report)
    assert "## Next Actions" in md


def test_markdown_has_dollar_amounts():
    report = _build_report()
    md = render_replay_markdown_report(report)
    assert "$" in md


def test_markdown_has_numbered_actions():
    report = _build_report()
    md = render_replay_markdown_report(report)
    assert "1." in md


def test_markdown_contains_generated_date():
    report = _build_report()
    md = render_replay_markdown_report(report)
    assert "_Generated:" in md


# ── _portfolio_savings ────────────────────────────────────────────────────────

def test_portfolio_savings_same_source_task_no_double_count():
    """Two targets for same (source, task_type) → only best savings count."""
    sims = [
        _make_sim("gpt-4o", "gpt-4o-mini", "summarization", savings=500.0),
        _make_sim("gpt-4o", "claude-3-haiku-20240307", "summarization", savings=400.0),
    ]
    assert _portfolio_savings(sims) == 500.0  # not 900.0


def test_portfolio_savings_different_tasks_are_additive():
    """Different task types under same source → savings add up."""
    sims = [
        _make_sim("gpt-4o", "gpt-4o-mini", "summarization", savings=500.0),
        _make_sim("gpt-4o", "gpt-4o-mini", "classification", savings=300.0),
    ]
    assert _portfolio_savings(sims) == 800.0


def test_portfolio_savings_excludes_non_recommended():
    """no_migration and hold do not contribute."""
    sims = [
        _make_sim("gpt-4o", "gpt-4o-mini", "summarization", savings=500.0, recommendation="no_migration"),
        _make_sim("gpt-4o", "gpt-4o-mini", "classification", savings=300.0, recommendation="hold"),
    ]
    assert _portfolio_savings(sims) == 0.0


def test_portfolio_savings_controlled_pilot_included():
    """controlled_pilot scenarios contribute to portfolio."""
    sims = [
        _make_sim("gpt-4o", "gpt-4o-mini", "research", savings=200.0, recommendation="controlled_pilot"),
    ]
    assert _portfolio_savings(sims) == 200.0


def test_executive_report_portfolio_savings_no_double_counting():
    """Integration: two competing targets for same slot → only best counted."""
    records = [_make_record(1, model="gpt-4o", cost=1.0, task_type="summarization")]
    results = [
        _make_result(1, candidate_model="gpt-4o-mini", quality_score=0.98, estimated_cost=0.05),
        _make_result(1, candidate_model="claude-3-haiku-20240307", quality_score=0.97, estimated_cost=0.06),
    ]
    report = build_executive_replay_report("run-portfolio", records, results, 30)
    # Only the best scenario for (gpt-4o, summarization) counts
    # gpt-4o-mini savings ≈ (1.0 - 0.05) * (365/30) ≈ 11.56
    # haiku savings ≈ (1.0 - 0.06) * (365/30) ≈ 11.44
    # total should not exceed ~11.6 (not doubled to ~23)
    max_single = (1.0 - 0.05) * (365 / 30)
    assert report.estimated_annual_savings <= max_single + 0.5


# ── build_executive_replay_report_from_simulations ───────────────────────────

def test_from_simulations_returns_correct_type():
    sims = [_make_sim("gpt-4o", "gpt-4o-mini", "summarization", savings=500.0)]
    report = build_executive_replay_report_from_simulations(
        replay_run_id="run-db", simulations=sims,
        current_annualized_spend=1000.0, total_requests=5, total_results=30,
    )
    assert isinstance(report, ExecutiveReplayReport)
    assert report.replay_run_id == "run-db"


def test_from_simulations_uses_portfolio_savings():
    sims = [
        _make_sim("gpt-4o", "gpt-4o-mini", "summarization", savings=500.0),
        _make_sim("gpt-4o", "claude-3-haiku-20240307", "summarization", savings=400.0),
    ]
    report = build_executive_replay_report_from_simulations(
        replay_run_id="run-portfolio", simulations=sims,
        current_annualized_spend=2000.0, total_requests=10, total_results=20,
    )
    assert report.estimated_annual_savings == 500.0  # portfolio, not 900


def test_from_simulations_empty_returns_zero_savings():
    report = build_executive_replay_report_from_simulations(
        replay_run_id="run-empty", simulations=[],
        current_annualized_spend=1000.0, total_requests=0, total_results=0,
    )
    assert report.estimated_annual_savings == 0.0
