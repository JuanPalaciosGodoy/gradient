from datetime import datetime, timedelta, timezone

import pytest

from app.audit.report_builder import build_report
from app.schemas import TaskType, UsageRecord


def make_records() -> list[UsageRecord]:
    return [
        UsageRecord(
            prompt="Summarize this quarterly earnings report",
            response="Revenue grew 12% YoY.",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=0.05,
        ),
        UsageRecord(
            prompt="Classify this support ticket",
            response="Category: billing",
            timestamp=datetime(2024, 3, 31, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=0.02,
        ),
    ]


def test_report_has_required_fields():
    report = build_report("test-run-id", make_records())
    assert report.audit_run_id == "test-run-id"
    assert report.spend_summary is not None
    assert isinstance(report.top_opportunities, list)
    assert isinstance(report.risk_notes, list)
    assert isinstance(report.recommended_next_actions, list)


def test_confidence_score_in_range():
    report = build_report("test-id", make_records())
    assert 0.0 <= report.confidence_score <= 1.0


def test_spend_total_is_correct():
    report = build_report("test-id", make_records())
    assert round(report.spend_summary.total_cost, 2) == 0.07


def test_tasks_are_classified_after_build():
    records = make_records()
    build_report("test-id", records)
    assert all(r.task_type is not None for r in records)
    assert records[0].task_type == TaskType.SUMMARIZATION
    assert records[1].task_type == TaskType.CLASSIFICATION


def test_opportunities_found_for_high_cost_model():
    report = build_report("test-id", make_records())
    assert len(report.top_opportunities) > 0
    assert any(opp.current_model == "gpt-4o" for opp in report.top_opportunities)


def test_report_generated_at_is_recent():
    report = build_report("test-id", make_records())
    now = datetime.now(timezone.utc)
    assert abs((now - report.generated_at).total_seconds()) < 5


def test_empty_records_returns_zero_spend():
    report = build_report("empty-id", [])
    assert report.spend_summary.total_cost == 0.0
    assert report.top_opportunities == []


# ── Report-level savings fields ─────────────────────────────────────────────

def test_report_has_potential_annual_savings():
    report = build_report("test-id", make_records())
    assert hasattr(report, "potential_annual_savings")
    assert report.potential_annual_savings >= 0.0


def test_report_has_savings_rate():
    report = build_report("test-id", make_records())
    assert hasattr(report, "savings_rate")
    assert 0.0 <= report.savings_rate <= 1.0


def test_potential_annual_savings_is_annualized():
    # Same two records as make_records but with a known 1-day spread so we
    # can reason about the annualization factor precisely.
    records = [
        UsageRecord(
            prompt="Summarize this report",
            response="Summary",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=1.00,
        ),
        UsageRecord(
            prompt="Summarize another report",
            response="Summary",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),  # 1-day spread
            model="gpt-4o",
            cost=1.00,
        ),
    ]
    report = build_report("annualized-test", records)
    # Over 1 day, $2 total → annualized = $730.  70% savings → $511.
    # potential_annual_savings must be >> 2.0 (the raw period cost).
    assert report.potential_annual_savings > 10.0


def test_potential_annual_savings_does_not_exceed_annualized_spend():
    report = build_report("test-id", make_records())
    assert report.potential_annual_savings <= report.spend_summary.annualized_cost + 0.01


def test_savings_rate_is_zero_when_no_records():
    report = build_report("empty-id", [])
    assert report.savings_rate == 0.0
    assert report.potential_annual_savings == 0.0


def test_savings_rate_consistent_with_annual_savings():
    report = build_report("test-id", make_records())
    if report.spend_summary.annualized_cost > 0:
        expected = report.potential_annual_savings / report.spend_summary.annualized_cost
        assert abs(report.savings_rate - expected) < 0.001


# ── Opportunity fields ───────────────────────────────────────────────────────

def test_opportunity_has_annualized_fields():
    report = build_report("test-id", make_records())
    for opp in report.top_opportunities:
        assert opp.current_spend >= 0
        assert opp.current_annualized_spend >= opp.current_spend
        assert opp.estimated_annual_savings > 0
        assert 0 < opp.estimated_savings_pct <= 100
        assert opp.recommended_action != ""


# ── New report-level fields ──────────────────────────────────────────────────

def test_executive_summary_is_present_and_non_empty():
    report = build_report("test-id", make_records())
    assert isinstance(report.executive_summary, str)
    assert len(report.executive_summary) > 50


def test_executive_summary_mentions_spend():
    report = build_report("test-id", make_records())
    # Should mention a dollar amount
    assert "$" in report.executive_summary


def test_executive_summary_no_opportunities_variant():
    records = [
        UsageRecord(
            prompt="Research the history of machine learning",
            response="...",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=0.10,
        )
    ]
    report = build_report("no-opps", records)
    assert "No optimization opportunities" in report.executive_summary


def test_current_annual_spend_matches_spend_summary():
    report = build_report("test-id", make_records())
    assert report.current_annual_spend == report.spend_summary.annualized_cost


def test_recommended_next_actions_is_list():
    report = build_report("test-id", make_records())
    assert isinstance(report.recommended_next_actions, list)
    assert len(report.recommended_next_actions) > 0


def test_report_always_has_replay_validation_risk_note():
    report = build_report("test-id", make_records())
    categories = [n.category for n in report.risk_notes]
    assert "replay_validation_required" in categories


def test_frontier_dependency_risk_note_triggered():
    records = [
        UsageRecord(
            prompt="Summarize this",
            response="summary",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=10.0,
            task_type=TaskType.SUMMARIZATION,
        )
    ]
    report = build_report("frontier-test", records)
    categories = [n.category for n in report.risk_notes]
    assert "frontier_model_dependency" in categories


def test_low_feedback_risk_note_triggered():
    records = make_records()  # records have no feedback
    report = build_report("feedback-test", records)
    categories = [n.category for n in report.risk_notes]
    assert "low_feedback_coverage" in categories


def test_opportunity_annual_savings_larger_than_period_savings():
    # Two records 10 days apart: annualized savings should dwarf period savings.
    records = [
        UsageRecord(
            prompt="Summarize this document",
            response="Summary",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=1.00,
        ),
        UsageRecord(
            prompt="Extract the key figures",
            response="Figures",
            timestamp=datetime(2024, 1, 10, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=1.00,
        ),
    ]
    report = build_report("test-opp", records)
    for opp in report.top_opportunities:
        assert opp.estimated_annual_savings > opp.current_spend


# ── model_concentration risk note ───────────────────────────────────────────

def test_model_concentration_risk_note_triggered():
    # All spend on a single model → should trigger model_concentration note
    records = [
        UsageRecord(
            prompt="Summarize this report",
            response="Summary",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=10.0,
        )
        for _ in range(5)
    ]
    report = build_report("conc-test", records)
    categories = [n.category for n in report.risk_notes]
    assert "model_concentration" in categories


def test_model_concentration_risk_note_not_triggered_when_spread():
    # Spend is evenly split across two models → should not trigger
    records = [
        UsageRecord(
            prompt="Summarize this",
            response="Summary",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=5.0,
        ),
        UsageRecord(
            prompt="Classify this ticket",
            response="Billing",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            model="claude-3-5-sonnet-20241022",
            cost=5.0,
        ),
    ]
    report = build_report("spread-test", records)
    categories = [n.category for n in report.risk_notes]
    assert "model_concentration" not in categories
