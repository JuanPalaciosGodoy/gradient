"""
Golden-path scenario test for Phase 1.

Verifies product-level behavior, not just field presence:

  Given a workload with summarization, classification, and research tasks
  all running on a frontier model (gpt-4o), the audit should:

  - Surface savings opportunities for summarization and classification.
  - Leave research alone (no automatic downgrade recommendation).
  - Include replay-validation language.
  - Produce a positive potential_annual_savings figure.
  - Render a Markdown memo with the required executive sections.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.audit.report_builder import build_report
from app.reports.markdown import render_markdown_report
from app.schemas import TaskType, UsageRecord


def _rec(prompt: str, task: TaskType, cost: float, days_ago: int = 0) -> UsageRecord:
    return UsageRecord(
        prompt=prompt,
        response="response",
        timestamp=datetime(2024, 3, 31, tzinfo=timezone.utc) - timedelta(days=days_ago),
        model="gpt-4o",
        cost=cost,
        task_type=task,
    )


@pytest.fixture(scope="module")
def scenario_report():
    records = [
        _rec("Summarize this earnings report",           TaskType.SUMMARIZATION, cost=10.0),
        _rec("Summarize the board minutes",              TaskType.SUMMARIZATION, cost=10.0, days_ago=30),
        _rec("Classify this support ticket",             TaskType.CLASSIFICATION, cost=5.0),
        _rec("Label this email by department",           TaskType.CLASSIFICATION, cost=5.0, days_ago=30),
        _rec("What are the tradeoffs of transformers?",  TaskType.RESEARCH, cost=4.0),
        _rec("Explain RLHF vs RLAIF",                   TaskType.RESEARCH, cost=4.0, days_ago=30),
    ]
    return build_report("golden-scenario", records)


@pytest.fixture(scope="module")
def scenario_markdown(scenario_report):
    return render_markdown_report(scenario_report)


# ── Opportunity logic ─────────────────────────────────────────────────────────

def test_summarization_opportunity_is_identified(scenario_report):
    task_types = [opp.task_type for opp in scenario_report.top_opportunities]
    assert "summarization" in task_types


def test_classification_opportunity_is_identified(scenario_report):
    task_types = [opp.task_type for opp in scenario_report.top_opportunities]
    assert "classification" in task_types


def test_research_does_not_get_downgrade_recommendation(scenario_report):
    task_types = [opp.task_type for opp in scenario_report.top_opportunities]
    assert "research" not in task_types


def test_summarization_recommends_balanced_tier(scenario_report):
    summ = next(o for o in scenario_report.top_opportunities if o.task_type == "summarization")
    assert summ.recommended_model_group == "balanced"


def test_classification_recommends_cheap_tier(scenario_report):
    clas = next(o for o in scenario_report.top_opportunities if o.task_type == "classification")
    assert clas.recommended_model_group == "cheap"


# ── Savings figures ───────────────────────────────────────────────────────────

def test_potential_annual_savings_is_positive(scenario_report):
    assert scenario_report.potential_annual_savings > 0


def test_savings_rate_consistent_with_annual_savings(scenario_report):
    if scenario_report.spend_summary.annualized_cost > 0:
        expected = scenario_report.potential_annual_savings / scenario_report.spend_summary.annualized_cost
        assert abs(scenario_report.savings_rate - expected) < 0.001


def test_classification_savings_exceed_research_savings(scenario_report):
    # Classification (70% rate) should produce higher savings than summarization (45%)
    summ = next(o for o in scenario_report.top_opportunities if o.task_type == "summarization")
    clas = next(o for o in scenario_report.top_opportunities if o.task_type == "classification")
    assert clas.estimated_savings_pct > summ.estimated_savings_pct


# ── Risk notes ────────────────────────────────────────────────────────────────

def test_replay_validation_risk_note_present(scenario_report):
    categories = [n.category for n in scenario_report.risk_notes]
    assert "replay_validation_required" in categories


def test_frontier_dependency_risk_note_present(scenario_report):
    # 100% of spend is on gpt-4o (frontier tier)
    categories = [n.category for n in scenario_report.risk_notes]
    assert "frontier_model_dependency" in categories


# ── Recommended next actions ──────────────────────────────────────────────────

def test_next_actions_is_non_empty(scenario_report):
    assert len(scenario_report.recommended_next_actions) > 0


def test_first_next_action_references_top_opportunity(scenario_report):
    top = scenario_report.top_opportunities[0]
    first_action = scenario_report.recommended_next_actions[0]
    assert top.recommended_model in first_action or top.task_type.replace("_", " ") in first_action


# ── Markdown executive memo ───────────────────────────────────────────────────

def test_markdown_has_executive_summary_section(scenario_markdown):
    assert "## Executive Summary" in scenario_markdown


def test_markdown_has_top_opportunities_section(scenario_markdown):
    assert "## Top Opportunities" in scenario_markdown


def test_markdown_has_recommended_next_actions_section(scenario_markdown):
    assert "## Recommended Next Actions" in scenario_markdown


def test_markdown_has_replay_language(scenario_markdown):
    assert "replay" in scenario_markdown.lower()


def test_markdown_has_dollar_figures(scenario_markdown):
    assert "Current Annual Spend" in scenario_markdown
    assert "Potential Annual Savings" in scenario_markdown
    assert "$" in scenario_markdown
