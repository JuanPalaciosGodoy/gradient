from datetime import datetime, timedelta, timezone

import pytest

from app.audit.opportunity_analyzer import find_opportunities
from app.schemas import TaskType, UsageRecord


def make_record(
    task_type: TaskType,
    model: str,
    cost: float = 1.0,
    days_ago: int = 0,
) -> UsageRecord:
    return UsageRecord(
        prompt="test prompt",
        response="test response",
        timestamp=datetime(2024, 3, 31, tzinfo=timezone.utc) - timedelta(days=days_ago),
        model=model,
        cost=cost,
        task_type=task_type,
    )


# ── No opportunity cases ──────────────────────────────────────────────────────

def test_no_opportunities_for_research_on_frontier():
    records = [make_record(TaskType.RESEARCH, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert opps == []


def test_no_opportunities_for_coding_on_frontier():
    records = [make_record(TaskType.CODING, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert opps == []


def test_no_opportunities_when_already_on_cheap_model():
    # gpt-4o-mini is "cheap"; classification target is also "cheap" → no downgrade possible
    records = [make_record(TaskType.CLASSIFICATION, "gpt-4o-mini")]
    opps = find_opportunities(records, date_range_days=30)
    assert opps == []


def test_no_opportunities_when_already_on_target_tier():
    # claude-3-5-sonnet is "balanced"; summarization target is "balanced" → no downgrade
    records = [make_record(TaskType.SUMMARIZATION, "claude-3-5-sonnet-20241022")]
    opps = find_opportunities(records, date_range_days=30)
    assert opps == []


def test_other_task_type_produces_no_opportunity():
    records = [make_record(TaskType.OTHER, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert opps == []


# ── Opportunity generation ────────────────────────────────────────────────────

def test_summarization_on_frontier_generates_opportunity():
    records = [make_record(TaskType.SUMMARIZATION, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert len(opps) == 1
    assert opps[0].task_type == "summarization"


def test_classification_on_frontier_generates_opportunity():
    records = [make_record(TaskType.CLASSIFICATION, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert len(opps) == 1
    assert opps[0].task_type == "classification"


def test_extraction_on_frontier_generates_opportunity():
    records = [make_record(TaskType.EXTRACTION, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert len(opps) == 1


def test_customer_support_on_frontier_generates_opportunity():
    records = [make_record(TaskType.CUSTOMER_SUPPORT, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert len(opps) == 1


# ── Recommended model group ───────────────────────────────────────────────────

def test_summarization_recommends_balanced_tier():
    records = [make_record(TaskType.SUMMARIZATION, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert opps[0].recommended_model_group == "balanced"


def test_classification_recommends_cheap_tier():
    records = [make_record(TaskType.CLASSIFICATION, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert opps[0].recommended_model_group == "cheap"


def test_extraction_recommends_cheap_tier():
    records = [make_record(TaskType.EXTRACTION, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert opps[0].recommended_model_group == "cheap"


def test_customer_support_recommends_balanced_tier():
    records = [make_record(TaskType.CUSTOMER_SUPPORT, "gpt-4o")]
    opps = find_opportunities(records, date_range_days=30)
    assert opps[0].recommended_model_group == "balanced"


# ── Per-task savings rates ────────────────────────────────────────────────────

def test_classification_savings_rate_higher_than_summarization():
    r_summ = [make_record(TaskType.SUMMARIZATION, "gpt-4o", cost=1.0)]
    r_clas = [make_record(TaskType.CLASSIFICATION, "gpt-4o", cost=1.0)]
    summ_opps = find_opportunities(r_summ, date_range_days=30)
    clas_opps = find_opportunities(r_clas, date_range_days=30)
    assert clas_opps[0].estimated_savings_pct > summ_opps[0].estimated_savings_pct


def test_customer_support_savings_rate_most_conservative():
    records = [make_record(TaskType.CUSTOMER_SUPPORT, "gpt-4o", cost=1.0)]
    opps = find_opportunities(records, date_range_days=30)
    assert opps[0].estimated_savings_pct <= 45.0  # more conservative than summarization


# ── Annualization ─────────────────────────────────────────────────────────────

def test_annual_savings_larger_than_period_spend():
    records = [make_record(TaskType.SUMMARIZATION, "gpt-4o", cost=1.0)]
    opps = find_opportunities(records, date_range_days=10)
    assert opps[0].estimated_annual_savings > 1.0


def test_single_day_denominator_prevents_infinite_savings():
    records = [make_record(TaskType.CLASSIFICATION, "gpt-4o", cost=0.01)]
    opps = find_opportunities(records, date_range_days=0)  # 0 → clamped to 1
    assert opps[0].estimated_annual_savings < 10_000  # should be finite and reasonable


# ── Ordering ──────────────────────────────────────────────────────────────────

def test_opportunities_sorted_by_annual_savings_descending():
    records = [
        make_record(TaskType.SUMMARIZATION, "gpt-4o", cost=0.10),
        make_record(TaskType.CLASSIFICATION, "gpt-4o", cost=1.00),
    ]
    opps = find_opportunities(records, date_range_days=30)
    assert len(opps) == 2
    assert opps[0].estimated_annual_savings >= opps[1].estimated_annual_savings


# ── Required fields ───────────────────────────────────────────────────────────

def test_all_required_opportunity_fields_present():
    records = [make_record(TaskType.SUMMARIZATION, "gpt-4o", cost=1.0)]
    opp = find_opportunities(records, date_range_days=30)[0]
    assert opp.task_type != ""
    assert opp.current_model != ""
    assert opp.recommended_model != ""
    assert opp.recommended_model_group != ""
    assert opp.current_spend > 0
    assert opp.current_annualized_spend > 0
    assert opp.estimated_annual_savings > 0
    assert 0 < opp.estimated_savings_pct <= 100
    assert 0 < opp.confidence <= 1
    assert opp.rationale != ""
    assert opp.recommended_action != ""
