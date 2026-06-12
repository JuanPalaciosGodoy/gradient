from datetime import datetime, timedelta, timezone

import pytest

from app.audit.spend_analyzer import analyze_spend
from app.schemas import TaskType, UsageRecord


def make_record(cost: float, model: str = "gpt-4o", days_ago: int = 0) -> UsageRecord:
    return UsageRecord(
        prompt="test prompt",
        response="test response",
        timestamp=datetime(2024, 3, 31, tzinfo=timezone.utc) - timedelta(days=days_ago),
        model=model,
        cost=cost,
    )


def test_empty_records_returns_zero_summary():
    summary = analyze_spend([])
    assert summary.total_cost == 0.0
    assert summary.annualized_cost == 0.0
    assert summary.avg_cost_per_request == 0.0
    assert summary.date_range_days == 0


def test_total_cost():
    records = [make_record(0.10), make_record(0.20), make_record(0.30)]
    summary = analyze_spend(records)
    assert round(summary.total_cost, 2) == 0.60


def test_avg_cost_per_request():
    records = [make_record(0.10), make_record(0.30)]
    summary = analyze_spend(records)
    assert round(summary.avg_cost_per_request, 4) == 0.20


def test_model_breakdown():
    records = [
        make_record(1.0, model="gpt-4o"),
        make_record(1.0, model="gpt-4o"),
        make_record(0.5, model="gpt-4o-mini"),
    ]
    summary = analyze_spend(records)
    assert summary.model_breakdown["gpt-4o"] == 2.0
    assert summary.model_breakdown["gpt-4o-mini"] == 0.5


def test_annualization():
    # 30 records spanning 29 days
    records = [make_record(1.0, days_ago=i) for i in range(30)]
    summary = analyze_spend(records)
    assert summary.date_range_days == 29
    expected_annualized = 30.0 * (365 / 29)
    assert abs(summary.annualized_cost - expected_annualized) < 1.0


def test_daily_avg_spend():
    records = [make_record(1.0, days_ago=i) for i in range(10)]  # 9-day spread, $10 total
    summary = analyze_spend(records)
    assert summary.date_range_days == 9
    assert abs(summary.daily_avg_spend - 10.0 / 9) < 0.001


def test_empty_records_daily_avg_is_zero():
    summary = analyze_spend([])
    assert summary.daily_avg_spend == 0.0


def test_provider_breakdown_maps_known_models():
    records = [
        make_record(1.0, model="gpt-4o"),
        make_record(0.5, model="claude-3-5-sonnet-20241022"),
    ]
    summary = analyze_spend(records)
    assert "openai" in summary.provider_breakdown
    assert "anthropic" in summary.provider_breakdown
    assert abs(summary.provider_breakdown["openai"] - 1.0) < 0.001
    assert abs(summary.provider_breakdown["anthropic"] - 0.5) < 0.001


def test_provider_breakdown_sums_to_total():
    records = [
        make_record(1.0, model="gpt-4o"),
        make_record(0.5, model="gemini-1.5-flash"),
        make_record(0.25, model="claude-3-haiku-20240307"),
    ]
    summary = analyze_spend(records)
    provider_total = sum(summary.provider_breakdown.values())
    assert abs(provider_total - summary.total_cost) < 0.001


def test_task_breakdown_uses_task_type():
    records = [
        UsageRecord(
            prompt="test",
            response="test",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=1.0,
            task_type=TaskType.SUMMARIZATION,
        ),
        UsageRecord(
            prompt="test",
            response="test",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=0.5,
            task_type=TaskType.CODING,
        ),
    ]
    summary = analyze_spend(records)
    assert summary.task_breakdown["summarization"] == 1.0
    assert summary.task_breakdown["coding"] == 0.5


def test_top_cost_driving_task_types_sorted_descending():
    records = [
        UsageRecord(
            prompt="test",
            response="test",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=0.30,
            task_type=TaskType.SUMMARIZATION,
        ),
        UsageRecord(
            prompt="test",
            response="test",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=0.10,
            task_type=TaskType.CODING,
        ),
        UsageRecord(
            prompt="test",
            response="test",
            timestamp=datetime(2024, 1, 3, tzinfo=timezone.utc),
            model="gpt-4o",
            cost=0.20,
            task_type=TaskType.RESEARCH,
        ),
    ]
    summary = analyze_spend(records)
    costs = [item.cost for item in summary.top_cost_driving_task_types]
    assert costs == sorted(costs, reverse=True)
    assert summary.top_cost_driving_task_types[0].task_type == "summarization"


def test_top_cost_driving_task_types_empty_for_no_records():
    summary = analyze_spend([])
    assert summary.top_cost_driving_task_types == []
