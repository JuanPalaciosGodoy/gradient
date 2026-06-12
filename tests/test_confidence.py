from datetime import datetime, timedelta, timezone

import pytest

from app.audit.confidence import compute_confidence
from app.schemas import TaskType, UsageRecord


def make_record(
    days_ago: int = 0,
    feedback: str | None = None,
    task_type: TaskType | None = None,
    cost: float = 0.01,
) -> UsageRecord:
    return UsageRecord(
        prompt="test prompt",
        response="test response",
        timestamp=datetime(2024, 3, 31, tzinfo=timezone.utc) - timedelta(days=days_ago),
        model="gpt-4o",
        cost=cost,
        feedback=feedback,
        task_type=task_type,
    )


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_records_returns_zero():
    assert compute_confidence([]) == 0.0


def test_result_is_between_zero_and_one():
    records = [make_record(days_ago=i) for i in range(5)]
    score = compute_confidence(records)
    assert 0.0 <= score <= 1.0


# ── Floor and cap ─────────────────────────────────────────────────────────────

def test_floor_is_40_percent():
    # Minimal input: 1 record, same-day, no feedback, no task type
    records = [make_record()]
    assert compute_confidence(records) >= 0.40


def test_cap_is_95_percent():
    # Best-case input: 1000+ records, 90-day spread, good feedback, diverse tasks
    task_cycle = [
        TaskType.SUMMARIZATION, TaskType.CLASSIFICATION, TaskType.EXTRACTION,
        TaskType.RESEARCH, TaskType.CODING,
    ]
    records = [
        make_record(
            days_ago=i % 90,
            feedback="positive",
            task_type=task_cycle[i % len(task_cycle)],
        )
        for i in range(1100)
    ]
    assert compute_confidence(records) <= 0.95


# ── Volume ────────────────────────────────────────────────────────────────────

def test_more_records_yield_higher_confidence():
    few = [make_record(days_ago=i % 30) for i in range(5)]
    many = [make_record(days_ago=i % 30) for i in range(150)]
    assert compute_confidence(many) > compute_confidence(few)


# ── Date range ────────────────────────────────────────────────────────────────

def test_longer_date_range_increases_confidence():
    short = [make_record(days_ago=i) for i in range(7)]   # 6-day spread
    long_ = [make_record(days_ago=i) for i in range(95)]  # 94-day spread
    assert compute_confidence(long_) > compute_confidence(short)


# ── Feedback coverage ─────────────────────────────────────────────────────────

def test_feedback_coverage_increases_confidence():
    no_feedback = [make_record(days_ago=i % 30) for i in range(50)]
    with_feedback = [
        make_record(days_ago=i % 30, feedback="positive" if i % 4 == 0 else None)
        for i in range(50)
    ]
    assert compute_confidence(with_feedback) > compute_confidence(no_feedback)


# ── Task diversity ────────────────────────────────────────────────────────────

def test_diverse_task_types_increase_confidence():
    mono = [make_record(task_type=TaskType.SUMMARIZATION) for _ in range(30)]
    diverse = [
        make_record(
            task_type=[TaskType.SUMMARIZATION, TaskType.CLASSIFICATION,
                       TaskType.EXTRACTION, TaskType.CODING][i % 4]
        )
        for i in range(30)
    ]
    assert compute_confidence(diverse) > compute_confidence(mono)


def test_unclassified_tasks_do_not_count_as_diverse():
    other_records = [make_record(task_type=TaskType.OTHER) for _ in range(30)]
    score = compute_confidence(other_records)
    # Should not get the diversity bonus
    classified_records = [
        make_record(task_type=[TaskType.SUMMARIZATION, TaskType.CLASSIFICATION,
                                TaskType.EXTRACTION, TaskType.CODING][i % 4])
        for i in range(30)
    ]
    assert compute_confidence(classified_records) > score
