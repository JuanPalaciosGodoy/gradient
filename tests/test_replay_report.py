from datetime import datetime, timezone

import pytest

from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.providers.base import GenerationProvider, ProviderResponse
from app.providers.fake import FakeProvider
from app.replay.replay_models import ReplayCandidate, ReplayRequest, ReplayResult
from app.replay.replay_report import build_replay_report, ModelReplaySummary, ReplayReport
from app.replay.replay_runner import build_replay_requests, ReplayRunner
from app.schemas import TaskType, UsageRecord


def _record(prompt: str, cost: float = 0.10) -> UsageRecord:
    return UsageRecord(
        prompt=prompt,
        response="Original response",
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        model="gpt-4o",
        cost=cost,
        task_type=TaskType.SUMMARIZATION,
    )


@pytest.fixture
def two_candidates():
    return [
        ReplayCandidate(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            model_group="cheap",
            estimated_input_cost_per_1k_tokens=0.00025,
            estimated_output_cost_per_1k_tokens=0.00125,
        ),
        ReplayCandidate(
            provider="google",
            model="gemini-1.5-flash",
            model_group="cheap",
            estimated_input_cost_per_1k_tokens=0.000075,
            estimated_output_cost_per_1k_tokens=0.0003,
        ),
    ]


@pytest.fixture
def replay_run_id():
    return "test-replay-run-001"


@pytest.fixture
def report(two_candidates, replay_run_id):
    records = [_record("Summarize Q3 earnings", cost=0.10) for _ in range(5)]
    runner = ReplayRunner(provider=FakeProvider(), evaluator=HeuristicEvaluator())
    requests = build_replay_requests(records)
    results = runner.run(requests, two_candidates)
    return build_replay_report(replay_run_id, requests, results, date_range_days=30)


def test_report_has_correct_request_count(report):
    assert report.total_requests == 5


def test_report_has_correct_result_count(report, two_candidates):
    # 5 requests × 2 candidates = 10 results
    assert report.total_results == 10


def test_report_has_no_errors_for_fake_provider(report):
    assert report.error_count == 0


def test_report_has_two_model_summaries(report, two_candidates):
    assert len(report.model_summaries) == len(two_candidates)


def test_model_summaries_cover_both_candidates(report):
    models = {s.model for s in report.model_summaries}
    assert "claude-3-haiku-20240307" in models
    assert "gemini-1.5-flash" in models


def test_model_summaries_sorted_by_savings_descending(report):
    savings = [s.estimated_annual_savings for s in report.model_summaries]
    assert savings == sorted(savings, reverse=True)


def test_model_summary_avg_quality_in_range(report):
    for s in report.model_summaries:
        assert 0.0 <= s.avg_quality_score <= 1.0


def test_model_summary_avg_latency_positive(report):
    for s in report.model_summaries:
        assert s.avg_latency_ms > 0


def test_report_generated_at_is_recent(report):
    now = datetime.now(timezone.utc)
    assert abs((now - report.generated_at).total_seconds()) < 5


def test_replay_run_id_is_preserved(report, replay_run_id):
    assert report.replay_run_id == replay_run_id


def test_model_summary_has_success_count(report):
    for s in report.model_summaries:
        assert s.success_count >= 0


def test_model_summary_has_coverage_rate(report):
    for s in report.model_summaries:
        assert 0.0 <= s.coverage_rate <= 1.0


def test_full_success_coverage_rate_is_one(report):
    # FakeProvider never fails, so coverage_rate should be 1.0 for all candidates
    for s in report.model_summaries:
        assert s.coverage_rate == 1.0


def test_failed_candidate_has_zero_savings():
    """A candidate that fails all requests must not report artificial savings."""

    class AlwaysFailProvider(GenerationProvider):
        def generate(self, prompt: str, model: str) -> ProviderResponse:
            raise RuntimeError("always fails")

    requests = [
        ReplayRequest(
            original_record_id=f"rec-{i}",
            prompt=f"Prompt {i}",
            original_response="Some response",
            original_model="gpt-4o",
            original_cost=1.0,
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        for i in range(3)
    ]
    candidate = ReplayCandidate(
        provider="broken", model="broken-model", model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.001,
        estimated_output_cost_per_1k_tokens=0.002,
    )
    runner = ReplayRunner(provider=AlwaysFailProvider(), evaluator=HeuristicEvaluator())
    results = runner.run(requests, [candidate])
    report = build_replay_report("run-001", requests, results, date_range_days=30)

    assert len(report.model_summaries) == 1
    summary = report.model_summaries[0]
    assert summary.success_count == 0
    assert summary.error_count == 3
    assert summary.coverage_rate == 0.0
    assert summary.estimated_annual_savings == 0.0


def test_failed_candidate_not_ranked_above_successful():
    """A failed candidate should not appear above a healthy one in the savings ranking."""

    class SelectiveProvider(GenerationProvider):
        def generate(self, prompt: str, model: str) -> ProviderResponse:
            if model == "broken-model":
                raise RuntimeError("fails")
            return ProviderResponse(
                text="ok", latency_ms=100.0, input_tokens=5,
                output_tokens=2, estimated_cost=0.0001,
            )

    requests = [
        ReplayRequest(
            original_record_id=f"rec-{i}",
            prompt=f"Prompt {i}",
            original_response="Some response",
            original_model="gpt-4o",
            original_cost=1.0,
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        for i in range(3)
    ]
    candidates = [
        ReplayCandidate(
            provider="test", model="good-model", model_group="cheap",
            estimated_input_cost_per_1k_tokens=0.001,
            estimated_output_cost_per_1k_tokens=0.002,
        ),
        ReplayCandidate(
            provider="test", model="broken-model", model_group="cheap",
            estimated_input_cost_per_1k_tokens=0.001,
            estimated_output_cost_per_1k_tokens=0.002,
        ),
    ]
    runner = ReplayRunner(provider=SelectiveProvider(), evaluator=HeuristicEvaluator())
    results = runner.run(requests, candidates)
    report = build_replay_report("run-002", requests, results, date_range_days=30)

    # good-model should rank above broken-model (savings > 0 vs savings == 0)
    models_by_rank = [s.model for s in report.model_summaries]
    assert models_by_rank[0] == "good-model"


def test_report_and_summary_are_pydantic_models(report):
    assert isinstance(report, ReplayReport)
    assert isinstance(report.model_summaries[0], ModelReplaySummary)
    # Pydantic models support .model_dump()
    d = report.model_dump()
    assert "model_summaries" in d
