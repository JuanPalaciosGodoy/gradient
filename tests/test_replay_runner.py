"""
Tests for the replay runner and fake provider.
Covers: request construction, multi-candidate runs, error isolation, persistence.
"""
import uuid
from datetime import datetime, timezone

import pytest

from app.evaluation.base import BaseEvaluator
from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.evaluation.quality_score import QualityEvaluation
from app.providers.base import GenerationProvider, ProviderResponse
from app.providers.fake import FakeProvider
from app.replay.replay_models import ReplayCandidate, ReplayRequest, REPLAY_CANDIDATES
from app.replay.replay_runner import build_replay_requests, build_replay_requests_from_rows, ReplayRunner
from app.replay.replay_store import _decode_quality_flags, get_replay_results, save_replay_results, save_replay_run
from app.schemas import TaskType, UsageRecord


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _record(prompt: str, model: str = "gpt-4o", cost: float = 0.05) -> UsageRecord:
    return UsageRecord(
        prompt=prompt,
        response="Original response text.",
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        model=model,
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
def runner():
    return ReplayRunner(provider=FakeProvider(), evaluator=HeuristicEvaluator())


# ── build_replay_requests ─────────────────────────────────────────────────────

def test_build_requests_returns_one_per_record():
    records = [_record("Prompt A"), _record("Prompt B")]
    requests = build_replay_requests(records)
    assert len(requests) == 2


def test_build_requests_preserves_prompt():
    records = [_record("Summarize this document")]
    requests = build_replay_requests(records)
    assert requests[0].prompt == "Summarize this document"


def test_build_requests_assigns_unique_ids():
    records = [_record("P1"), _record("P2")]
    requests = build_replay_requests(records)
    assert requests[0].original_record_id != requests[1].original_record_id


def test_build_requests_includes_task_type():
    records = [_record("P")]
    requests = build_replay_requests(records)
    assert requests[0].task_type == "summarization"


def test_build_requests_carries_feedback():
    record = UsageRecord(
        prompt="Summarize this",
        response="A summary.",
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        model="gpt-4o",
        cost=0.05,
        task_type=TaskType.SUMMARIZATION,
        feedback="positive",
    )
    requests = build_replay_requests([record])
    assert requests[0].feedback == "positive"


def test_build_requests_feedback_none_when_absent():
    requests = build_replay_requests([_record("P")])
    assert requests[0].feedback is None


# ── build_replay_requests_from_rows ──────────────────────────────────────────

def test_build_from_rows_uses_db_id():
    rows = [
        {
            "id": 42,
            "prompt": "Summarize this",
            "response": "A summary.",
            "model": "gpt-4o",
            "cost": 0.05,
            "task_type": "summarization",
            "timestamp": "2024-01-15T00:00:00+00:00",
        }
    ]
    requests = build_replay_requests_from_rows(rows)
    assert len(requests) == 1
    assert requests[0].original_record_id == "42"


def test_build_from_rows_returns_one_per_row():
    rows = [
        {
            "id": 1, "prompt": "P1", "response": "R1",
            "model": "gpt-4o", "cost": 0.01, "task_type": None,
            "timestamp": "2024-01-15T00:00:00+00:00",
        },
        {
            "id": 2, "prompt": "P2", "response": "R2",
            "model": "gpt-4o", "cost": 0.02, "task_type": None,
            "timestamp": "2024-01-15T00:00:00+00:00",
        },
    ]
    requests = build_replay_requests_from_rows(rows)
    assert len(requests) == 2


def test_build_from_rows_ids_are_distinct():
    rows = [
        {
            "id": 10, "prompt": "P1", "response": "R1",
            "model": "gpt-4o", "cost": 0.01, "task_type": None,
            "timestamp": "2024-01-15T00:00:00+00:00",
        },
        {
            "id": 20, "prompt": "P2", "response": "R2",
            "model": "gpt-4o", "cost": 0.02, "task_type": None,
            "timestamp": "2024-01-15T00:00:00+00:00",
        },
    ]
    requests = build_replay_requests_from_rows(rows)
    assert requests[0].original_record_id == "10"
    assert requests[1].original_record_id == "20"


def test_build_from_rows_preserves_task_type():
    rows = [
        {
            "id": 1, "prompt": "P", "response": "R",
            "model": "gpt-4o", "cost": 0.01, "task_type": "coding",
            "timestamp": "2024-01-15T00:00:00+00:00",
        }
    ]
    requests = build_replay_requests_from_rows(rows)
    assert requests[0].task_type == "coding"


def test_build_from_rows_carries_feedback():
    rows = [
        {
            "id": 1, "prompt": "P", "response": "R",
            "model": "gpt-4o", "cost": 0.01, "task_type": None,
            "feedback": "negative",
            "timestamp": "2024-01-15T00:00:00+00:00",
        }
    ]
    requests = build_replay_requests_from_rows(rows)
    assert requests[0].feedback == "negative"


def test_build_from_rows_feedback_none_when_absent():
    rows = [
        {
            "id": 1, "prompt": "P", "response": "R",
            "model": "gpt-4o", "cost": 0.01, "task_type": None,
            "timestamp": "2024-01-15T00:00:00+00:00",
        }
    ]
    requests = build_replay_requests_from_rows(rows)
    assert requests[0].feedback is None


# ── FakeProvider ──────────────────────────────────────────────────────────────

def test_fake_provider_returns_response():
    provider = FakeProvider()
    resp = provider.generate("Summarize this", "gpt-4o-mini")
    assert isinstance(resp.text, str)
    assert len(resp.text) > 0


def test_fake_provider_is_deterministic():
    provider = FakeProvider()
    r1 = provider.generate("Hello world", "gpt-4o")
    r2 = provider.generate("Hello world", "gpt-4o")
    assert r1.text == r2.text
    assert r1.latency_ms == r2.latency_ms
    assert r1.estimated_cost == r2.estimated_cost


def test_fake_provider_cost_is_positive():
    provider = FakeProvider()
    resp = provider.generate("A longer prompt to generate cost", "claude-3-5-sonnet-20241022")
    assert resp.estimated_cost > 0


def test_fake_provider_latency_is_positive():
    provider = FakeProvider()
    resp = provider.generate("Some prompt", "gpt-4o")
    assert resp.latency_ms > 0


def test_fake_provider_frontier_costs_more_than_cheap():
    provider = FakeProvider()
    frontier = provider.generate("Same prompt", "gpt-4o")
    cheap = provider.generate("Same prompt", "gpt-4o-mini")
    assert frontier.estimated_cost > cheap.estimated_cost


def test_fake_provider_works_for_unknown_model():
    provider = FakeProvider()
    resp = provider.generate("Prompt", "some-future-model-v3")
    assert resp.estimated_cost >= 0


# ── ReplayRunner ──────────────────────────────────────────────────────────────

def test_runner_returns_n_results_per_request(runner, two_candidates):
    requests = build_replay_requests([_record("P1"), _record("P2")])
    results = runner.run(requests, two_candidates)
    # 2 requests × 2 candidates = 4 results
    assert len(results) == 4


def test_runner_result_has_correct_provider(runner, two_candidates):
    requests = build_replay_requests([_record("P")])
    results = runner.run(requests, two_candidates)
    providers = {r.candidate_provider for r in results}
    assert "anthropic" in providers
    assert "google" in providers


def test_runner_result_quality_score_in_range(runner, two_candidates):
    requests = build_replay_requests([_record("Summarize this report")])
    results = runner.run(requests, two_candidates)
    for r in results:
        assert 0.0 <= r.quality_score <= 1.0


def test_runner_successful_result_has_no_error(runner, two_candidates):
    requests = build_replay_requests([_record("P")])
    results = runner.run(requests, two_candidates)
    for r in results:
        assert r.error_message is None


def test_disabled_candidate_is_skipped(runner):
    disabled = ReplayCandidate(
        provider="openai",
        model="gpt-4o-mini",
        model_group="cheap",
        enabled=False,
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    requests = build_replay_requests([_record("P")])
    results = runner.run(requests, [disabled])
    assert results == []


def test_failing_provider_does_not_crash_runner():
    """A provider that always raises should produce error results, not exceptions."""

    class BrokenProvider(GenerationProvider):
        def generate(self, prompt: str, model: str) -> ProviderResponse:
            raise RuntimeError("Provider unavailable")

    broken_runner = ReplayRunner(
        provider=BrokenProvider(),
        evaluator=HeuristicEvaluator(),
    )
    candidate = ReplayCandidate(
        provider="openai",
        model="gpt-4o-mini",
        model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    requests = build_replay_requests([_record("P1"), _record("P2")])
    results = broken_runner.run(requests, [candidate])
    assert len(results) == 2
    for r in results:
        assert r.error_message is not None
        assert r.quality_method == "error"


def test_mixed_providers_partial_failure():
    """One broken provider should not affect results from a healthy one."""

    class BrokenProvider(GenerationProvider):
        def generate(self, prompt: str, model: str) -> ProviderResponse:
            if "broken" in model:
                raise RuntimeError("broken")
            return ProviderResponse(
                text="ok", latency_ms=100.0, input_tokens=5,
                output_tokens=2, estimated_cost=0.001,
            )

    mixed_runner = ReplayRunner(
        provider=BrokenProvider(),
        evaluator=HeuristicEvaluator(),
    )
    candidates = [
        ReplayCandidate(
            provider="test", model="good-model",
            model_group="cheap",
            estimated_input_cost_per_1k_tokens=0.001,
            estimated_output_cost_per_1k_tokens=0.002,
        ),
        ReplayCandidate(
            provider="test", model="broken-model",
            model_group="cheap",
            estimated_input_cost_per_1k_tokens=0.001,
            estimated_output_cost_per_1k_tokens=0.002,
        ),
    ]
    requests = build_replay_requests([_record("P")])
    results = mixed_runner.run(requests, candidates)
    assert len(results) == 2
    good = next(r for r in results if r.candidate_model == "good-model")
    bad = next(r for r in results if r.candidate_model == "broken-model")
    assert good.error_message is None
    assert bad.error_message is not None


# ── Feedback reaches evaluator ────────────────────────────────────────────────

class _FeedbackCapturingEvaluator(BaseEvaluator):
    """Spy evaluator that records the feedback argument it received."""

    def __init__(self):
        self.last_feedback = "NOT_CALLED"

    def evaluate(self, prompt, original_response, candidate_response, task_type=None, feedback=None):
        self.last_feedback = feedback
        return QualityEvaluation(
            score=0.8, method="heuristic",
            explanation="spy", confidence=0.8, flags=[],
        )


def test_runner_passes_feedback_to_evaluator():
    spy = _FeedbackCapturingEvaluator()
    runner = ReplayRunner(provider=FakeProvider(), evaluator=spy)
    record = UsageRecord(
        prompt="Summarize this",
        response="A summary.",
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        model="gpt-4o",
        cost=0.05,
        task_type=TaskType.SUMMARIZATION,
        feedback="positive",
    )
    requests = build_replay_requests([record])
    candidate = ReplayCandidate(
        provider="openai", model="gpt-4o-mini", model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    runner.run(requests, [candidate])
    assert spy.last_feedback == "positive"


def test_runner_passes_none_feedback_when_absent():
    spy = _FeedbackCapturingEvaluator()
    runner = ReplayRunner(provider=FakeProvider(), evaluator=spy)
    requests = build_replay_requests([_record("P")])
    candidate = ReplayCandidate(
        provider="openai", model="gpt-4o-mini", model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    runner.run(requests, [candidate])
    assert spy.last_feedback is None


# ── Persistence ───────────────────────────────────────────────────────────────

def test_replay_results_are_persisted(db, runner, two_candidates):
    replay_run_id = str(uuid.uuid4())
    requests = build_replay_requests([_record("P")])
    results = runner.run(requests, two_candidates)

    save_replay_run(
        replay_run_id=replay_run_id,
        audit_run_id=None,
        candidate_models=[c.model for c in two_candidates],
        record_count=len(requests),
    )
    save_replay_results(replay_run_id, results)

    persisted = get_replay_results(replay_run_id)
    assert len(persisted) == len(results)


def test_replay_results_have_correct_run_id(db, runner, two_candidates):
    replay_run_id = str(uuid.uuid4())
    requests = build_replay_requests([_record("P")])
    results = runner.run(requests, two_candidates)

    save_replay_run(
        replay_run_id=replay_run_id,
        audit_run_id=None,
        candidate_models=[c.model for c in two_candidates],
        record_count=len(requests),
    )
    save_replay_results(replay_run_id, results)

    persisted = get_replay_results(replay_run_id)
    for row in persisted:
        assert row["replay_run_id"] == replay_run_id


def test_quality_flags_round_trip(db, runner, two_candidates):
    """quality_flags saved as JSON are decoded back to a Python list on read."""
    replay_run_id = str(uuid.uuid4())
    requests = build_replay_requests([_record("P")])
    results = runner.run(requests, two_candidates)

    save_replay_run(
        replay_run_id=replay_run_id,
        audit_run_id=None,
        candidate_models=[c.model for c in two_candidates],
        record_count=len(requests),
    )
    save_replay_results(replay_run_id, results)

    persisted = get_replay_results(replay_run_id)
    for row in persisted:
        assert isinstance(row["quality_flags"], list)


# ── _decode_quality_flags edge cases ─────────────────────────────────────────

def test_decode_flags_valid_list():
    assert _decode_quality_flags('["empty_response", "short_response"]') == [
        "empty_response", "short_response"
    ]


def test_decode_flags_null_returns_empty_list():
    assert _decode_quality_flags(None) == []


def test_decode_flags_malformed_json_returns_empty_list():
    assert _decode_quality_flags("{not valid json!!!}") == []


def test_decode_flags_valid_json_non_list_string_returns_empty_list():
    assert _decode_quality_flags('"just_a_string"') == []


def test_decode_flags_valid_json_dict_returns_empty_list():
    assert _decode_quality_flags('{"flag": "value"}') == []


def test_decode_flags_empty_list_string():
    assert _decode_quality_flags("[]") == []
