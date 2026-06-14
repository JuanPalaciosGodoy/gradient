"""
Tests for real provider mode, provider router, and new ProviderResponse fields.

All tests run in fake mode (no real API calls). Real provider paths are tested
with monkeypatching or by verifying provider selection logic.
"""
import pytest

from app.providers.base import ProviderResponse
from app.providers.fake import FakeProvider
from app.providers.router import get_generation_provider, _is_openai_model, _is_anthropic_model, _is_gemini_model


# ── ProviderResponse new fields ───────────────────────────────────────────────

def test_provider_response_has_new_fields():
    r = ProviderResponse(
        text="hello",
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=5,
        estimated_cost=0.001,
    )
    assert r.provider == ""
    assert r.model == ""
    assert r.cost_source == "estimated_catalog"
    assert r.latency_source == "fake"


def test_provider_response_fields_settable():
    r = ProviderResponse(
        text="ok",
        latency_ms=50.0,
        input_tokens=20,
        output_tokens=10,
        estimated_cost=0.002,
        provider="openai",
        model="gpt-4o-mini",
        cost_source="observed",
        latency_source="observed",
    )
    assert r.provider == "openai"
    assert r.model == "gpt-4o-mini"
    assert r.cost_source == "observed"
    assert r.latency_source == "observed"


# ── FakeProvider new fields ───────────────────────────────────────────────────

def test_fake_provider_sets_provider_and_model():
    fp = FakeProvider()
    r = fp.generate("hello world", "gpt-4o-mini")
    assert r.provider == "fake"
    assert r.model == "gpt-4o-mini"
    assert r.cost_source == "estimated_catalog"
    assert r.latency_source == "fake"


def test_fake_provider_returns_tokens():
    fp = FakeProvider()
    r = fp.generate("A short prompt.", "gpt-4o-mini")
    assert r.input_tokens > 0
    assert r.output_tokens > 0


# ── Router: fake mode (default) ───────────────────────────────────────────────

def test_router_returns_fake_when_real_mode_off(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "real_provider_mode", False)
    provider = get_generation_provider("gpt-4o-mini")
    assert isinstance(provider, FakeProvider)


def test_router_returns_fake_for_unknown_model_even_in_real_mode(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "real_provider_mode", True)
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    provider = get_generation_provider("some-unknown-model-xyz")
    assert isinstance(provider, FakeProvider)


def test_router_returns_fake_when_real_mode_true_but_no_keys(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "real_provider_mode", True)
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    for model in ("gpt-4o", "claude-3-haiku-20240307", "gemini-1.5-flash"):
        assert isinstance(get_generation_provider(model), FakeProvider)


# ── Model family detection ────────────────────────────────────────────────────

def test_openai_model_detection():
    assert _is_openai_model("gpt-4o")
    assert _is_openai_model("gpt-4o-mini")
    assert _is_openai_model("gpt-3.5-turbo")
    assert _is_openai_model("o1-preview")
    assert not _is_openai_model("claude-3-sonnet")
    assert not _is_openai_model("gemini-1.5-pro")


def test_anthropic_model_detection():
    assert _is_anthropic_model("claude-3-haiku-20240307")
    assert _is_anthropic_model("claude-3-5-sonnet-20241022")
    assert not _is_anthropic_model("gpt-4o")
    assert not _is_anthropic_model("gemini-1.5-flash")


def test_gemini_model_detection():
    assert _is_gemini_model("gemini-1.5-pro")
    assert _is_gemini_model("gemini-1.5-flash")
    assert not _is_gemini_model("gpt-4o")
    assert not _is_gemini_model("claude-3-haiku-20240307")


# ── Provider factory: OpenAI ImportError graceful handling ────────────────────

def test_openai_provider_raises_clear_error_when_sdk_missing(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "openai", None)
    from app.providers.openai_generation import OpenAIGenerationProvider
    with pytest.raises(ImportError, match="openai package"):
        OpenAIGenerationProvider(api_key="sk-fake")


def test_anthropic_provider_raises_clear_error_when_sdk_missing(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "anthropic", None)
    from app.providers.anthropic_generation import AnthropicGenerationProvider
    with pytest.raises(ImportError, match="anthropic package"):
        AnthropicGenerationProvider(api_key="sk-fake")


# ── ReplayRunner uses provider_factory ───────────────────────────────────────

def test_replay_runner_uses_provider_factory(monkeypatch):
    from datetime import datetime, timezone
    from app.evaluation.heuristic_evaluator import HeuristicEvaluator
    from app.replay.replay_models import ReplayCandidate, ReplayRequest
    from app.replay.replay_runner import ReplayRunner

    call_log = []

    def mock_factory(model: str):
        call_log.append(model)
        return FakeProvider()

    runner = ReplayRunner(provider_factory=mock_factory, evaluator=HeuristicEvaluator())
    req = ReplayRequest(
        original_record_id="1",
        prompt="Hello?",
        original_response="Hi there.",
        original_model="gpt-4o",
        original_cost=0.01,
        task_type="summarization",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    candidate = ReplayCandidate(
        provider="openai",
        model="gpt-4o-mini",
        model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    runner.run([req], [candidate])
    assert "gpt-4o-mini" in call_log


def test_replay_runner_backward_compat_with_provider_arg():
    from datetime import datetime, timezone
    from app.evaluation.heuristic_evaluator import HeuristicEvaluator
    from app.replay.replay_models import ReplayCandidate, ReplayRequest
    from app.replay.replay_runner import ReplayRunner

    runner = ReplayRunner(provider=FakeProvider(), evaluator=HeuristicEvaluator())
    req = ReplayRequest(
        original_record_id="1",
        prompt="Hello?",
        original_response="Hi.",
        original_model="gpt-4o",
        original_cost=0.01,
        task_type="summarization",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    candidate = ReplayCandidate(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00025,
        estimated_output_cost_per_1k_tokens=0.00125,
    )
    results = runner.run([req], [candidate])
    assert len(results) == 1
    assert results[0].error_message is None


# ── New cost/latency fields in ReplayResult ───────────────────────────────────

def test_replay_result_has_cost_latency_provenance_fields():
    from datetime import datetime, timezone
    from app.evaluation.heuristic_evaluator import HeuristicEvaluator
    from app.replay.replay_models import ReplayCandidate, ReplayRequest
    from app.replay.replay_runner import ReplayRunner

    runner = ReplayRunner(provider=FakeProvider(), evaluator=HeuristicEvaluator())
    req = ReplayRequest(
        original_record_id="42",
        prompt="Summarize this.",
        original_response="Short summary.",
        original_model="gpt-4o",
        original_cost=0.01,
        task_type="summarization",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    candidate = ReplayCandidate(
        provider="openai",
        model="gpt-4o-mini",
        model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    results = runner.run([req], [candidate])
    r = results[0]
    assert r.input_tokens > 0
    assert r.output_tokens > 0
    assert r.cost_source in ("observed", "estimated_catalog", "fake", "missing")
    assert r.latency_source in ("observed", "fake", "missing")


def test_error_result_has_missing_cost_latency_source():
    from datetime import datetime, timezone
    from app.evaluation.heuristic_evaluator import HeuristicEvaluator
    from app.providers.base import GenerationProvider
    from app.replay.replay_models import ReplayCandidate, ReplayRequest
    from app.replay.replay_runner import ReplayRunner

    class ErrorProvider(GenerationProvider):
        def generate(self, prompt, model):
            raise RuntimeError("Provider is down")

    runner = ReplayRunner(provider=ErrorProvider(), evaluator=HeuristicEvaluator())
    req = ReplayRequest(
        original_record_id="1",
        prompt="Hello",
        original_response="Hi",
        original_model="gpt-4o",
        original_cost=0.01,
        task_type="summarization",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    candidate = ReplayCandidate(
        provider="openai",
        model="gpt-4o-mini",
        model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    results = runner.run([req], [candidate])
    r = results[0]
    assert r.cost_source == "missing"
    assert r.latency_source == "missing"
    assert r.input_tokens == 0
    assert r.output_tokens == 0
