"""Tests for LLMJudgeEvaluator — JSON parsing, fallback, evidence level."""
import json

import pytest

from app.evaluation.llm_judge_evaluator import (
    LLMJudgeEvaluator,
    _build_judge_prompt,
    _parse_judge_response,
)
from app.providers.base import GenerationProvider, ProviderResponse
from app.providers.fake import FakeProvider


def _make_response(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        latency_ms=100.0,
        input_tokens=50,
        output_tokens=30,
        estimated_cost=0.001,
    )


class FixedProvider(GenerationProvider):
    def __init__(self, text: str):
        self._text = text

    def generate(self, prompt, model):
        return _make_response(self._text)


class RaisingProvider(GenerationProvider):
    def generate(self, prompt, model):
        raise RuntimeError("API unavailable")


# ── _parse_judge_response ─────────────────────────────────────────────────────

def test_parse_valid_json():
    payload = {
        "candidate_quality": 0.85,
        "quality_delta": 0.05,
        "acceptable_replacement": True,
        "risk_flags": [],
        "explanation": "Very similar output.",
        "confidence": 0.9,
    }
    result = _parse_judge_response(json.dumps(payload))
    assert result.score == 0.85
    assert result.confidence == 0.9
    assert result.explanation == "Very similar output."
    assert result.method == "llm_judge"
    assert result.flags == []


def test_parse_json_embedded_in_text():
    text = 'Here is my evaluation:\n{"candidate_quality": 0.7, "quality_delta": -0.1, "acceptable_replacement": false, "risk_flags": ["quality_loss"], "explanation": "Some loss.", "confidence": 0.8}'
    result = _parse_judge_response(text)
    assert result.score == 0.7
    assert "quality_loss" in result.flags
    assert "not_acceptable_replacement" in result.flags


def test_parse_clamps_out_of_range_score():
    payload = {"candidate_quality": 1.5, "confidence": 0.8, "explanation": "x"}
    result = _parse_judge_response(json.dumps(payload))
    assert result.score == 1.0


def test_parse_missing_json_raises():
    with pytest.raises((ValueError, Exception)):
        _parse_judge_response("I cannot score this response.")


def test_parse_invalid_json_raises():
    with pytest.raises((ValueError, json.JSONDecodeError, Exception)):
        _parse_judge_response("{not valid json}")


# ── LLMJudgeEvaluator — successful call ──────────────────────────────────────

def test_llm_judge_returns_llm_judge_method():
    payload = {
        "candidate_quality": 0.9,
        "quality_delta": 0.0,
        "acceptable_replacement": True,
        "risk_flags": [],
        "explanation": "High quality match.",
        "confidence": 0.95,
    }
    evaluator = LLMJudgeEvaluator(
        provider=FixedProvider(json.dumps(payload)),
        model="gpt-4o-mini",
    )
    result = evaluator.evaluate(
        prompt="Summarize this article.",
        original_response="The article discusses climate change.",
        candidate_response="Climate change article reviewed.",
    )
    assert result.method == "llm_judge"
    assert result.score == 0.9
    assert "llm_judge_fallback" not in result.flags


# ── LLMJudgeEvaluator — fallback on provider error ───────────────────────────

def test_llm_judge_falls_back_on_provider_error():
    evaluator = LLMJudgeEvaluator(
        provider=RaisingProvider(),
        model="gpt-4o-mini",
    )
    result = evaluator.evaluate(
        prompt="Classify: billing or technical?",
        original_response="billing",
        candidate_response="billing",
        task_type="classification",
    )
    # Fallback uses a distinct method so evidence level is not overclaimed
    assert result.method == "llm_judge_fallback"
    assert "llm_judge_fallback" in result.flags
    assert 0.0 <= result.score <= 1.0


def test_llm_judge_falls_back_on_invalid_json():
    evaluator = LLMJudgeEvaluator(
        provider=FixedProvider("I cannot evaluate this."),
        model="gpt-4o-mini",
    )
    result = evaluator.evaluate("p", "original", "candidate")
    assert result.method == "llm_judge_fallback"
    assert "llm_judge_fallback" in result.flags


def test_llm_judge_fallback_confidence_is_discounted():
    evaluator = LLMJudgeEvaluator(
        provider=RaisingProvider(),
        model="gpt-4o-mini",
    )
    result = evaluator.evaluate("p", "original", "original")
    assert result.confidence <= 0.9  # discounted from heuristic baseline


# ── Evidence level in runner ──────────────────────────────────────────────────

def test_llm_judge_replay_result_has_llm_judge_evidence():
    from datetime import datetime, timezone
    from app.replay.replay_models import ReplayCandidate, ReplayRequest
    from app.replay.replay_runner import ReplayRunner
    from app.schemas import EvidenceLevel

    payload = {
        "candidate_quality": 0.85,
        "quality_delta": 0.0,
        "acceptable_replacement": True,
        "risk_flags": [],
        "explanation": "Good.",
        "confidence": 0.9,
    }
    evaluator = LLMJudgeEvaluator(
        provider=FixedProvider(json.dumps(payload)),
        model="gpt-4o-mini",
    )
    runner = ReplayRunner(provider=FakeProvider(), evaluator=evaluator)
    req = ReplayRequest(
        original_record_id="1",
        prompt="Summarize.",
        original_response="summary",
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
    assert results[0].evidence_level == EvidenceLevel.LLM_JUDGE


# ── Factory wires up LLM judge ────────────────────────────────────────────────

def test_factory_returns_llm_judge_evaluator(monkeypatch):
    from app.config import settings
    from app.evaluation.factory import get_evaluator

    monkeypatch.setattr(settings, "real_provider_mode", False)
    monkeypatch.setattr(settings, "llm_judge_model", "gpt-4o-mini")
    evaluator = get_evaluator("llm_judge")
    assert isinstance(evaluator, LLMJudgeEvaluator)


def test_factory_raises_for_unknown_mode():
    from app.evaluation.factory import get_evaluator
    with pytest.raises(ValueError, match="Unknown evaluator mode"):
        get_evaluator("nonsense_mode")
