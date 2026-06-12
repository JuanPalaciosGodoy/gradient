"""Tests for the evaluator factory and config-driven evaluator selection."""
import pytest

from app.config import settings
from app.evaluation.factory import get_evaluator
from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.evaluation.llm_judge_evaluator import LLMJudgeEvaluator
from app.evaluation.prometheus_evaluator import PrometheusEvaluator


# ── Factory return types ──────────────────────────────────────────────────────

def test_heuristic_mode_returns_heuristic_evaluator():
    ev = get_evaluator("heuristic")
    assert isinstance(ev, HeuristicEvaluator)


def test_prometheus_mode_returns_prometheus_evaluator():
    ev = get_evaluator("prometheus")
    assert isinstance(ev, PrometheusEvaluator)


def test_llm_judge_mode_returns_llm_judge_evaluator():
    ev = get_evaluator("llm_judge")
    assert isinstance(ev, LLMJudgeEvaluator)


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError, match="Unknown evaluator mode"):
        get_evaluator("magic_eval_9000")


def test_default_mode_is_heuristic():
    ev = get_evaluator()
    assert isinstance(ev, HeuristicEvaluator)


# ── Stub evaluators raise NotImplementedError ─────────────────────────────────

def test_prometheus_raises_not_implemented_on_evaluate():
    ev = get_evaluator("prometheus")
    with pytest.raises(NotImplementedError):
        ev.evaluate("prompt", "original", "candidate", task_type="summarization")


def test_llm_judge_raises_not_implemented_on_evaluate():
    ev = get_evaluator("llm_judge")
    with pytest.raises(NotImplementedError):
        ev.evaluate("prompt", "original", "candidate", task_type="summarization")


# ── Config-driven selection ───────────────────────────────────────────────────

def test_settings_evaluator_mode_default_is_heuristic():
    assert settings.evaluator_mode == "heuristic"


def test_config_driven_evaluator_selection(monkeypatch):
    monkeypatch.setattr(settings, "evaluator_mode", "heuristic")
    ev = get_evaluator(settings.evaluator_mode)
    assert isinstance(ev, HeuristicEvaluator)
