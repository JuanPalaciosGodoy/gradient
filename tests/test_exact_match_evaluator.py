"""Tests for ExactMatchEvaluator and TaskRouterEvaluator."""
import pytest

from app.evaluation.exact_match_evaluator import ExactMatchEvaluator
from app.evaluation.task_router_evaluator import TaskRouterEvaluator


# ── ExactMatchEvaluator ───────────────────────────────────────────────────────

@pytest.fixture
def em():
    return ExactMatchEvaluator()


def test_exact_string_match_returns_1(em):
    result = em.evaluate("p", "billing", "billing")
    assert result.score == 1.0
    assert result.method == "exact_match"
    assert "exact_match" in result.flags


def test_normalized_match_returns_095(em):
    result = em.evaluate("p", "Billing", "  billing  ")
    assert result.score == 0.95
    assert "normalized_match" in result.flags


def test_label_contained_returns_07(em):
    result = em.evaluate("p", "billing", "billing — account issue")
    assert result.score == 0.70
    assert "label_contained" in result.flags


def test_mismatch_returns_low_score(em):
    result = em.evaluate("p", "billing", "technical_support")
    assert result.score < 0.5
    assert "label_mismatch" in result.flags
    assert result.method == "exact_match"


def test_confidence_is_1_for_exact_match(em):
    result = em.evaluate("p", "refund", "refund")
    assert result.confidence == 1.0


def test_mismatch_confidence_is_limited(em):
    result = em.evaluate("p", "billing", "something completely different")
    assert result.confidence < 0.8


def test_empty_original_falls_through_to_fallback(em):
    result = em.evaluate("p", "", "anything")
    assert 0.0 <= result.score <= 1.0
    assert result.method == "exact_match"


# ── TaskRouterEvaluator ───────────────────────────────────────────────────────

@pytest.fixture
def router():
    return TaskRouterEvaluator()


def test_task_router_uses_exact_match_for_classification(router):
    result = router.evaluate(
        "classify the sentiment",
        "positive",
        "positive",
        task_type="classification",
    )
    assert result.method == "exact_match"
    assert result.score == 1.0


def test_task_router_uses_heuristic_for_summarization(router):
    result = router.evaluate(
        "Summarize the article.",
        "The article discusses AI trends.",
        "AI trends in this article include LLMs.",
        task_type="summarization",
    )
    assert result.method == "heuristic"
    assert 0.0 <= result.score <= 1.0


def test_task_router_uses_heuristic_for_none_task_type(router):
    result = router.evaluate("p", "reference", "candidate")
    assert result.method == "heuristic"


def test_task_router_classification_mismatch(router):
    result = router.evaluate("classify", "billing", "refund", task_type="classification")
    assert result.method == "exact_match"
    assert result.score < 0.7
    assert "label_mismatch" in result.flags


def test_task_router_coding_uses_heuristic(router):
    result = router.evaluate(
        "Write a function",
        "def foo(): return 1",
        "def foo():\n    return 1",
        task_type="coding",
    )
    assert result.method == "heuristic"
