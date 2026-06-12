"""
Tests for HeuristicEvaluator: task-specific scoring paths and invariants.
"""
import pytest

from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.evaluation.quality_score import QualityEvaluation


@pytest.fixture
def ev():
    return HeuristicEvaluator()


# ── Result type ───────────────────────────────────────────────────────────────

def test_returns_quality_evaluation(ev):
    result = ev.evaluate("prompt", "original", "candidate", task_type="summarization")
    assert isinstance(result, QualityEvaluation)


def test_method_is_heuristic(ev):
    result = ev.evaluate("p", "o", "c")
    assert result.method == "heuristic"


def test_score_always_in_range(ev):
    for task in ["summarization", "classification", "extraction", "research", "coding", "customer_support", None]:
        result = ev.evaluate("some prompt", "original text", "candidate text", task_type=task)
        assert 0.0 <= result.score <= 1.0, f"Out of range for task={task}"


def test_explanation_is_nonempty(ev):
    result = ev.evaluate("p", "original", "candidate", task_type="summarization")
    assert len(result.explanation) > 0


def test_confidence_in_range(ev):
    result = ev.evaluate("p", "o", "c", task_type="coding")
    assert 0.0 <= result.confidence <= 1.0


# ── Empty candidate ───────────────────────────────────────────────────────────

def test_empty_candidate_scores_zero_summarization(ev):
    result = ev.evaluate("Summarize this", "A long original response.", "", task_type="summarization")
    assert result.score == 0.0
    assert "empty_response" in result.flags


def test_empty_candidate_scores_zero_classification(ev):
    result = ev.evaluate("Classify this", "positive", "", task_type="classification")
    assert result.score == 0.0


def test_empty_candidate_scores_zero_research(ev):
    result = ev.evaluate("Explain quantum computing", "Detailed explanation...", "", task_type="research")
    assert result.score == 0.0


def test_empty_candidate_scores_zero_coding(ev):
    result = ev.evaluate("Write a sort function", "def sort(arr): ...", "", task_type="coding")
    assert result.score == 0.0


def test_both_empty_generic_scores_one(ev):
    result = ev.evaluate("p", "", "")
    assert result.score == 1.0


# ── Summarization ─────────────────────────────────────────────────────────────

def test_summarization_preserves_keywords(ev):
    original = "The quarterly earnings report shows revenue growth of fifteen percent driven by software sales."
    candidate = "Revenue grew fifteen percent driven by software sales."
    result = ev.evaluate("Summarize the report", original, candidate, task_type="summarization")
    assert result.score > 0.5


def test_summarization_candidate_longer_than_original_penalized(ev):
    original = "Short summary here."
    candidate = "This is a much much much much much much much much longer response than the original."
    result = ev.evaluate("Summarize", original, candidate, task_type="summarization")
    assert "candidate_longer_than_original" in result.flags
    assert result.score < 0.7


def test_summarization_empty_candidate_flags(ev):
    result = ev.evaluate("Summarize", "some original text", "", task_type="summarization")
    assert result.score == 0.0
    assert "empty_response" in result.flags


# ── Classification ────────────────────────────────────────────────────────────

def test_classification_exact_label_match_scores_high(ev):
    result = ev.evaluate(
        "Classify the sentiment",
        "positive",
        "positive",
        task_type="classification",
    )
    assert result.score >= 0.90
    assert "exact_label_match" in result.flags


def test_classification_exact_match_case_insensitive(ev):
    result = ev.evaluate("Classify", "Positive", "positive", task_type="classification")
    assert "exact_label_match" in result.flags


def test_classification_verbose_response_penalized(ev):
    result = ev.evaluate(
        "Classify the sentiment",
        "positive",
        "The sentiment of this text is definitely positive because the author uses many upbeat words and phrases.",
        task_type="classification",
    )
    assert "verbose_for_classification" in result.flags
    assert result.score < 0.80


def test_classification_label_in_verbose_response(ev):
    result = ev.evaluate(
        "Classify",
        "negative",
        "The sentiment here is negative.",
        task_type="classification",
    )
    # Label present but verbose — middle score
    assert 0.4 < result.score < 0.9


# ── Extraction ────────────────────────────────────────────────────────────────

def test_extraction_structured_output_preserved_bonus(ev):
    original = '{"name": "John", "age": 30, "city": "New York"}'
    candidate = '{"name": "John", "age": 30, "city": "New York"}'
    result = ev.evaluate(
        "Extract name, age, and city",
        original,
        candidate,
        task_type="extraction",
    )
    assert "structured_output_preserved" in result.flags
    assert result.score > 0.6


def test_extraction_missing_structure_penalized(ev):
    original = '{"name": "Alice", "role": "engineer"}'
    candidate = "The person is Alice who works as an engineer."
    result = ev.evaluate(
        "Extract name and role",
        original,
        candidate,
        task_type="extraction",
    )
    assert "missing_structured_output" in result.flags


# ── Research ──────────────────────────────────────────────────────────────────

def test_research_score_is_conservative(ev):
    original = "Quantum computing uses qubits and superposition to process information faster."
    candidate = "Quantum computing uses qubits and superposition to process information much faster than classical computers."
    result = ev.evaluate(
        "Explain quantum computing",
        original,
        candidate,
        task_type="research",
    )
    assert result.score <= 0.80
    assert "research_conservative" in result.flags


def test_research_too_short_penalized(ev):
    result = ev.evaluate(
        "Explain quantum computing",
        "A long detailed explanation of quantum computing principles.",
        "It uses qubits.",
        task_type="research",
    )
    assert result.score <= 0.25
    assert "research_too_short" in result.flags


def test_research_confidence_is_lower(ev):
    result = ev.evaluate("Explain X", "Long explanation...", "Another long explanation...", task_type="research")
    assert result.confidence <= 0.65


# ── Coding ────────────────────────────────────────────────────────────────────

def test_coding_missing_code_structure_penalized(ev):
    original = "def sort_list(arr):\n    return sorted(arr)"
    candidate = "You can sort a list using the sorted function."
    result = ev.evaluate(
        "Write a sort function",
        original,
        candidate,
        task_type="coding",
    )
    assert "missing_code_structure" in result.flags
    assert result.score <= 0.35


def test_coding_score_capped_below_one(ev):
    original = "def add(a, b):\n    return a + b"
    candidate = "def add(a, b):\n    return a + b"
    result = ev.evaluate("Write an add function", original, candidate, task_type="coding")
    assert result.score <= 0.85
    assert "coding_conservative" in result.flags


def test_coding_confidence_is_lower(ev):
    result = ev.evaluate("Write code", "def f(): pass", "def f(): pass", task_type="coding")
    assert result.confidence <= 0.65


# ── Customer support ──────────────────────────────────────────────────────────

def test_customer_support_short_response_penalized(ev):
    result = ev.evaluate(
        "Help me reset my password",
        "Sure! Please visit our password reset page at settings.example.com and follow the steps.",
        "ok",
        task_type="customer_support",
    )
    assert "short_response" in result.flags
    assert result.score < 0.5


def test_customer_support_reasonable_response_scores_ok(ev):
    original = "Thank you for reaching out! To reset your password, please visit settings."
    candidate = "To reset your password, please go to the settings page and click reset."
    result = ev.evaluate("Reset my password", original, candidate, task_type="customer_support")
    assert result.score > 0.3


# ── Generic / other ───────────────────────────────────────────────────────────

def test_generic_identical_scores_high(ev):
    result = ev.evaluate("prompt", "identical text here", "identical text here")
    assert result.score > 0.95


def test_generic_completely_different_scores_low(ev):
    result = ev.evaluate("prompt", "apple banana cherry", "xylophone zeppelin quorum")
    assert result.score < 0.3


def test_unknown_task_type_uses_generic(ev):
    result = ev.evaluate("p", "original text", "original text", task_type="unknown_future_type")
    assert result.score > 0.95
