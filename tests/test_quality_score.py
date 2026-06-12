"""Tests for quality_score utilities."""
import pytest

from app.evaluation.quality_score import (
    QualityEvaluation,
    average_quality,
    clamp_score,
    quality_delta,
    quality_loss_percentage,
    summarize_quality_distribution,
)


def _eval(score: float) -> QualityEvaluation:
    return QualityEvaluation(
        score=score, method="heuristic",
        explanation="test", confidence=0.8, flags=[],
    )


# ── clamp_score ───────────────────────────────────────────────────────────────

def test_clamp_score_within_range():
    assert clamp_score(0.5) == 0.5


def test_clamp_score_below_zero():
    assert clamp_score(-0.3) == 0.0


def test_clamp_score_above_one():
    assert clamp_score(1.5) == 1.0


def test_clamp_score_boundaries():
    assert clamp_score(0.0) == 0.0
    assert clamp_score(1.0) == 1.0


# ── average_quality ───────────────────────────────────────────────────────────

def test_average_quality_empty_list():
    assert average_quality([]) == 0.0


def test_average_quality_single():
    assert average_quality([_eval(0.8)]) == 0.8


def test_average_quality_multiple():
    result = average_quality([_eval(0.6), _eval(0.8), _eval(1.0)])
    assert abs(result - 0.8) < 0.001


# ── quality_delta ─────────────────────────────────────────────────────────────

def test_quality_delta_positive_improvement():
    assert quality_delta(0.7, 0.9) == pytest.approx(0.2, abs=1e-4)


def test_quality_delta_degradation_is_negative():
    assert quality_delta(0.9, 0.7) == pytest.approx(-0.2, abs=1e-4)


def test_quality_delta_equal_is_zero():
    assert quality_delta(0.8, 0.8) == 0.0


# ── quality_loss_percentage ───────────────────────────────────────────────────

def test_quality_loss_no_loss():
    assert quality_loss_percentage(0.8, 0.9) == 0.0


def test_quality_loss_fifty_percent():
    assert quality_loss_percentage(1.0, 0.5) == pytest.approx(50.0, abs=0.01)


def test_quality_loss_zero_baseline():
    assert quality_loss_percentage(0.0, 0.5) == 0.0


def test_quality_loss_complete_degradation():
    assert quality_loss_percentage(1.0, 0.0) == pytest.approx(100.0, abs=0.01)


# ── summarize_quality_distribution ───────────────────────────────────────────

def test_summarize_empty_list():
    summary = summarize_quality_distribution([])
    assert summary["count"] == 0
    assert summary["avg"] == 0.0


def test_summarize_counts_below_threshold():
    evals = [_eval(0.9), _eval(0.5), _eval(0.6), _eval(0.8)]
    summary = summarize_quality_distribution(evals, threshold=0.7)
    assert summary["below_threshold"] == 2


def test_summarize_min_max():
    evals = [_eval(0.3), _eval(0.7), _eval(1.0)]
    summary = summarize_quality_distribution(evals)
    assert summary["min"] == pytest.approx(0.3, abs=1e-4)
    assert summary["max"] == pytest.approx(1.0, abs=1e-4)


def test_summarize_count():
    evals = [_eval(0.5)] * 5
    assert summarize_quality_distribution(evals)["count"] == 5


# ── QualityEvaluation validation ──────────────────────────────────────────────

def test_quality_evaluation_score_below_zero_raises():
    with pytest.raises(Exception):
        QualityEvaluation(score=-0.1, method="heuristic", explanation="x", confidence=0.8, flags=[])


def test_quality_evaluation_score_above_one_raises():
    with pytest.raises(Exception):
        QualityEvaluation(score=1.1, method="heuristic", explanation="x", confidence=0.8, flags=[])


def test_quality_evaluation_confidence_below_zero_raises():
    with pytest.raises(Exception):
        QualityEvaluation(score=0.5, method="heuristic", explanation="x", confidence=-0.1, flags=[])


def test_quality_evaluation_confidence_above_one_raises():
    with pytest.raises(Exception):
        QualityEvaluation(score=0.5, method="heuristic", explanation="x", confidence=1.1, flags=[])


def test_quality_evaluation_boundary_values_are_valid():
    e1 = QualityEvaluation(score=0.0, method="heuristic", explanation="x", confidence=0.0, flags=[])
    e2 = QualityEvaluation(score=1.0, method="heuristic", explanation="x", confidence=1.0, flags=[])
    assert e1.score == 0.0
    assert e2.confidence == 1.0


def test_quality_evaluation_flags_are_independent_instances():
    e1 = QualityEvaluation(score=0.5, method="heuristic", explanation="x", confidence=0.8, flags=[])
    e2 = QualityEvaluation(score=0.5, method="heuristic", explanation="x", confidence=0.8, flags=[])
    e1.flags.append("some_flag")
    assert "some_flag" not in e2.flags
