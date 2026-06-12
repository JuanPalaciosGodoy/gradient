import pytest

from app.evaluation.heuristic_evaluator import HeuristicEvaluator


@pytest.fixture
def evaluator():
    return HeuristicEvaluator()


def test_identical_strings_score_near_one(evaluator):
    score = evaluator.evaluate("The quick brown fox", "The quick brown fox")
    assert score.score >= 0.99


def test_empty_strings_both_score_one(evaluator):
    score = evaluator.evaluate("", "")
    assert score.score == 1.0


def test_completely_different_strings_score_low(evaluator):
    score = evaluator.evaluate(
        "apple banana cherry",
        "xylophone zeppelin quorum",
    )
    assert score.score < 0.3


def test_similar_strings_score_higher_than_different(evaluator):
    original = "Summarize the quarterly earnings report for Q3."
    similar = "Summary of Q3 quarterly earnings report."
    different = "Write a Python function to sort a list."
    assert evaluator.evaluate(original, similar).score > evaluator.evaluate(original, different).score


def test_score_is_between_zero_and_one(evaluator):
    score = evaluator.evaluate("some original text here", "completely different words")
    assert 0.0 <= score.score <= 1.0


def test_method_is_heuristic(evaluator):
    score = evaluator.evaluate("hello world", "hello world")
    assert score.method == "heuristic"


def test_details_contain_token_overlap_and_length_ratio(evaluator):
    score = evaluator.evaluate("one two three", "one two four")
    assert "token_overlap" in score.details
    assert "length_ratio" in score.details


def test_one_empty_string_scores_zero(evaluator):
    score = evaluator.evaluate("some text", "")
    assert score.score == 0.0


def test_partial_overlap_score_is_intermediate(evaluator):
    original = "the cat sat on the mat"
    candidate = "the cat sat on a chair"  # shares most tokens
    score = evaluator.evaluate(original, candidate)
    assert 0.3 < score.score < 1.0


def test_length_ratio_penalizes_very_short_candidate(evaluator):
    score_full = evaluator.evaluate("a b c d e f g h i j", "a b c d e f g h i j")
    score_short = evaluator.evaluate("a b c d e f g h i j", "a")
    assert score_full.score > score_short.score
