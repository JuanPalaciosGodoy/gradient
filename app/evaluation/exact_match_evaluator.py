"""
Exact-match evaluator for classification tasks.

Compares candidate response against original using label matching:
  1.0  — exact string match
  0.95 — case/whitespace-normalized match
  0.70 — original label is a substring of candidate (model added explanation)
  fallback — HeuristicEvaluator (general-purpose scoring)
"""
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.quality_score import QualityEvaluation, clamp_score


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


class ExactMatchEvaluator(BaseEvaluator):
    """Label-exact evaluator, best suited for classification tasks."""

    def evaluate(
        self,
        prompt: str,
        original_response: str,
        candidate_response: str,
        task_type: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> QualityEvaluation:
        orig = original_response.strip()
        cand = candidate_response.strip()

        if orig == cand:
            return QualityEvaluation(
                score=1.0,
                method="exact_match",
                explanation="Exact string match.",
                confidence=1.0,
                flags=["exact_match"],
            )

        orig_norm = _normalize(orig)
        cand_norm = _normalize(cand)

        if orig_norm == cand_norm:
            return QualityEvaluation(
                score=0.95,
                method="exact_match",
                explanation="Normalized match (case/whitespace differences only).",
                confidence=0.95,
                flags=["normalized_match"],
            )

        if orig_norm and orig_norm in cand_norm:
            return QualityEvaluation(
                score=0.70,
                method="exact_match",
                explanation=(
                    f"Original label '{orig_norm}' found within candidate response. "
                    "Candidate may have added explanation text."
                ),
                confidence=0.75,
                flags=["label_contained"],
            )

        # No match — fall back to heuristic for a baseline score
        from app.evaluation.heuristic_evaluator import HeuristicEvaluator
        fallback = HeuristicEvaluator()
        result = fallback.evaluate(prompt, original_response, candidate_response, task_type, feedback)
        score = clamp_score(result.score * 0.5)  # penalise label mismatch
        return QualityEvaluation(
            score=score,
            method="exact_match",
            explanation=f"Label mismatch. Heuristic fallback score: {result.score:.2f}. {result.explanation}",
            confidence=0.60,
            flags=["label_mismatch"] + result.flags,
        )
