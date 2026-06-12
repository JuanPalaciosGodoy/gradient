"""
Heuristic evaluator — no LLM calls required.

Combines two signals:
  - Token overlap (Jaccard similarity on word tokens)
  - Length ratio (penalizes extreme length differences)

Appropriate for local testing and as a fast baseline signal.
Not a substitute for semantic evaluation in production.
"""
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.quality_score import QualityScore


class HeuristicEvaluator(BaseEvaluator):

    def evaluate(
        self,
        original: str,
        candidate: str,
        task_type: Optional[str] = None,
    ) -> QualityScore:
        orig_tokens = set(original.lower().split())
        cand_tokens = set(candidate.lower().split())

        # Token overlap (Jaccard)
        union = orig_tokens | cand_tokens
        if not union:
            token_overlap = 1.0
        else:
            token_overlap = len(orig_tokens & cand_tokens) / len(union)

        # Length ratio: 1.0 when equal length, decreases as ratio diverges
        if len(original) == 0 and len(candidate) == 0:
            length_ratio = 1.0
        elif len(original) == 0 or len(candidate) == 0:
            length_ratio = 0.0
        else:
            ratio = len(candidate) / len(original)
            length_ratio = min(ratio, 1.0 / ratio)

        score = round(0.7 * token_overlap + 0.3 * length_ratio, 4)

        return QualityScore(
            score=score,
            method="heuristic",
            details={
                "token_overlap": round(token_overlap, 4),
                "length_ratio": round(length_ratio, 4),
            },
        )
