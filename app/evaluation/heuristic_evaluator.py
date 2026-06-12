"""
Heuristic evaluator — no LLM calls, no external APIs.

Uses task-specific signals to produce a transparent, conservative score.
Not a substitute for semantic evaluation; designed as a reliable baseline.
"""
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.quality_score import QualityEvaluation, clamp_score

# Words too common to carry meaning for keyword preservation checks
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "this", "that", "it", "its", "as", "not", "no", "so", "if", "i", "you",
    "we", "they", "he", "she", "what", "how", "when", "where", "which",
}

# Fragments that indicate code in the response
_CODE_MARKERS = (
    "def ", "class ", "import ", "return ", "function ", "=> ", "->",
    "```", "#!/", ".append(", "for (", "while (",
)


def _keywords(text: str) -> set[str]:
    """Meaningful words: stripped punctuation, length > 4, not stopwords."""
    words = text.lower().split()
    return {
        w.strip(".,;:!?()[]{}\"'") for w in words
        if len(w) > 4 and w.strip(".,;:!?()[]{}\"'") not in _STOPWORDS
    }


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    union = ta | tb
    if not union:
        return 1.0
    return len(ta & tb) / len(union)


def _length_ratio(original: str, candidate: str) -> float:
    if not original and not candidate:
        return 1.0
    if not original or not candidate:
        return 0.0
    r = len(candidate) / len(original)
    return min(r, 1.0 / r)


def _has_code(text: str) -> bool:
    return any(m in text for m in _CODE_MARKERS)


def _is_short_label(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) < 60 and "\n" not in stripped


def _looks_structured(text: str) -> bool:
    return any(c in text for c in ("{", "[", '": ', '":', ": "))


# ── Task-specific scorers ──────────────────────────────────────────────────────
# Each returns (raw_score: float, flags: list[str], explanation: str).
# Callers clamp and set confidence.

def _score_summarization(
    prompt: str, original: str, candidate: str
) -> tuple[float, list[str], str]:
    flags: list[str] = []

    if not candidate.strip():
        return 0.0, ["empty_response"], "Candidate is empty"

    if len(candidate.strip()) < 20:
        flags.append("short_response")

    # Keyword preservation: union of keywords from prompt and original
    source_keywords = _keywords(original) | _keywords(prompt)
    if source_keywords:
        preserved = sum(1 for kw in source_keywords if kw in candidate.lower())
        keyword_score = preserved / len(source_keywords)
    else:
        keyword_score = _jaccard(original, candidate)

    # Length check: good summaries compress; longer-than-original is suspicious
    orig_len = max(len(original), 1)
    ratio = len(candidate) / orig_len
    if ratio > 1.5:
        length_score = 0.4
        flags.append("candidate_longer_than_original")
    elif ratio > 1.0:
        length_score = 0.7
    elif ratio >= 0.05:
        length_score = 1.0
    else:
        length_score = 0.3
        flags.append("short_response")

    score = clamp_score(0.65 * keyword_score + 0.35 * length_score)
    explanation = (
        f"Summarization: {round(keyword_score * 100)}% of source keywords preserved; "
        f"candidate is {round(ratio * 100)}% of original length"
    )
    return score, flags, explanation


def _score_classification(
    prompt: str, original: str, candidate: str
) -> tuple[float, list[str], str]:
    flags: list[str] = []

    if not candidate.strip():
        return 0.0, ["empty_response"], "Candidate is empty"

    orig_stripped = original.strip().lower()
    cand_stripped = candidate.strip().lower()

    if _is_short_label(original):
        # Original is a short label — candidate should also be a label
        if orig_stripped == cand_stripped:
            flags.append("exact_label_match")
            return 0.95, flags, f"Exact label match: '{original.strip()}'"

        if orig_stripped in cand_stripped:
            if len(candidate.strip()) > len(original.strip()) * 3:
                flags.append("verbose_for_classification")
                return 0.55, flags, "Label present but response is verbose for a classification task"
            return 0.75, flags, "Label found in candidate response"

        if len(candidate.strip()) > len(original.strip()) * 3:
            flags.append("verbose_for_classification")

        score = clamp_score(_jaccard(original, candidate))
        return score, flags, f"Classification: Jaccard similarity {round(score, 2)}"

    # Original is a longer response — fall back to generic similarity
    score = clamp_score(0.7 * _jaccard(original, candidate) + 0.3 * _length_ratio(original, candidate))
    return score, flags, f"Classification (verbose original): similarity {round(score, 2)}"


def _score_extraction(
    prompt: str, original: str, candidate: str
) -> tuple[float, list[str], str]:
    flags: list[str] = []

    if not candidate.strip():
        return 0.0, ["empty_response"], "Candidate is empty"

    original_structured = _looks_structured(original)
    candidate_structured = _looks_structured(candidate)

    structure_bonus = 0.0
    if original_structured and candidate_structured:
        structure_bonus = 0.15
        flags.append("structured_output_preserved")
    elif original_structured and not candidate_structured:
        flags.append("missing_structured_output")
        structure_bonus = -0.2

    # Entity/field overlap: terms in prompt that appear in both responses
    prompt_keywords = _keywords(prompt)
    if prompt_keywords:
        in_original = sum(1 for kw in prompt_keywords if kw in original.lower())
        in_candidate = sum(1 for kw in prompt_keywords if kw in candidate.lower())
        entity_score = in_candidate / max(in_original, 1)
    else:
        entity_score = _jaccard(original, candidate)

    base = clamp_score(0.7 * entity_score + 0.3 * _length_ratio(original, candidate))
    score = clamp_score(base + structure_bonus)
    explanation = (
        f"Extraction: entity coverage {round(entity_score * 100)}%; "
        f"{'structured output preserved' if 'structured_output_preserved' in flags else 'structure not matched'}"
    )
    return score, flags, explanation


def _score_research(
    prompt: str, original: str, candidate: str
) -> tuple[float, list[str], str]:
    flags: list[str] = ["research_conservative"]

    if not candidate.strip():
        return 0.0, ["empty_response"], "Candidate is empty"

    if len(candidate.strip()) < 80:
        flags.append("research_too_short")
        return 0.2, flags, "Research response is too short; research tasks require substantive answers"

    keyword_overlap = _jaccard(original, candidate)
    prompt_keywords = _keywords(prompt)
    if prompt_keywords:
        coverage = sum(1 for kw in prompt_keywords if kw in candidate.lower()) / len(prompt_keywords)
    else:
        coverage = keyword_overlap

    raw = clamp_score(0.6 * keyword_overlap + 0.4 * coverage)
    # Cap at 0.80 — research is high-risk; be conservative about claiming quality
    score = min(raw, 0.80)
    explanation = (
        f"Research (conservative): {round(keyword_overlap * 100)}% token overlap, "
        f"{round(coverage * 100)}% prompt-term coverage; score capped at 0.80"
    )
    return score, flags, explanation


def _score_coding(
    prompt: str, original: str, candidate: str
) -> tuple[float, list[str], str]:
    flags: list[str] = ["coding_conservative"]

    if not candidate.strip():
        return 0.0, ["empty_response"], "Candidate is empty"

    original_has_code = _has_code(original)
    candidate_has_code = _has_code(candidate)

    if original_has_code and not candidate_has_code:
        flags.append("missing_code_structure")
        return 0.25, flags, "Original contains code but candidate does not"

    base = clamp_score(0.7 * _jaccard(original, candidate) + 0.3 * _length_ratio(original, candidate))

    if original_has_code and candidate_has_code:
        base = clamp_score(base + 0.1)

    # Cap at 0.85 — code correctness requires execution, not text comparison
    score = min(base, 0.85)
    explanation = (
        f"Coding (conservative): similarity {round(base * 100)}%; "
        f"{'code structure present in both' if original_has_code and candidate_has_code else 'no code structure detected'}; "
        f"score capped at 0.85"
    )
    return score, flags, explanation


def _score_customer_support(
    prompt: str, original: str, candidate: str
) -> tuple[float, list[str], str]:
    flags: list[str] = []

    if not candidate.strip():
        return 0.0, ["empty_response"], "Candidate is empty"

    if len(candidate.strip()) < 15:
        flags.append("short_response")
        return 0.2, flags, "Customer support response is too short"

    orig_len = max(len(original), 1)
    if len(candidate) > orig_len * 5:
        flags.append("excessively_long_response")

    score = clamp_score(0.7 * _jaccard(original, candidate) + 0.3 * _length_ratio(original, candidate))
    explanation = f"Customer support: similarity {round(score * 100)}%"
    return score, flags, explanation


def _score_generic(
    prompt: str, original: str, candidate: str
) -> tuple[float, list[str], str]:
    flags: list[str] = []

    if not original and not candidate:
        return 1.0, flags, "Both responses are empty"

    if not candidate.strip():
        return 0.0, ["empty_response"], "Candidate is empty"

    score = clamp_score(0.7 * _jaccard(original, candidate) + 0.3 * _length_ratio(original, candidate))
    explanation = f"Generic heuristic: similarity {round(score * 100)}%"
    return score, flags, explanation


_TASK_SCORERS = {
    "summarization": _score_summarization,
    "classification": _score_classification,
    "extraction": _score_extraction,
    "research": _score_research,
    "coding": _score_coding,
    "customer_support": _score_customer_support,
}

# Confidence levels per task type — reflects how reliable heuristics are
_TASK_CONFIDENCE = {
    "summarization": 0.75,
    "classification": 0.85,
    "extraction": 0.70,
    "research": 0.60,
    "coding": 0.60,
    "customer_support": 0.75,
}


class HeuristicEvaluator(BaseEvaluator):

    def evaluate(
        self,
        prompt: str,
        original_response: str,
        candidate_response: str,
        task_type: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> QualityEvaluation:
        scorer = _TASK_SCORERS.get(task_type or "", _score_generic)
        score, flags, explanation = scorer(prompt, original_response, candidate_response)
        confidence = _TASK_CONFIDENCE.get(task_type or "", 0.65)

        return QualityEvaluation(
            score=round(clamp_score(score), 4),
            method="heuristic",
            explanation=explanation,
            confidence=confidence,
            flags=flags,
        )
