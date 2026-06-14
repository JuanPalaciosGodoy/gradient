"""
LLM-as-judge evaluator.

Uses a judge model (configurable via settings.llm_judge_model) to score
candidate responses against the original. Returns structured JSON rubric output.

Falls back to HeuristicEvaluator on parse failure or provider error.
"""
import json
import re
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.quality_score import QualityEvaluation, clamp_score
from app.providers.base import GenerationProvider

_JUDGE_SYSTEM_CONTEXT = """\
You are an expert evaluator assessing whether a candidate AI response is an
acceptable replacement for a reference response. Output ONLY a JSON object
with these exact keys (no extra text before or after):
{
  "candidate_quality": <float 0.0–1.0>,
  "quality_delta": <float -1.0–1.0, positive = candidate better>,
  "acceptable_replacement": <true or false>,
  "risk_flags": [<string>, ...],
  "explanation": "<one sentence>",
  "confidence": <float 0.0–1.0>
}"""

_JUDGE_USER_TEMPLATE = """\
Task type: {task_type}
Original prompt: {prompt}
Reference response: {original_response}
Candidate response: {candidate_response}"""


def _build_judge_prompt(
    prompt: str,
    original_response: str,
    candidate_response: str,
    task_type: Optional[str],
) -> str:
    return (
        _JUDGE_SYSTEM_CONTEXT
        + "\n\n"
        + _JUDGE_USER_TEMPLATE.format(
            task_type=task_type or "unspecified",
            prompt=prompt[:800],
            original_response=original_response[:800],
            candidate_response=candidate_response[:800],
        )
    )


def _extract_json(text: str) -> dict:
    """Find the first JSON object in text and parse it."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge response")
    return json.loads(match.group())


def _parse_judge_response(raw: str) -> QualityEvaluation:
    data = _extract_json(raw)

    score = clamp_score(float(data.get("candidate_quality", 0.5)))
    confidence = clamp_score(float(data.get("confidence", 0.7)))
    explanation = str(data.get("explanation", "LLM judge evaluation."))
    risk_flags: list[str] = [str(f) for f in data.get("risk_flags", [])]
    acceptable = bool(data.get("acceptable_replacement", score >= 0.7))

    flags = list(risk_flags)
    if not acceptable:
        flags.append("not_acceptable_replacement")

    return QualityEvaluation(
        score=score,
        method="llm_judge",
        explanation=explanation,
        confidence=confidence,
        flags=flags,
    )


class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluates using a judge LLM; falls back to heuristic on any error."""

    def __init__(self, provider: GenerationProvider, model: str):
        self._provider = provider
        self._model = model

    def evaluate(
        self,
        prompt: str,
        original_response: str,
        candidate_response: str,
        task_type: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> QualityEvaluation:
        judge_prompt = _build_judge_prompt(prompt, original_response, candidate_response, task_type)
        try:
            response = self._provider.generate(judge_prompt, self._model)
            return _parse_judge_response(response.text)
        except Exception as exc:
            from app.evaluation.heuristic_evaluator import HeuristicEvaluator
            fallback = HeuristicEvaluator()
            result = fallback.evaluate(prompt, original_response, candidate_response, task_type, feedback)
            # Use a distinct method name so the runner does not claim LLM_JUDGE evidence level.
            return QualityEvaluation(
                score=result.score,
                method="llm_judge_fallback",
                explanation=f"LLM judge failed ({exc}); heuristic fallback used. {result.explanation}",
                confidence=result.confidence * 0.7,
                flags=result.flags + ["llm_judge_fallback"],
            )
