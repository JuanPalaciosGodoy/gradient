"""
Replay runner: runs historical prompts against candidate models.

Design principles:
- A single failing candidate never aborts the full replay.
- The runner is stateless; callers decide what to persist.
- Works with any GenerationProvider (fake or real).
"""
import uuid
from datetime import datetime
from typing import Callable, Optional

from app.evaluation.base import BaseEvaluator
from app.providers.base import GenerationProvider
from app.providers.fake import FakeProvider
from app.replay.replay_models import ReplayCandidate, ReplayRequest, ReplayResult
from app.schemas import EvidenceLevel, UsageRecord, ValidationStatus
from app.utils.evidence import adjust_confidence_for_evidence


def build_replay_requests(records: list[UsageRecord]) -> list[ReplayRequest]:
    """Convert in-memory UsageRecord objects into ReplayRequest objects.

    Assigns synthetic UUIDs as original_record_id — suitable for in-memory
    analysis but the IDs cannot be traced back to database rows. Use
    build_replay_requests_from_rows when DB traceability is required.
    """
    requests = []
    for record in records:
        requests.append(ReplayRequest(
            original_record_id=str(uuid.uuid4()),
            prompt=record.prompt,
            original_response=record.response,
            original_model=record.model,
            original_cost=record.cost,
            task_type=record.task_type.value if record.task_type else None,
            feedback=record.feedback,
            timestamp=record.timestamp,
        ))
    return requests


def build_replay_requests_from_rows(rows: list[dict]) -> list[ReplayRequest]:
    """Convert usage_records DB rows into ReplayRequest objects.

    Uses the DB row `id` as original_record_id so replay results can be
    traced back to specific rows in the database.
    """
    requests = []
    for row in rows:
        requests.append(ReplayRequest(
            original_record_id=str(row["id"]),
            prompt=row["prompt"],
            original_response=row["response"],
            original_model=row["model"],
            original_cost=row["cost"],
            task_type=row.get("task_type"),
            feedback=row.get("feedback"),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        ))
    return requests


class ReplayRunner:

    def __init__(
        self,
        evaluator: BaseEvaluator,
        provider: Optional[GenerationProvider] = None,
        provider_factory: Optional[Callable[[str], GenerationProvider]] = None,
    ):
        if provider_factory is not None:
            self._factory = provider_factory
        elif provider is not None:
            self._factory = lambda model: provider
        else:
            self._factory = lambda model: FakeProvider()
        self.evaluator = evaluator

    def run(
        self,
        requests: list[ReplayRequest],
        candidates: list[ReplayCandidate],
    ) -> list[ReplayResult]:
        """
        Run every request against every enabled candidate.
        Returns one ReplayResult per (request, candidate) pair.
        Errors are captured in result.error_message; they do not raise.
        """
        results = []
        for req in requests:
            for candidate in candidates:
                if not candidate.enabled:
                    continue
                if candidate.model == req.original_model:
                    continue  # never compare a model against itself
                results.append(self._run_single(req, candidate))
        return results

    def _run_single(self, req: ReplayRequest, candidate: ReplayCandidate) -> ReplayResult:
        replay_id = str(uuid.uuid4())
        try:
            provider = self._factory(candidate.model)
            response = provider.generate(req.prompt, candidate.model)
            quality = self.evaluator.evaluate(
                prompt=req.prompt,
                original_response=req.original_response,
                candidate_response=response.text,
                task_type=req.task_type,
                feedback=req.feedback,
            )
            evidence_level = (
                EvidenceLevel.LLM_JUDGE
                if quality.method == "llm_judge"
                else EvidenceLevel.OBSERVED_REPLAY
            )
            return ReplayResult(
                replay_id=replay_id,
                original_record_id=req.original_record_id,
                candidate_provider=candidate.provider,
                candidate_model=candidate.model,
                candidate_response=response.text,
                estimated_cost=response.estimated_cost,
                latency_ms=response.latency_ms,
                quality_score=quality.score,
                quality_method=quality.method,
                quality_explanation=quality.explanation,
                quality_confidence=quality.confidence,
                quality_flags=quality.flags,
                error_message=None,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_source=response.cost_source,
                latency_source=response.latency_source,
                evidence_level=evidence_level,
                evidence_summary=(
                    f"Replayed on {candidate.model} and scored by {quality.method} evaluator. "
                    f"Quality score: {quality.score:.2f} (confidence: {quality.confidence:.0%})."
                ),
                limitations=[
                    f"Quality scored by {quality.method} evaluator, not human review.",
                    "Latency measured in test conditions; production latency may vary.",
                ],
                validation_status=ValidationStatus.EVALUATOR_SCORED,
                confidence_score=adjust_confidence_for_evidence(
                    quality.confidence, evidence_level, ValidationStatus.EVALUATOR_SCORED
                ),
            )
        except Exception as exc:
            return ReplayResult(
                replay_id=replay_id,
                original_record_id=req.original_record_id,
                candidate_provider=candidate.provider,
                candidate_model=candidate.model,
                candidate_response="",
                estimated_cost=0.0,
                latency_ms=0.0,
                quality_score=0.0,
                quality_method="error",
                quality_explanation=f"Evaluation failed: {exc}",
                quality_confidence=0.0,
                quality_flags=["evaluation_error"],
                error_message=str(exc),
                input_tokens=0,
                output_tokens=0,
                cost_source="missing",
                latency_source="missing",
                evidence_level=EvidenceLevel.HEURISTIC,
                evidence_summary=f"Replay attempt failed: {exc}. No usable output was produced.",
                limitations=["Replay execution failed; no quality data available."],
                validation_status=ValidationStatus.NOT_VALIDATED,
                confidence_score=0.0,
            )
