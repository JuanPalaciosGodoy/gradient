"""
Human review workflow: CSV export and import for replay result labelling.

Export flow:
  GET /replay/{id}/review/export  →  CSV file for reviewers

Import flow:
  POST /replay/{id}/review/import  →  updates evidence fields on replay_results
"""
import csv
import io
from datetime import datetime, timezone
from typing import Optional

VALID_REVIEWER_LABELS = frozenset({
    "candidate_better",
    "candidate_equivalent",
    "candidate_worse",
    "unacceptable",
})

EXPORT_FIELDS = [
    "replay_result_id",
    "replay_run_id",
    "original_record_id",
    "candidate_model",
    "task_type",
    "original_response",
    "candidate_response",
    "quality_score",
    "quality_method",
    "evaluator_explanation",
    "estimated_cost",
    "cost_source",
    "latency_ms",
    "latency_source",
]

IMPORT_FIELDS = EXPORT_FIELDS + ["reviewer_label", "reviewer_notes"]


def export_replay_results_csv(
    replay_run_id: str,
    replay_results: list[dict],
    usage_records: list[dict],
) -> str:
    """Return a CSV string of replay results for human review.

    Joins replay_results with usage_records to include original_response.
    """
    orig_by_id: dict[str, dict] = {str(r["id"]): r for r in usage_records}

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()

    for rr in replay_results:
        orig = orig_by_id.get(str(rr.get("original_record_id", "")), {})
        writer.writerow({
            "replay_result_id": rr.get("replay_id", ""),
            "replay_run_id": replay_run_id,
            "original_record_id": rr.get("original_record_id", ""),
            "candidate_model": rr.get("candidate_model", ""),
            "task_type": orig.get("task_type", ""),
            "original_response": orig.get("response", ""),
            "candidate_response": rr.get("candidate_response", ""),
            "quality_score": rr.get("quality_score", ""),
            "quality_method": rr.get("quality_method", ""),
            "evaluator_explanation": rr.get("quality_explanation", ""),
            "estimated_cost": rr.get("estimated_cost", ""),
            "cost_source": rr.get("cost_source", "estimated_catalog"),
            "latency_ms": rr.get("latency_ms", ""),
            "latency_source": rr.get("latency_source", "fake"),
        })

    return buf.getvalue()


def parse_import_csv(csv_content: str) -> list[dict]:
    """Parse and validate a human-review CSV import.

    Returns a list of validated row dicts.
    Raises ValueError listing all invalid rows.
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    errors: list[str] = []
    rows: list[dict] = []

    for i, row in enumerate(reader, start=2):  # row 1 = header
        replay_id = row.get("replay_result_id", "").strip()
        label = row.get("reviewer_label", "").strip().lower()

        if not replay_id:
            errors.append(f"Row {i}: missing replay_result_id")
            continue
        if label not in VALID_REVIEWER_LABELS:
            errors.append(
                f"Row {i}: invalid reviewer_label '{label}'. "
                f"Must be one of: {', '.join(sorted(VALID_REVIEWER_LABELS))}"
            )
            continue

        rows.append({
            "replay_id": replay_id,
            "replay_run_id": row.get("replay_run_id", "").strip(),
            "reviewer_label": label,
            "reviewer_notes": row.get("reviewer_notes", "").strip(),
        })

    if errors:
        raise ValueError(f"Import validation failed:\n" + "\n".join(errors))

    return rows


def _confidence_from_label(base_confidence: float, label: str) -> float:
    """Adjust confidence based on human judgement."""
    from app.schemas import EvidenceLevel, ValidationStatus
    from app.utils.evidence import adjust_confidence_for_evidence

    label_scores = {
        "candidate_better": 1.0,
        "candidate_equivalent": 0.9,
        "candidate_worse": 0.4,
        "unacceptable": 0.0,
    }
    quality_score = label_scores.get(label, base_confidence)
    return adjust_confidence_for_evidence(
        quality_score, EvidenceLevel.HUMAN_REVIEWED, ValidationStatus.HUMAN_REVIEWED
    )


def apply_human_reviews(
    rows: list[dict],
    replay_results: list[dict],
) -> list[dict]:
    """Compute the DB updates to apply given human review rows.

    Returns a list of update dicts, one per reviewed result.
    """
    from app.schemas import EvidenceLevel, ValidationStatus

    by_replay_id = {rr.get("replay_id"): rr for rr in replay_results}
    updates: list[dict] = []

    for row in rows:
        replay_id = row["replay_id"]
        rr = by_replay_id.get(replay_id)
        base_confidence = float(rr.get("quality_confidence", 0.5)) if rr else 0.5
        new_confidence = _confidence_from_label(base_confidence, row["reviewer_label"])

        updates.append({
            "replay_id": replay_id,
            "replay_run_id": row["replay_run_id"],
            "reviewer_label": row["reviewer_label"],
            "reviewer_notes": row["reviewer_notes"],
            "evidence_level": EvidenceLevel.HUMAN_REVIEWED.value,
            "validation_status": ValidationStatus.HUMAN_REVIEWED.value,
            "confidence_score": new_confidence,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        })

    return updates
