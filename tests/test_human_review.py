"""Tests for human review CSV export/import workflow."""
import csv
import io
import uuid

import pytest

from app.replay.human_review import (
    VALID_REVIEWER_LABELS,
    apply_human_reviews,
    export_replay_results_csv,
    parse_import_csv,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _rr(
    replay_id: str | None = None,
    candidate_model: str = "gpt-4o-mini",
    quality_score: float = 0.85,
    quality_confidence: float = 0.80,
    original_record_id: str = "42",
) -> dict:
    return {
        "replay_id": replay_id or str(uuid.uuid4()),
        "original_record_id": original_record_id,
        "candidate_model": candidate_model,
        "candidate_response": "Candidate output here.",
        "quality_score": quality_score,
        "quality_method": "heuristic",
        "quality_explanation": "Close match.",
        "quality_confidence": quality_confidence,
        "estimated_cost": 0.002,
        "cost_source": "estimated_catalog",
        "latency_ms": 200.0,
        "latency_source": "fake",
    }


def _usage(record_id: str = "42", task_type: str = "summarization") -> dict:
    return {
        "id": int(record_id) if record_id.isdigit() else 1,
        "response": "Original response text.",
        "task_type": task_type,
    }


# ── Export ────────────────────────────────────────────────────────────────────

def test_export_returns_csv_string():
    run_id = "run-A"
    rr = _rr(original_record_id="1")
    csv_text = export_replay_results_csv(run_id, [rr], [_usage("1")])
    assert isinstance(csv_text, str)
    assert "replay_result_id" in csv_text  # header row


def test_export_header_contains_required_columns():
    csv_text = export_replay_results_csv("run-A", [_rr()], [])
    reader = csv.DictReader(io.StringIO(csv_text))
    assert reader.fieldnames is not None
    for col in ("replay_result_id", "candidate_model", "quality_score", "cost_source", "latency_source"):
        assert col in reader.fieldnames


def test_export_row_values_are_correct():
    rr = _rr(replay_id="rr-99", candidate_model="claude-3-haiku-20240307", quality_score=0.91, original_record_id="5")
    usage = _usage("5", task_type="classification")
    csv_text = export_replay_results_csv("run-B", [rr], [usage])
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["replay_result_id"] == "rr-99"
    assert row["candidate_model"] == "claude-3-haiku-20240307"
    assert row["quality_score"] == "0.91"
    assert row["task_type"] == "classification"
    assert row["original_response"] == "Original response text."


def test_export_multiple_rows():
    rrs = [_rr(replay_id=f"rr-{i}", original_record_id=str(i)) for i in range(3)]
    usage = [_usage(str(i)) for i in range(3)]
    csv_text = export_replay_results_csv("run-C", rrs, usage)
    reader = csv.DictReader(io.StringIO(csv_text))
    assert len(list(reader)) == 3


def test_export_missing_usage_record_leaves_original_response_empty():
    rr = _rr(original_record_id="999")
    csv_text = export_replay_results_csv("run-D", [rr], [])  # no matching usage record
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert rows[0]["original_response"] == ""


# ── Import/parse ──────────────────────────────────────────────────────────────

def _make_import_csv(rows: list[dict]) -> str:
    from app.replay.human_review import EXPORT_FIELDS
    fields = EXPORT_FIELDS + ["reviewer_label", "reviewer_notes"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def _row(replay_id: str, label: str, notes: str = "") -> dict:
    return {
        "replay_result_id": replay_id,
        "replay_run_id": "run-A",
        "original_record_id": "1",
        "candidate_model": "gpt-4o-mini",
        "task_type": "summarization",
        "original_response": "orig",
        "candidate_response": "cand",
        "quality_score": "0.85",
        "quality_method": "heuristic",
        "evaluator_explanation": "Good.",
        "estimated_cost": "0.002",
        "cost_source": "estimated_catalog",
        "latency_ms": "200",
        "latency_source": "fake",
        "reviewer_label": label,
        "reviewer_notes": notes,
    }


def test_parse_valid_import_csv():
    csv_text = _make_import_csv([_row("rr-1", "candidate_better")])
    rows = parse_import_csv(csv_text)
    assert len(rows) == 1
    assert rows[0]["replay_id"] == "rr-1"
    assert rows[0]["reviewer_label"] == "candidate_better"


def test_parse_all_valid_labels():
    rows_data = [_row(f"rr-{i}", label) for i, label in enumerate(sorted(VALID_REVIEWER_LABELS))]
    csv_text = _make_import_csv(rows_data)
    rows = parse_import_csv(csv_text)
    assert len(rows) == len(VALID_REVIEWER_LABELS)


def test_parse_invalid_label_raises():
    csv_text = _make_import_csv([_row("rr-1", "thumbs_up")])
    with pytest.raises(ValueError, match="invalid reviewer_label"):
        parse_import_csv(csv_text)


def test_parse_missing_replay_id_raises():
    csv_text = _make_import_csv([_row("", "candidate_better")])
    with pytest.raises(ValueError, match="missing replay_result_id"):
        parse_import_csv(csv_text)


def test_parse_multiple_rows_including_invalid_collects_all_errors():
    rows_data = [
        _row("rr-1", "candidate_better"),
        _row("", "candidate_worse"),         # invalid: missing id
        _row("rr-3", "invalid_label"),       # invalid: bad label
    ]
    csv_text = _make_import_csv(rows_data)
    with pytest.raises(ValueError) as exc_info:
        parse_import_csv(csv_text)
    # Both errors should appear in the message
    msg = str(exc_info.value)
    assert "missing replay_result_id" in msg
    assert "invalid reviewer_label" in msg


# ── apply_human_reviews ───────────────────────────────────────────────────────

def test_apply_reviews_returns_update_dicts():
    rr = _rr(replay_id="rr-X", quality_confidence=0.80)
    rows = [{"replay_id": "rr-X", "replay_run_id": "run-A", "reviewer_label": "candidate_better", "reviewer_notes": "Looks great"}]
    updates = apply_human_reviews(rows, [rr])
    assert len(updates) == 1
    upd = updates[0]
    assert upd["replay_id"] == "rr-X"
    assert upd["evidence_level"] == "human_reviewed"
    assert upd["validation_status"] == "human_reviewed"
    assert 0.0 <= upd["confidence_score"] <= 1.0


def test_apply_reviews_unacceptable_gets_zero_confidence():
    rr = _rr(replay_id="rr-Y", quality_confidence=0.90)
    rows = [{"replay_id": "rr-Y", "replay_run_id": "run-A", "reviewer_label": "unacceptable", "reviewer_notes": ""}]
    updates = apply_human_reviews(rows, [rr])
    upd = updates[0]
    assert upd["confidence_score"] == 0.0


def test_apply_reviews_equivalent_gets_high_confidence():
    rr = _rr(replay_id="rr-Z", quality_confidence=0.85)
    rows = [{"replay_id": "rr-Z", "replay_run_id": "run-A", "reviewer_label": "candidate_equivalent", "reviewer_notes": ""}]
    updates = apply_human_reviews(rows, [rr])
    upd = updates[0]
    assert upd["confidence_score"] > 0.5


def test_apply_reviews_missing_rr_uses_default_confidence():
    rows = [{"replay_id": "rr-missing", "replay_run_id": "run-A", "reviewer_label": "candidate_better", "reviewer_notes": ""}]
    updates = apply_human_reviews(rows, [])  # no matching replay results
    assert len(updates) == 1
    assert 0.0 <= updates[0]["confidence_score"] <= 1.0


# ── DB integration: human review roundtrip ────────────────────────────────────

def test_human_review_db_roundtrip(db):
    from app.database import get_human_reviews_for_run, save_human_review
    from datetime import datetime, timezone

    run_id = "review-run-1"
    save_human_review(
        replay_run_id=run_id,
        replay_id="rr-db-1",
        reviewer_label="candidate_better",
        reviewer_notes="All good.",
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )
    reviews = get_human_reviews_for_run(run_id)
    assert len(reviews) == 1
    assert reviews[0]["reviewer_label"] == "candidate_better"
    assert reviews[0]["reviewer_notes"] == "All good."


def test_human_review_upsert_on_rereview(db):
    from app.database import get_human_reviews_for_run, save_human_review
    from datetime import datetime, timezone

    run_id = "review-run-2"
    now = datetime.now(timezone.utc).isoformat()
    save_human_review(run_id, "rr-1", "candidate_better", "", now)
    save_human_review(run_id, "rr-1", "candidate_worse", "Changed mind.", now)  # upsert

    reviews = get_human_reviews_for_run(run_id)
    assert len(reviews) == 1
    assert reviews[0]["reviewer_label"] == "candidate_worse"


def test_update_replay_result_evidence(db):
    from app.database import update_replay_result_evidence
    from app.replay.replay_models import ReplayResult
    from app.replay.replay_store import get_replay_results, save_replay_results, save_replay_run
    from app.schemas import EvidenceLevel, ValidationStatus

    run_id = "rr-ev-1"
    rr_id = str(uuid.uuid4())
    save_replay_run(run_id, "audit-X", ["gpt-4o-mini"], record_count=1)

    result = ReplayResult(
        replay_id=rr_id,
        original_record_id="1",
        candidate_provider="openai",
        candidate_model="gpt-4o-mini",
        candidate_response="ok",
        estimated_cost=0.001,
        latency_ms=100.0,
        quality_score=0.85,
        quality_method="heuristic",
        evidence_level=EvidenceLevel.OBSERVED_REPLAY,
        validation_status=ValidationStatus.EVALUATOR_SCORED,
        confidence_score=0.70,
    )
    save_replay_results(run_id, [result])

    update_replay_result_evidence(
        replay_id=rr_id,
        replay_run_id=run_id,
        evidence_level="human_reviewed",
        validation_status="human_reviewed",
        confidence_score=0.92,
    )

    rows = get_replay_results(run_id)
    assert rows[0]["evidence_level"] == "human_reviewed"
    assert rows[0]["validation_status"] == "human_reviewed"
    assert rows[0]["confidence_score"] == pytest.approx(0.92)
