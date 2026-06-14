"""Tests for POST /sdk/events backend endpoint."""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _event_payload(**kwargs) -> dict:
    defaults = dict(
        event_id=str(uuid.uuid4()),
        timestamp="2024-01-15T12:00:00+00:00",
        provider="openai",
        model="gpt-4o-mini",
        workflow="test_workflow",
        task_type="summarization",
        status="success",
        latency_ms=145.0,
        input_tokens=80,
        output_tokens=25,
        estimated_cost=0.0000175,
        team="engineering",
    )
    defaults.update(kwargs)
    return defaults


# ── Happy path ────────────────────────────────────────────────────────────────

def test_ingest_single_event_returns_200(db):
    payload = {"events": [_event_payload()]}
    resp = client.post("/sdk/events", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] == 1
    assert body["rejected"] == 0
    assert body["errors"] == []


def test_ingest_batch_of_events(db):
    events = [_event_payload() for _ in range(5)]
    resp = client.post("/sdk/events", json={"events": events})
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] == 5
    assert body["rejected"] == 0


def test_ingest_metadata_only_event(db):
    """Event without prompt or response must be accepted."""
    payload = {"events": [_event_payload(prompt=None, response=None)]}
    resp = client.post("/sdk/events", json=payload)
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 1


def test_ingest_event_with_all_fields(db):
    event = _event_payload(
        prompt="Summarize this document.",
        response="The document covers Q3 results.",
        user_id_hash="sha256-abc",
        team="product",
        business_unit="growth",
        process_name="weekly_report",
        tool_name="report_bot",
        environment="production",
        risk_level="medium",
        value_metric_name="reports_generated",
        value_metric_value=1.0,
        feedback_score=0.9,
        feedback_label="accepted",
        error_message=None,
    )
    resp = client.post("/sdk/events", json={"events": [event]})
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 1


def test_ingest_duplicate_event_id_is_idempotent(db):
    """Duplicate event_ids must be silently ignored, not rejected."""
    event = _event_payload()
    client.post("/sdk/events", json={"events": [event]})
    resp = client.post("/sdk/events", json={"events": [event]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["rejected"] == 0
    assert body["duplicate"] == 1
    assert body["accepted"] == 0


def test_ingest_empty_batch_returns_zero_accepted(db):
    resp = client.post("/sdk/events", json={"events": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] == 0
    assert body["rejected"] == 0


# ── Validation ────────────────────────────────────────────────────────────────

def test_ingest_missing_events_field_returns_422(db):
    resp = client.post("/sdk/events", json={"bad_field": []})
    assert resp.status_code == 422


def test_ingest_event_missing_event_id_returns_422(db):
    event = _event_payload()
    del event["event_id"]
    resp = client.post("/sdk/events", json={"events": [event]})
    assert resp.status_code == 422


# ── Error reporting ───────────────────────────────────────────────────────────

# ── Duplicate counting ────────────────────────────────────────────────────────

def test_first_event_counts_as_accepted(db):
    """A fresh event_id must be counted in accepted, not duplicate."""
    resp = client.post("/sdk/events", json={"events": [_event_payload()]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] == 1
    assert body["duplicate"] == 0
    assert body["rejected"] == 0


def test_duplicate_event_counted_separately(db):
    """Resending the same event_id must increment duplicate, not accepted."""
    event = _event_payload()
    client.post("/sdk/events", json={"events": [event]})
    resp = client.post("/sdk/events", json={"events": [event]})
    body = resp.json()
    assert body["accepted"] == 0
    assert body["duplicate"] == 1
    assert body["rejected"] == 0


def test_mixed_new_and_duplicate_counted_correctly(db):
    """Batch with 2 new + 1 duplicate → accepted=2, duplicate=1."""
    event_a = _event_payload()
    event_b = _event_payload()
    client.post("/sdk/events", json={"events": [event_a]})
    resp = client.post("/sdk/events", json={"events": [event_a, event_b, _event_payload()]})
    body = resp.json()
    assert body["accepted"] == 2
    assert body["duplicate"] == 1
    assert body["rejected"] == 0


def test_ingest_partial_failure_reports_accepted_and_rejected(db, monkeypatch):
    """If one event fails to save, the rest should still be accepted."""
    import app.main as main_module
    from app.sdk_ingestion import save_sdk_event as real_save

    call_count = [0]

    def patched_save(event):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("DB write failed")
        return real_save(event)

    # Patch the name as imported into main.py
    monkeypatch.setattr(main_module, "save_sdk_event", patched_save)

    events = [_event_payload() for _ in range(3)]
    resp = client.post("/sdk/events", json={"events": events})
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] == 2
    assert body["rejected"] == 1
    assert len(body["errors"]) == 1


# ── Feedback endpoint ─────────────────────────────────────────────────────────

def test_patch_feedback_updates_event(db):
    """PATCH /sdk/events/{id}/feedback must persist score, label, and notes."""
    event = _event_payload()
    client.post("/sdk/events", json={"events": [event]})

    resp = client.patch(
        f"/sdk/events/{event['event_id']}/feedback",
        json={"feedback_score": 0.85, "feedback_label": "accepted", "notes": "looks good"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["event_id"] == event["event_id"]
    assert body["updated"] is True


def test_patch_feedback_partial_fields_allowed(db):
    """PATCH with only notes (no score/label) must succeed."""
    event = _event_payload()
    client.post("/sdk/events", json={"events": [event]})

    resp = client.patch(
        f"/sdk/events/{event['event_id']}/feedback",
        json={"notes": "reviewed manually"},
    )
    assert resp.status_code == 200
    assert resp.json()["updated"] is True


def test_patch_feedback_unknown_event_returns_404(db):
    """PATCH for an event_id not in the database must return 404."""
    resp = client.patch(
        "/sdk/events/nonexistent-event-id/feedback",
        json={"feedback_score": 1.0},
    )
    assert resp.status_code == 404


def test_patch_feedback_score_only(db):
    """PATCH with score only (no label or notes) must be accepted."""
    event = _event_payload()
    client.post("/sdk/events", json={"events": [event]})

    resp = client.patch(
        f"/sdk/events/{event['event_id']}/feedback",
        json={"feedback_score": 0.5},
    )
    assert resp.status_code == 200


# ── Identity field validation ─────────────────────────────────────────────────

def test_backend_rejects_empty_workflow(db):
    """An event with blank workflow must be rejected with 422."""
    event = _event_payload(workflow="")
    resp = client.post("/sdk/events", json={"events": [event]})
    assert resp.status_code == 422


def test_backend_rejects_whitespace_workflow(db):
    """An event with whitespace-only workflow must be rejected with 422."""
    event = _event_payload(workflow="   ")
    resp = client.post("/sdk/events", json={"events": [event]})
    assert resp.status_code == 422


def test_backend_rejects_empty_task_type(db):
    """An event with blank task_type must be rejected with 422."""
    event = _event_payload(task_type="")
    resp = client.post("/sdk/events", json={"events": [event]})
    assert resp.status_code == 422
