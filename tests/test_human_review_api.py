"""API-level tests for human review export/import endpoints."""
import csv
import io

import pytest

VALID_CSV = b"""prompt,response,timestamp,model,cost
"Summarize this report","Summary here",2024-01-15 09:00:00,gpt-4o,0.0342
"Classify this ticket","Category: billing",2024-01-16 10:00:00,gpt-4o,0.0012
"""


def _upload_and_replay(client):
    """Helper: upload, generate, run replay, simulate. Returns replay_run_id."""
    resp = client.post(
        "/audits/upload",
        files={"file": ("data.csv", VALID_CSV, "text/csv")},
    )
    audit_id = resp.json()["audit_run_id"]

    client.post(f"/audits/{audit_id}/generate")

    resp = client.post(
        f"/audits/{audit_id}/replay/run",
        json={"candidate_models": ["gpt-4o-mini"], "max_records": 2},
    )
    assert resp.status_code == 200
    return resp.json()["replay_run_id"]


# ── Export ────────────────────────────────────────────────────────────────────

def test_export_returns_csv(client):
    run_id = _upload_and_replay(client)
    resp = client.get(f"/replay/{run_id}/review/export")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")


def test_export_contains_header_row(client):
    run_id = _upload_and_replay(client)
    resp = client.get(f"/replay/{run_id}/review/export")
    text = resp.text
    reader = csv.DictReader(io.StringIO(text))
    assert "replay_result_id" in (reader.fieldnames or [])
    assert "candidate_model" in (reader.fieldnames or [])


def test_export_has_correct_number_of_rows(client):
    run_id = _upload_and_replay(client)
    resp = client.get(f"/replay/{run_id}/review/export")
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = list(reader)
    assert len(rows) > 0


def test_export_404_for_unknown_run(client):
    resp = client.get("/replay/nonexistent-run/review/export")
    assert resp.status_code == 404


# ── Import ────────────────────────────────────────────────────────────────────

def _make_import_csv(export_text: str, label: str = "candidate_better") -> bytes:
    """Add reviewer columns to an export CSV."""
    reader = csv.DictReader(io.StringIO(export_text))
    rows = list(reader)
    if not rows:
        return b""

    fieldnames = (reader.fieldnames or []) + ["reviewer_label", "reviewer_notes"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        row["reviewer_label"] = label
        row["reviewer_notes"] = "Looks good."
        writer.writerow(row)
    return buf.getvalue().encode()


def test_import_accepts_valid_csv(client):
    run_id = _upload_and_replay(client)
    export_resp = client.get(f"/replay/{run_id}/review/export")
    import_csv = _make_import_csv(export_resp.text, "candidate_equivalent")

    resp = client.post(
        f"/replay/{run_id}/review/import",
        files={"file": ("review.csv", import_csv, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "complete"
    assert data["reviews_applied"] > 0


def test_import_updates_evidence_in_db(client):
    from app.replay.replay_store import get_replay_results
    from app.config import settings
    import app.database as db_mod

    run_id = _upload_and_replay(client)
    export_resp = client.get(f"/replay/{run_id}/review/export")
    import_csv = _make_import_csv(export_resp.text, "candidate_better")

    client.post(
        f"/replay/{run_id}/review/import",
        files={"file": ("review.csv", import_csv, "text/csv")},
    )

    results = get_replay_results(run_id)
    for r in results:
        assert r["evidence_level"] == "human_reviewed"
        assert r["validation_status"] == "human_reviewed"


def test_import_invalid_label_returns_422(client):
    run_id = _upload_and_replay(client)
    export_resp = client.get(f"/replay/{run_id}/review/export")
    import_csv = _make_import_csv(export_resp.text, "thumbs_up")  # invalid

    resp = client.post(
        f"/replay/{run_id}/review/import",
        files={"file": ("review.csv", import_csv, "text/csv")},
    )
    assert resp.status_code == 422


def test_import_404_for_unknown_run(client):
    csv_bytes = b"replay_result_id,reviewer_label\nrr-1,candidate_better\n"
    resp = client.post(
        "/replay/nonexistent-run/review/import",
        files={"file": ("review.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 404


def test_import_evaluator_mode_task_router_works(client):
    """Verify task_router mode runs without errors (tests the evaluator wiring)."""
    resp = client.post(
        "/audits/upload",
        files={"file": ("data.csv", VALID_CSV, "text/csv")},
    )
    audit_id = resp.json()["audit_run_id"]
    client.post(f"/audits/{audit_id}/generate")

    resp = client.post(
        f"/audits/{audit_id}/replay/run",
        json={
            "candidate_models": ["gpt-4o-mini"],
            "max_records": 2,
            "evaluator_mode": "task_router",
        },
    )
    assert resp.status_code == 200
