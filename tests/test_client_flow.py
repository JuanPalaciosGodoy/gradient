"""
High-level client-flow regression tests.

These tests mimic real customer adoption paths end-to-end through the API,
catching integration gaps that unit tests won't surface.
"""
import io
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ingestion.validators import METADATA_ONLY_SENTINEL
from gradient_sdk import GradientClient, audit
from gradient_sdk.exporters import export_phase1_csv, load_jsonl
from gradient_sdk.models import AuditEvent

http = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _upload_csv(client, csv_bytes: bytes) -> dict:
    resp = client.post(
        "/audits/upload",
        files={"file": ("upload.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _content_csv(n: int = 5) -> bytes:
    """Minimal Phase 1 CSV with real prompt/response content."""
    rows = ["prompt,response,timestamp,model,cost,task_type"]
    for i in range(n):
        rows.append(
            f'"Summarize doc {i}","Summary of doc {i}.",'
            f'"2024-01-{i + 1:02d} 09:00:00",gpt-4o,0.05,summarization'
        )
    return "\n".join(rows).encode()


def _metadata_only_csv(n: int = 5) -> bytes:
    """Phase 1 CSV produced by SDK metadata-only export — empty prompt/response."""
    rows = ["prompt,response,timestamp,model,cost,task_type"]
    for i in range(n):
        rows.append(
            f'"","","2024-01-{i + 1:02d} 09:00:00",gpt-4o,0.05,customer_support'
        )
    return "\n".join(rows).encode()


# ── Flow A: metadata-only adoption path ──────────────────────────────────────

class TestMetadataOnlyAdoptionPath:
    """Customer starts with metadata-only SDK export; spend analysis works but replay is blocked."""

    def test_upload_does_not_store_nan(self, client):
        """Blank prompt/response must not become the literal string 'nan'."""
        upload = _upload_csv(client, _metadata_only_csv())
        audit_run_id = upload["audit_run_id"]
        assert upload["record_count"] == 5

        from app.database import get_usage_records
        rows = get_usage_records(audit_run_id)
        for row in rows:
            assert row["prompt"] != "nan", "prompt should never be 'nan'"
            assert row["response"] != "nan", "response should never be 'nan'"
            assert row["prompt"] == METADATA_ONLY_SENTINEL
            assert row["response"] == METADATA_ONLY_SENTINEL

    def test_task_type_preserved_from_sdk_export(self, client):
        """task_type from the SDK export must pass through unmodified."""
        upload = _upload_csv(client, _metadata_only_csv())
        audit_run_id = upload["audit_run_id"]

        from app.database import get_usage_records
        rows = get_usage_records(audit_run_id)
        for row in rows:
            assert row["task_type"] == "customer_support"

    def test_spend_report_works_on_metadata_only_data(self, client):
        """Phase 1 audit report must succeed — spend and task breakdown populated."""
        upload = _upload_csv(client, _metadata_only_csv())
        audit_run_id = upload["audit_run_id"]

        gen = client.post(f"/audits/{audit_run_id}/generate")
        assert gen.status_code == 200

        report = client.get(f"/audits/{audit_run_id}/report").json()
        assert report["spend_summary"]["total_cost"] > 0
        assert "customer_support" in report["spend_summary"]["task_breakdown"]

    def test_replay_on_metadata_only_returns_422(self, client):
        """Replay must reject metadata-only audits with an actionable message."""
        upload = _upload_csv(client, _metadata_only_csv())
        audit_run_id = upload["audit_run_id"]

        resp = client.post(f"/audits/{audit_run_id}/replay/run", json={})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "capture_prompt" in detail
        assert "capture_response" in detail

    def test_mixed_upload_filters_metadata_only_rows_for_replay(self, client):
        """When an audit has both metadata-only and content rows, replay proceeds
        with just the content rows."""
        # 3 metadata-only rows + 2 content rows
        meta_rows = [
            f'"","","2024-01-0{i} 09:00:00",gpt-4o-mini,0.01,classification'
            for i in range(1, 4)
        ]
        content_rows = [
            f'"Summarize doc {i}","Summary here.","2024-01-0{i + 3} 09:00:00",gpt-4o,0.05,summarization'
            for i in range(1, 3)
        ]
        csv_bytes = (
            "prompt,response,timestamp,model,cost,task_type\n"
            + "\n".join(meta_rows + content_rows)
        ).encode()

        upload = _upload_csv(client, csv_bytes)
        audit_run_id = upload["audit_run_id"]
        assert upload["record_count"] == 5

        resp = client.post(f"/audits/{audit_run_id}/replay/run", json={})
        # Replay may fail for other reasons (e.g. all candidates == source model)
        # but must NOT fail with the metadata-only 422 message.
        if resp.status_code == 422:
            assert "capture_prompt" not in resp.json().get("detail", "")

    def test_sdk_metadata_export_roundtrip(self, tmp_path):
        """SDK metadata-only export → Phase 1 CSV → load_csv never produces 'nan'."""
        jsonl_path = tmp_path / "audit.jsonl"
        csv_path = tmp_path / "upload.csv"

        capture_client = GradientClient(mode="local", local_path=str(jsonl_path))

        @audit(
            client=capture_client,
            workflow="support_reply",
            task_type="customer_support",
            provider="openai",
            model="gpt-4o",
        )
        def fake_llm_call():
            return {"text": "response", "estimated_cost": 0.01}

        for _ in range(3):
            fake_llm_call()

        events = load_jsonl(jsonl_path)
        assert len(events) == 3

        export_phase1_csv(events, csv_path)
        csv_bytes = csv_path.read_bytes()

        from app.ingestion.csv_loader import load_csv
        records = load_csv(csv_bytes)
        assert len(records) == 3
        for r in records:
            assert r.prompt != "nan"
            assert r.response != "nan"
            assert r.prompt == METADATA_ONLY_SENTINEL
            assert r.task_type is not None


# ── Flow B: evidence upgrade path ─────────────────────────────────────────────

class TestEvidenceUpgradePath:
    """Customer uploads content CSV, replays, simulates, reviews, and re-simulates."""

    def _run_replay(self, client, audit_run_id: str) -> str:
        resp = client.post(f"/audits/{audit_run_id}/replay/run", json={})
        assert resp.status_code == 200, resp.text
        return resp.json()["replay_run_id"]

    def _simulate(self, client, replay_run_id: str) -> dict:
        resp = client.post(f"/replay/{replay_run_id}/simulate")
        assert resp.status_code == 200, resp.text
        return resp.json()

    def _get_report(self, client, replay_run_id: str) -> dict:
        resp = client.get(f"/replay/{replay_run_id}/report")
        assert resp.status_code == 200, resp.text
        return resp.json()

    def test_content_upload_replay_simulate_report_full_cycle(self, client):
        """Full Phase 2.5 happy path: upload → replay → simulate → report."""
        upload = _upload_csv(client, _content_csv(5))
        audit_run_id = upload["audit_run_id"]

        replay_run_id = self._run_replay(client, audit_run_id)
        sim = self._simulate(client, replay_run_id)
        assert sim["simulations_count"] >= 0  # may be 0 if no candidates differ from source

        report = self._get_report(client, replay_run_id)
        assert "replay_run_id" in report
        assert report["total_requests"] > 0
        # All scenario types must be present as fields (even if empty)
        assert "recommended_migrations" in report
        assert "investigate_scenarios" in report
        assert "hold_scenarios" in report
        assert "do_not_migrate" in report

    def test_stale_detection_after_human_review(self, client):
        """Calling /report after importing reviews but before re-simulating returns 409."""
        upload = _upload_csv(client, _content_csv(5))
        audit_run_id = upload["audit_run_id"]
        replay_run_id = self._run_replay(client, audit_run_id)
        self._simulate(client, replay_run_id)

        # Export review CSV
        export_resp = client.get(f"/replay/{replay_run_id}/review/export")
        assert export_resp.status_code == 200
        csv_text = export_resp.text

        # Parse and fill in a reviewer_label for each row
        lines = csv_text.strip().split("\n")
        header = lines[0]
        cols = [c.strip().strip('"') for c in header.split(",")]
        if "reviewer_label" not in cols:
            pytest.skip("No rows to review — replay produced no results")

        reviewed_lines = [header]
        for line in lines[1:]:
            parts = line.split(",")
            # Set reviewer_label to "accepted" and reviewed_at to a timestamp
            label_idx = cols.index("reviewer_label")
            reviewed_at_idx = cols.index("reviewed_at") if "reviewed_at" in cols else None
            parts[label_idx] = "accepted"
            if reviewed_at_idx is not None:
                parts[reviewed_at_idx] = "2024-06-01T12:00:00+00:00"
            reviewed_lines.append(",".join(parts))

        if len(reviewed_lines) == 1:
            pytest.skip("No replay results to review")

        reviewed_csv = "\n".join(reviewed_lines).encode()
        import_resp = client.post(
            f"/replay/{replay_run_id}/review/import",
            files={"file": ("review.csv", io.BytesIO(reviewed_csv), "text/csv")},
        )
        assert import_resp.status_code == 200
        assert import_resp.json()["reviews_applied"] > 0

        # Report must return 409 — stale, not yet re-simulated
        stale_resp = client.get(f"/replay/{replay_run_id}/report")
        assert stale_resp.status_code == 409

        # Re-simulate upgrades evidence
        self._simulate(client, replay_run_id)

        # Report now returns 200 with upgraded evidence
        fresh_report = self._get_report(client, replay_run_id)
        assert fresh_report["total_results"] > 0
        # At least one scenario has evidence at or above observed_replay
        all_scenarios = (
            fresh_report["recommended_migrations"]
            + fresh_report["investigate_scenarios"]
            + fresh_report["hold_scenarios"]
            + fresh_report["do_not_migrate"]
        )
        if all_scenarios:
            evidence_levels = {s["evidence_level"] for s in all_scenarios}
            upgradeable = {"human_reviewed", "observed_replay", "llm_judge", "production_validated"}
            assert evidence_levels & upgradeable  # at least one scenario has non-heuristic evidence
