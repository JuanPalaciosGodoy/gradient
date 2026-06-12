from pathlib import Path

SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample_usage.csv"

VALID_CSV = b"""prompt,response,timestamp,model,cost
"Summarize this report","Summary here",2024-01-15 09:00:00,gpt-4o,0.0342
"Classify this ticket","Category: billing",2024-01-16 10:00:00,gpt-4o-mini,0.0012
"""


def _upload(client, content: bytes, filename: str = "data.csv"):
    return client.post(
        "/audits/upload",
        files={"file": (filename, content, "text/csv")},
    )


def _generate(client, run_id: str):
    return client.post(f"/audits/{run_id}/generate")


def _upload_and_generate(client, content: bytes = VALID_CSV):
    run_id = _upload(client, content).json()["audit_run_id"]
    _generate(client, run_id)
    return run_id


# ── Health check ─────────────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Upload ────────────────────────────────────────────────────────────────────

def test_upload_sample_csv_returns_200(client):
    with open(SAMPLE_CSV, "rb") as f:
        resp = _upload(client, f.read(), "sample_usage.csv")
    assert resp.status_code == 200


def test_upload_returns_audit_run_id(client):
    with open(SAMPLE_CSV, "rb") as f:
        resp = _upload(client, f.read(), "sample_usage.csv")
    data = resp.json()
    assert "audit_run_id" in data
    assert isinstance(data["audit_run_id"], str)
    assert len(data["audit_run_id"]) > 0


def test_upload_returns_record_count(client):
    with open(SAMPLE_CSV, "rb") as f:
        resp = _upload(client, f.read(), "sample_usage.csv")
    data = resp.json()
    assert "record_count" in data
    assert data["record_count"] == 41


def test_upload_valid_csv_status_is_complete(client):
    resp = _upload(client, VALID_CSV)
    assert resp.json()["status"] == "complete"


def test_upload_two_calls_produce_distinct_run_ids(client):
    r1 = _upload(client, VALID_CSV).json()["audit_run_id"]
    r2 = _upload(client, VALID_CSV).json()["audit_run_id"]
    assert r1 != r2


def test_upload_returns_validation_summary(client):
    resp = _upload(client, VALID_CSV)
    data = resp.json()
    vs = data["validation_summary"]
    assert vs["total_rows"] == 2
    assert vs["valid_rows"] == 2
    assert vs["invalid_rows"] == 0
    assert isinstance(vs["error_samples"], list)


# ── Client errors ─────────────────────────────────────────────────────────────

def test_upload_missing_columns_returns_422(client):
    csv = b"prompt,response,timestamp\nhello,world,2024-01-01\n"
    resp = _upload(client, csv)
    assert resp.status_code == 422
    assert "Missing required columns" in resp.json()["detail"]


def test_upload_invalid_cost_returns_422(client):
    csv = b"prompt,response,timestamp,model,cost\nhello,world,2024-01-15 09:00:00,gpt-4o,not_a_number\n"
    resp = _upload(client, csv)
    assert resp.status_code == 422


def test_upload_invalid_timestamp_returns_422(client):
    csv = b"prompt,response,timestamp,model,cost\nhello,world,not-a-date,gpt-4o,0.01\n"
    resp = _upload(client, csv)
    assert resp.status_code == 422


def test_upload_non_csv_extension_returns_400(client):
    resp = _upload(client, b'{"key": "val"}', filename="data.json")
    assert resp.status_code == 400


# ── Generate ──────────────────────────────────────────────────────────────────

def test_generate_returns_200(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = _generate(client, run_id)
    assert resp.status_code == 200


def test_generate_returns_report_id(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    data = _generate(client, run_id).json()
    assert "report_id" in data
    assert isinstance(data["report_id"], str)
    assert len(data["report_id"]) > 0


def test_generate_returns_audit_run_id(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    data = _generate(client, run_id).json()
    assert data["audit_run_id"] == run_id


def test_generate_status_is_complete(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    data = _generate(client, run_id).json()
    assert data["status"] == "complete"


def test_generate_not_found_returns_404(client):
    resp = _generate(client, "nonexistent-id")
    assert resp.status_code == 404


def test_generate_twice_produces_new_report_id(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    id1 = _generate(client, run_id).json()["report_id"]
    id2 = _generate(client, run_id).json()["report_id"]
    assert id1 != id2


# ── Report JSON ───────────────────────────────────────────────────────────────

def test_report_not_found_before_generate(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = client.get(f"/audits/{run_id}/report")
    assert resp.status_code == 404


def test_report_endpoint_returns_200(client):
    run_id = _upload_and_generate(client)
    resp = client.get(f"/audits/{run_id}/report")
    assert resp.status_code == 200


def test_report_has_report_id(client):
    run_id = _upload_and_generate(client)
    report = client.get(f"/audits/{run_id}/report").json()
    assert "report_id" in report
    assert isinstance(report["report_id"], str)


def test_report_has_savings_fields(client):
    run_id = _upload_and_generate(client)
    report = client.get(f"/audits/{run_id}/report").json()
    assert "potential_annual_savings" in report
    assert "savings_rate" in report
    assert report["potential_annual_savings"] >= 0
    assert 0.0 <= report["savings_rate"] <= 1.0


def test_report_has_executive_summary(client):
    run_id = _upload_and_generate(client)
    report = client.get(f"/audits/{run_id}/report").json()
    assert "executive_summary" in report
    assert len(report["executive_summary"]) > 50


def test_report_has_current_annual_spend(client):
    run_id = _upload_and_generate(client)
    report = client.get(f"/audits/{run_id}/report").json()
    assert "current_annual_spend" in report
    assert report["current_annual_spend"] == report["spend_summary"]["annualized_cost"]


def test_report_has_recommended_next_actions(client):
    run_id = _upload_and_generate(client)
    report = client.get(f"/audits/{run_id}/report").json()
    assert "recommended_next_actions" in report
    assert isinstance(report["recommended_next_actions"], list)


def test_report_not_found_returns_404(client):
    resp = client.get("/audits/nonexistent-id/report")
    assert resp.status_code == 404


def test_report_has_top_opportunities(client):
    run_id = _upload_and_generate(client)
    report = client.get(f"/audits/{run_id}/report").json()
    assert "top_opportunities" in report
    assert isinstance(report["top_opportunities"], list)


def test_report_has_recommended_next_actions_list(client):
    run_id = _upload_and_generate(client)
    report = client.get(f"/audits/{run_id}/report").json()
    assert isinstance(report["recommended_next_actions"], list)
    assert len(report["recommended_next_actions"]) > 0


def test_spend_summary_has_top_cost_driving_task_types(client):
    run_id = _upload_and_generate(client)
    report = client.get(f"/audits/{run_id}/report").json()
    summary = report["spend_summary"]
    assert "top_cost_driving_task_types" in summary
    assert isinstance(summary["top_cost_driving_task_types"], list)


def test_top_cost_driving_task_types_item_shape(client):
    run_id = _upload_and_generate(client)
    report = client.get(f"/audits/{run_id}/report").json()
    items = report["spend_summary"]["top_cost_driving_task_types"]
    if items:
        for item in items:
            assert "task_type" in item
            assert "cost" in item
            assert isinstance(item["task_type"], str)
            assert isinstance(item["cost"], float)


# ── Markdown report ───────────────────────────────────────────────────────────

def test_markdown_not_found_before_generate(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = client.get(f"/audits/{run_id}/report/markdown")
    assert resp.status_code == 404


def test_markdown_report_returns_200(client):
    run_id = _upload_and_generate(client)
    resp = client.get(f"/audits/{run_id}/report/markdown")
    assert resp.status_code == 200


def test_markdown_report_is_text(client):
    run_id = _upload_and_generate(client)
    resp = client.get(f"/audits/{run_id}/report/markdown")
    assert "text/plain" in resp.headers["content-type"]


def test_markdown_report_has_required_sections(client):
    run_id = _upload_and_generate(client)
    md = client.get(f"/audits/{run_id}/report/markdown").text
    assert "# Gradient AI Procurement Audit" in md
    assert "## Executive Summary" in md
    assert "## Top Opportunities" in md
    assert "## Spend Concentration" in md
    assert "## Risk Notes" in md
    assert "## Recommended Next Actions" in md


def test_markdown_report_contains_spend_figures(client):
    run_id = _upload_and_generate(client)
    md = client.get(f"/audits/{run_id}/report/markdown").text
    assert "$" in md
    assert "Annual" in md


def test_markdown_report_not_found_for_unknown_id(client):
    resp = client.get("/audits/nonexistent-id/report/markdown")
    assert resp.status_code == 404


# ── 404 detail messages ───────────────────────────────────────────────────────

def test_report_unknown_run_id_says_audit_run_not_found(client):
    resp = client.get("/audits/nonexistent-id/report")
    assert resp.status_code == 404
    assert "Audit run not found" in resp.json()["detail"]


def test_report_before_generate_says_call_generate(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = client.get(f"/audits/{run_id}/report")
    assert resp.status_code == 404
    assert "generate" in resp.json()["detail"].lower()


def test_markdown_unknown_run_id_says_audit_run_not_found(client):
    resp = client.get("/audits/nonexistent-id/report/markdown")
    assert resp.status_code == 404
    assert "Audit run not found" in resp.json()["detail"]


def test_markdown_before_generate_says_call_generate(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = client.get(f"/audits/{run_id}/report/markdown")
    assert resp.status_code == 404
    assert "generate" in resp.json()["detail"].lower()


# ── Markdown content ──────────────────────────────────────────────────────────

def test_markdown_has_spend_labels(client):
    run_id = _upload_and_generate(client)
    md = client.get(f"/audits/{run_id}/report/markdown").text
    assert "Current Annual Spend" in md
    assert "Potential Annual Savings" in md
    assert "Savings Rate" in md


def test_markdown_has_numbered_next_action(client):
    run_id = _upload_and_generate(client)
    md = client.get(f"/audits/{run_id}/report/markdown").text
    assert "1." in md


def test_markdown_has_replay_validation_language(client):
    run_id = _upload_and_generate(client)
    md = client.get(f"/audits/{run_id}/report/markdown").text
    assert "replay" in md.lower()
