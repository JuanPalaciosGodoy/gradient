from pathlib import Path

SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample_usage.csv"

VALID_CSV = b"""prompt,response,timestamp,model,cost
"Summarize this report","Summary here",2024-01-15 09:00:00,gpt-4o,0.0342
"Classify this ticket","Category: billing",2024-01-16 10:00:00,gpt-4o-mini,0.0012
"""

# All records use gpt-4o; used to trigger self-candidate rejection
GPT4O_ONLY_CSV = b"""prompt,response,timestamp,model,cost
"Summarize this report","Summary here",2024-01-15 09:00:00,gpt-4o,0.0342
"Classify this ticket","Category: billing",2024-01-16 10:00:00,gpt-4o,0.0012
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


# ── Replay run endpoint ───────────────────────────────────────────────────────

def _run_replay(client, audit_run_id: str, body: dict | None = None):
    return client.post(
        f"/audits/{audit_run_id}/replay/run",
        json=body or {},
    )


def test_replay_run_returns_200(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id)
    assert resp.status_code == 200


def test_replay_run_returns_run_id(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    data = _run_replay(client, run_id).json()
    assert "replay_run_id" in data
    assert len(data["replay_run_id"]) > 0


def test_replay_run_returns_records_selected(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    data = _run_replay(client, run_id).json()
    assert "records_selected" in data
    assert data["records_selected"] == 2  # VALID_CSV has 2 rows


def test_replay_run_returns_candidates_selected(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    data = _run_replay(client, run_id).json()
    assert "candidates_selected" in data
    assert isinstance(data["candidates_selected"], list)
    assert len(data["candidates_selected"]) > 0


def test_replay_run_status_is_complete(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    data = _run_replay(client, run_id).json()
    assert data["status"] == "complete"


def test_replay_run_not_found_returns_404(client):
    resp = _run_replay(client, "nonexistent-audit-id")
    assert resp.status_code == 404


def test_replay_run_with_max_records(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    data = _run_replay(client, run_id, {"max_records": 1}).json()
    assert data["records_selected"] == 1


def test_replay_run_with_candidate_models(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    data = _run_replay(client, run_id, {"candidate_models": ["gpt-4o-mini"]}).json()
    assert data["candidates_selected"] == ["gpt-4o-mini"]


# ── Simulate endpoint ─────────────────────────────────────────────────────────

def _upload_and_replay(client, content: bytes = VALID_CSV):
    run_id = _upload(client, content).json()["audit_run_id"]
    replay_id = _run_replay(client, run_id).json()["replay_run_id"]
    return run_id, replay_id


def _upload_replay_simulate(client, content: bytes = VALID_CSV):
    run_id, replay_id = _upload_and_replay(client, content)
    client.post(f"/replay/{replay_id}/simulate")
    return run_id, replay_id


def test_simulate_returns_200(client):
    _, replay_id = _upload_and_replay(client)
    resp = client.post(f"/replay/{replay_id}/simulate")
    assert resp.status_code == 200


def test_simulate_returns_simulations_count(client):
    _, replay_id = _upload_and_replay(client)
    data = client.post(f"/replay/{replay_id}/simulate").json()
    assert "simulations_count" in data
    assert isinstance(data["simulations_count"], int)
    assert data["simulations_count"] >= 0


def test_simulate_returns_top_scenarios(client):
    _, replay_id = _upload_and_replay(client)
    data = client.post(f"/replay/{replay_id}/simulate").json()
    assert "top_scenarios" in data
    assert isinstance(data["top_scenarios"], list)


def test_simulate_not_found_returns_404(client):
    resp = client.post("/replay/nonexistent-id/simulate")
    assert resp.status_code == 404


def test_simulate_replay_run_id_in_response(client):
    _, replay_id = _upload_and_replay(client)
    data = client.post(f"/replay/{replay_id}/simulate").json()
    assert data["replay_run_id"] == replay_id


def test_simulate_twice_no_duplicate_rows(client):
    _, replay_id = _upload_and_replay(client)
    client.post(f"/replay/{replay_id}/simulate")

    from app.database import get_migration_simulations_for_replay_run
    count_after_first = len(get_migration_simulations_for_replay_run(replay_id))

    client.post(f"/replay/{replay_id}/simulate")
    count_after_second = len(get_migration_simulations_for_replay_run(replay_id))

    assert count_after_first == count_after_second
    assert count_after_first > 0


# ── Replay report endpoints ───────────────────────────────────────────────────

def test_replay_report_returns_200(client):
    _, replay_id = _upload_replay_simulate(client)
    resp = client.get(f"/replay/{replay_id}/report")
    assert resp.status_code == 200


def test_replay_report_has_required_fields(client):
    _, replay_id = _upload_replay_simulate(client)
    data = client.get(f"/replay/{replay_id}/report").json()
    assert "replay_run_id" in data
    assert "executive_summary" in data
    assert "recommended_migrations" in data
    assert "do_not_migrate" in data
    assert "risk_notes" in data
    assert "next_actions" in data
    assert "current_annualized_spend" in data
    assert "estimated_annual_savings" in data
    assert "overall_confidence" in data


def test_replay_report_before_simulate_returns_422(client):
    _, replay_id = _upload_and_replay(client)
    resp = client.get(f"/replay/{replay_id}/report")
    assert resp.status_code == 422
    assert "simulate" in resp.json()["detail"].lower()


def test_replay_report_not_found_returns_404(client):
    resp = client.get("/replay/nonexistent-id/report")
    assert resp.status_code == 404


def test_replay_report_markdown_returns_200(client):
    _, replay_id = _upload_replay_simulate(client)
    resp = client.get(f"/replay/{replay_id}/report/markdown")
    assert resp.status_code == 200


def test_replay_report_markdown_is_text(client):
    _, replay_id = _upload_replay_simulate(client)
    resp = client.get(f"/replay/{replay_id}/report/markdown")
    assert "text/plain" in resp.headers["content-type"]


def test_replay_report_markdown_has_required_sections(client):
    _, replay_id = _upload_replay_simulate(client)
    md = client.get(f"/replay/{replay_id}/report/markdown").text
    assert "# Gradient Replay Analysis" in md
    assert "## Executive Summary" in md
    assert "## Recommended Migrations" in md
    assert "## Do Not Migrate" in md
    assert "## Risk Notes" in md
    assert "## Next Actions" in md


def test_replay_report_markdown_before_simulate_returns_422(client):
    _, replay_id = _upload_and_replay(client)
    resp = client.get(f"/replay/{replay_id}/report/markdown")
    assert resp.status_code == 422


def test_replay_report_markdown_not_found_returns_404(client):
    resp = client.get("/replay/nonexistent-id/report/markdown")
    assert resp.status_code == 404


# ── Self-migration prevention ─────────────────────────────────────────────────

def test_no_self_migration_in_simulation_scenarios(client):
    """Simulation scenarios must not have source_model == target_model."""
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    # Include both models from VALID_CSV as explicit candidates
    replay_id = _run_replay(
        client, run_id, {"candidate_models": ["gpt-4o", "gpt-4o-mini", "claude-3-haiku-20240307"]}
    ).json()["replay_run_id"]
    client.post(f"/replay/{replay_id}/simulate")
    report = client.get(f"/replay/{replay_id}/report").json()
    all_scenarios = report["recommended_migrations"] + report["do_not_migrate"]
    for s in all_scenarios:
        assert s["source_model"] != s["target_model"], (
            f"Self-migration scenario: {s['source_model']}"
        )


def test_no_self_migration_in_db_rows(client):
    """Every persisted simulation row — including hold/investigate — must have source != target."""
    from app.database import get_migration_simulations_for_replay_run
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    replay_id = _run_replay(
        client, run_id, {"candidate_models": ["gpt-4o", "gpt-4o-mini", "claude-3-haiku-20240307"]}
    ).json()["replay_run_id"]
    client.post(f"/replay/{replay_id}/simulate")
    rows = get_migration_simulations_for_replay_run(replay_id)
    assert len(rows) > 0, "Expected at least one simulation row persisted"
    for row in rows:
        assert row["source_model"] != row["target_model"], (
            f"Self-migration row in DB: source_model={row['source_model']}"
        )


# ── Task type classification in API flow ──────────────────────────────────────

def test_upload_classifies_task_type_for_records(client):
    from app.database import get_usage_records as _get_records
    csv = (
        b"prompt,response,timestamp,model,cost\n"
        b'"Summarize this quarterly report","Summary here",2024-01-15 09:00:00,gpt-4o,0.03\n'
        b'"Classify this support ticket","billing",2024-01-16 10:00:00,gpt-4o,0.002\n'
    )
    run_id = _upload(client, csv).json()["audit_run_id"]
    records = _get_records(run_id)
    task_types = {r["task_type"] for r in records}
    assert "summarization" in task_types
    assert "classification" in task_types


def test_replay_scenarios_reflect_classified_task_type(client):
    from app.database import get_usage_records as _get_records
    csv = (
        b"prompt,response,timestamp,model,cost\n"
        b'"Summarize this quarterly report into bullet points","Summary here",2024-01-15 09:00:00,gpt-4o,0.05\n'
    )
    run_id = _upload(client, csv).json()["audit_run_id"]
    records = _get_records(run_id)
    assert records[0]["task_type"] == "summarization"

    replay_id = _run_replay(client, run_id).json()["replay_run_id"]
    client.post(f"/replay/{replay_id}/simulate")
    report = client.get(f"/replay/{replay_id}/report").json()
    all_scenarios = report["recommended_migrations"] + report["do_not_migrate"]
    # All scenarios from this record should be tagged "summarization", not "other"
    if all_scenarios:
        for s in all_scenarios:
            assert s["task_type"] == "summarization"


# ── Full integration flow ─────────────────────────────────────────────────────

def test_full_flow_upload_replay_simulate_report(client):
    """End-to-end: upload → replay/run → simulate → report/markdown"""
    audit_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    replay_id = _run_replay(client, audit_id).json()["replay_run_id"]
    sim = client.post(f"/replay/{replay_id}/simulate").json()
    assert "simulations_count" in sim
    md = client.get(f"/replay/{replay_id}/report/markdown").text
    assert "# Gradient Replay Analysis" in md


def test_phase1_and_phase2_annualized_spend_match(client):
    """Phase 1 current_annual_spend and Phase 2 current_annualized_spend must agree
    when computed over the same records using the shared date-range helper."""
    audit_id = _upload(client, VALID_CSV).json()["audit_run_id"]

    # Phase 1 report
    _generate(client, audit_id)
    p1_spend = client.get(f"/audits/{audit_id}/report").json()["current_annual_spend"]

    # Phase 2: replay against the same audit, then simulate + report
    replay_id = _run_replay(client, audit_id).json()["replay_run_id"]
    client.post(f"/replay/{replay_id}/simulate")
    p2_spend = client.get(f"/replay/{replay_id}/report").json()["current_annualized_spend"]

    assert abs(p1_spend - p2_spend) < 0.01, (
        f"Phase 1 annualized={p1_spend:.4f} != Phase 2 annualized={p2_spend:.4f}"
    )


# ── Gap 1: all-self-candidate rejection ──────────────────────────────────────

def test_replay_run_all_self_candidates_returns_422(client):
    """All candidates match source model → no results → 422."""
    run_id = _upload(client, GPT4O_ONLY_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id, {"candidate_models": ["gpt-4o"]})
    assert resp.status_code == 422
    assert "candidate" in resp.json()["detail"].lower()


def test_replay_run_self_candidate_no_db_writes(client):
    """A rejected replay run must not persist any rows to the database."""
    from app.database import get_connection
    run_id = _upload(client, GPT4O_ONLY_CSV).json()["audit_run_id"]
    _run_replay(client, run_id, {"candidate_models": ["gpt-4o"]})
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM replay_runs WHERE audit_run_id = ?", (run_id,)
        ).fetchall()
    assert len(rows) == 0


def test_replay_run_mixed_candidates_succeeds_when_one_differs(client):
    """Even with one self-candidate, results from other candidates → 200."""
    run_id = _upload(client, GPT4O_ONLY_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id, {"candidate_models": ["gpt-4o", "gpt-4o-mini"]})
    assert resp.status_code == 200


# ── Gap 2 & 4: evaluator_mode validation ─────────────────────────────────────

def test_replay_run_invalid_evaluator_mode_returns_422(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id, {"evaluator_mode": "bad_mode"})
    assert resp.status_code == 422


def test_replay_run_prometheus_evaluator_mode_returns_422(client):
    """prometheus is a Phase 3 stub; API rejects it until configured."""
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id, {"evaluator_mode": "prometheus"})
    assert resp.status_code == 422


def test_replay_run_llm_judge_evaluator_mode_succeeds(client):
    """llm_judge uses FakeProvider in default mode and falls back to heuristic."""
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id, {"evaluator_mode": "llm_judge"})
    assert resp.status_code == 200


def test_replay_run_heuristic_evaluator_mode_accepted(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id, {"evaluator_mode": "heuristic"})
    assert resp.status_code == 200


# ── Gap 3: max_records validation ────────────────────────────────────────────

def test_replay_run_max_records_zero_returns_422(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id, {"max_records": 0})
    assert resp.status_code == 422


def test_replay_run_max_records_negative_returns_422(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id, {"max_records": -5})
    assert resp.status_code == 422


def test_replay_run_max_records_one_is_valid(client):
    run_id = _upload(client, VALID_CSV).json()["audit_run_id"]
    resp = _run_replay(client, run_id, {"max_records": 1})
    assert resp.status_code == 200
    assert resp.json()["records_selected"] == 1
