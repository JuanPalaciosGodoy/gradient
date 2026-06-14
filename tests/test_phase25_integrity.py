"""
Phase 2.5 integrity tests — five evidence and data-integrity fixes.

Fix 1: LLM judge fallback does not claim LLM_JUDGE evidence level.
Fix 2: Migration simulations inherit the strongest replay-result evidence.
Fix 3: Human review import is scoped to the correct replay run.
Fix 4: Report returns 409 when simulations are stale after human review.
Fix 5: ReplayResult rejects invalid cost/latency provenance field values.
"""
import csv
import io
import json
import uuid
from datetime import datetime, timezone

import pytest

# ════════════════════════════════════════════════════════════════════════════════
# Fix 1 — LLM judge fallback method and evidence level
# ════════════════════════════════════════════════════════════════════════════════

from app.evaluation.llm_judge_evaluator import LLMJudgeEvaluator
from app.providers.base import GenerationProvider, ProviderResponse
from app.providers.fake import FakeProvider
from app.schemas import EvidenceLevel, ValidationStatus


class _RaisingProvider(GenerationProvider):
    def generate(self, prompt, model):
        raise RuntimeError("API unavailable")


class _FixedProvider(GenerationProvider):
    def __init__(self, text):
        self._text = text

    def generate(self, prompt, model):
        return ProviderResponse(
            text=self._text, latency_ms=50.0,
            input_tokens=10, output_tokens=5, estimated_cost=0.0,
        )


def _good_json_payload():
    return json.dumps({
        "candidate_quality": 0.88,
        "quality_delta": 0.0,
        "acceptable_replacement": True,
        "risk_flags": [],
        "explanation": "High quality.",
        "confidence": 0.92,
    })


def test_valid_llm_judge_response_method_is_llm_judge():
    ev = LLMJudgeEvaluator(_FixedProvider(_good_json_payload()), "gpt-4o-mini")
    result = ev.evaluate("p", "orig", "cand")
    assert result.method == "llm_judge"
    assert "llm_judge_fallback" not in result.flags


def test_provider_error_fallback_method_is_llm_judge_fallback():
    ev = LLMJudgeEvaluator(_RaisingProvider(), "gpt-4o-mini")
    result = ev.evaluate("p", "orig", "cand")
    assert result.method == "llm_judge_fallback"
    assert "llm_judge_fallback" in result.flags


def test_invalid_json_fallback_method_is_llm_judge_fallback():
    ev = LLMJudgeEvaluator(_FixedProvider("I cannot evaluate this."), "gpt-4o-mini")
    result = ev.evaluate("p", "orig", "cand")
    assert result.method == "llm_judge_fallback"
    assert "llm_judge_fallback" in result.flags


def test_valid_judge_produces_llm_judge_evidence_in_runner():
    from app.replay.replay_models import ReplayCandidate, ReplayRequest
    from app.replay.replay_runner import ReplayRunner

    ev = LLMJudgeEvaluator(_FixedProvider(_good_json_payload()), "gpt-4o-mini")
    runner = ReplayRunner(provider=FakeProvider(), evaluator=ev)
    req = ReplayRequest(
        original_record_id="1", prompt="p", original_response="orig",
        original_model="gpt-4o", original_cost=0.01, task_type="summarization",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    candidate = ReplayCandidate(
        provider="openai", model="gpt-4o-mini", model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    results = runner.run([req], [candidate])
    assert results[0].evidence_level == EvidenceLevel.LLM_JUDGE


def test_provider_error_fallback_gets_observed_replay_evidence_not_llm_judge():
    from app.replay.replay_models import ReplayCandidate, ReplayRequest
    from app.replay.replay_runner import ReplayRunner

    ev = LLMJudgeEvaluator(_RaisingProvider(), "gpt-4o-mini")
    runner = ReplayRunner(provider=FakeProvider(), evaluator=ev)
    req = ReplayRequest(
        original_record_id="1", prompt="p", original_response="orig",
        original_model="gpt-4o", original_cost=0.01, task_type="summarization",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    candidate = ReplayCandidate(
        provider="openai", model="gpt-4o-mini", model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    results = runner.run([req], [candidate])
    assert results[0].evidence_level == EvidenceLevel.OBSERVED_REPLAY
    assert results[0].evidence_level != EvidenceLevel.LLM_JUDGE


def test_invalid_json_fallback_gets_observed_replay_not_llm_judge():
    from app.replay.replay_models import ReplayCandidate, ReplayRequest
    from app.replay.replay_runner import ReplayRunner

    ev = LLMJudgeEvaluator(_FixedProvider("no JSON here at all"), "gpt-4o-mini")
    runner = ReplayRunner(provider=FakeProvider(), evaluator=ev)
    req = ReplayRequest(
        original_record_id="1", prompt="p", original_response="orig",
        original_model="gpt-4o", original_cost=0.01, task_type="summarization",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    candidate = ReplayCandidate(
        provider="openai", model="gpt-4o-mini", model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
    )
    results = runner.run([req], [candidate])
    assert results[0].evidence_level == EvidenceLevel.OBSERVED_REPLAY


# ════════════════════════════════════════════════════════════════════════════════
# Fix 2 — Migration simulations inherit strongest replay-result evidence
# ════════════════════════════════════════════════════════════════════════════════

from app.replay.migration_simulator import simulate_from_replay_data, _derive_scenario_evidence


def _orig(record_id: int, model: str = "gpt-4o", cost: float = 0.10, task_type: str = "summarization") -> dict:
    return {"id": record_id, "model": model, "cost": cost, "task_type": task_type}


def _rr(orig_id: int, evidence_level: str = "observed_replay",
        validation_status: str = "evaluator_scored", quality: float = 0.85) -> dict:
    return {
        "original_record_id": str(orig_id),
        "candidate_model": "gpt-4o-mini",
        "quality_score": quality,
        "quality_confidence": 0.80,
        "estimated_cost": 0.005,
        "latency_ms": 200.0,
        "error_message": None,
        "evidence_level": evidence_level,
        "validation_status": validation_status,
    }


# _derive_scenario_evidence unit tests (coverage-aware policy: ≥80% threshold)

def test_scenario_evidence_all_observed_replay():
    rrs = [_rr(i, "observed_replay", "evaluator_scored") for i in range(3)]
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(rrs)
    assert level == EvidenceLevel.OBSERVED_REPLAY
    assert status == ValidationStatus.EVALUATOR_SCORED


def test_scenario_evidence_all_human_reviewed():
    # 100% HR → meets 80% threshold → HUMAN_REVIEWED
    rrs = [_rr(i, "human_reviewed", "human_reviewed") for i in range(3)]
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(rrs)
    assert level == EvidenceLevel.HUMAN_REVIEWED
    assert status == ValidationStatus.HUMAN_REVIEWED


def test_scenario_evidence_mixed_below_threshold_stays_observed():
    # 1 HR out of 3 = 33% — below 80% threshold → OBSERVED_REPLAY
    rrs = [
        _rr(1, "observed_replay", "evaluator_scored"),
        _rr(2, "human_reviewed", "human_reviewed"),
        _rr(3, "observed_replay", "evaluator_scored"),
    ]
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(rrs)
    assert level == EvidenceLevel.OBSERVED_REPLAY
    assert "human review coverage" in summary


def test_scenario_evidence_llm_judge_below_threshold_stays_observed():
    # 1 LLM out of 2 = 50% — below 80% threshold → OBSERVED_REPLAY
    rrs = [
        _rr(1, "observed_replay", "evaluator_scored"),
        _rr(2, "llm_judge", "evaluator_scored"),
    ]
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(rrs)
    assert level == EvidenceLevel.OBSERVED_REPLAY


def test_scenario_evidence_summary_includes_counts():
    # 2 HR + 1 OBS = 67% HR — below threshold → OBSERVED_REPLAY, but counts still in summary
    rrs = [
        _rr(1, "human_reviewed", "human_reviewed"),
        _rr(2, "observed_replay", "evaluator_scored"),
        _rr(3, "human_reviewed", "human_reviewed"),
    ]
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(rrs)
    assert level == EvidenceLevel.OBSERVED_REPLAY
    assert "2 human-reviewed" in summary
    assert "1 evaluator-scored" in summary


# simulate_from_replay_data integration tests

def test_simulation_with_all_human_reviewed_results():
    originals = [_orig(i) for i in range(5)]
    replays = [_rr(i, "human_reviewed", "human_reviewed") for i in range(5)]
    sims = simulate_from_replay_data(originals, replays, date_range_days=30)
    assert sims
    assert sims[0].evidence_level == EvidenceLevel.HUMAN_REVIEWED
    assert sims[0].validation_status == ValidationStatus.HUMAN_REVIEWED


def test_simulation_with_all_llm_judge_results():
    originals = [_orig(i) for i in range(4)]
    replays = [_rr(i, "llm_judge", "evaluator_scored") for i in range(4)]
    sims = simulate_from_replay_data(originals, replays, date_range_days=30)
    assert sims
    assert sims[0].evidence_level == EvidenceLevel.LLM_JUDGE


def test_simulation_mixed_observed_and_human_reviewed_below_threshold():
    # 2 HR out of 4 = 50% — below 80% threshold → OBSERVED_REPLAY (not HUMAN_REVIEWED)
    originals = [_orig(i) for i in range(4)]
    replays = [
        _rr(0, "observed_replay", "evaluator_scored"),
        _rr(1, "observed_replay", "evaluator_scored"),
        _rr(2, "human_reviewed", "human_reviewed"),
        _rr(3, "human_reviewed", "human_reviewed"),
    ]
    sims = simulate_from_replay_data(originals, replays, date_range_days=30)
    assert sims
    assert sims[0].evidence_level == EvidenceLevel.OBSERVED_REPLAY
    assert sims[0].evidence_counts.get("human_reviewed") == 2


def test_simulation_evidence_summary_mentions_counts():
    originals = [_orig(i) for i in range(3)]
    replays = [
        _rr(0, "human_reviewed", "human_reviewed"),
        _rr(1, "observed_replay", "evaluator_scored"),
        _rr(2, "observed_replay", "evaluator_scored"),
    ]
    sims = simulate_from_replay_data(originals, replays, date_range_days=30)
    assert sims
    summary = sims[0].evidence_summary
    assert "1 human-reviewed" in summary
    assert "2 evaluator-scored" in summary


def test_simulation_human_reviewed_confidence_higher_than_observed_replay():
    originals = [_orig(i) for i in range(5)]
    replays_hr = [_rr(i, "human_reviewed", "human_reviewed") for i in range(5)]
    replays_obs = [_rr(i, "observed_replay", "evaluator_scored") for i in range(5)]
    sims_hr = simulate_from_replay_data(originals, replays_hr, date_range_days=30)
    sims_obs = simulate_from_replay_data(originals, replays_obs, date_range_days=30)
    assert sims_hr and sims_obs
    # Human-reviewed evidence should yield higher adjusted confidence
    assert sims_hr[0].confidence_score > sims_obs[0].confidence_score


def test_replay_report_places_human_reviewed_in_correct_tier():
    from app.replay.replay_report import build_executive_replay_report_from_simulations
    from app.replay.replay_models import MigrationSimulationResult

    sim = MigrationSimulationResult(
        scenario_name="gpt-4o→gpt-4o-mini:summarization",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=50000.0,
        simulated_annualized_cost=40000.0,
        estimated_annual_savings=10000.0,
        estimated_savings_pct=20.0,
        average_quality_delta=-0.01,
        base_confidence_score=0.80,
        confidence_score=0.75,
        recommendation="migrate",
        rationale="Test.",
        records_analyzed=10,
        evidence_level=EvidenceLevel.HUMAN_REVIEWED,
        validation_status=ValidationStatus.HUMAN_REVIEWED,
    )
    report = build_executive_replay_report_from_simulations(
        replay_run_id="test",
        simulations=[sim],
        current_annualized_spend=50000.0,
        total_requests=10,
        total_results=10,
    )
    assert report.recommended_migrations[0].evidence_level == EvidenceLevel.HUMAN_REVIEWED
    assert "human" in report.executive_summary.lower()


# ════════════════════════════════════════════════════════════════════════════════
# Fix 3 — Human review import scoped to replay run
# ════════════════════════════════════════════════════════════════════════════════

VALID_CSV = b"""prompt,response,timestamp,model,cost
"Summarize this","Summary here",2024-01-15 09:00:00,gpt-4o,0.0342
"Classify this","Category: billing",2024-01-16 10:00:00,gpt-4o,0.0012
"""


def _upload_and_replay(client, csv_data=VALID_CSV):
    resp = client.post("/audits/upload", files={"file": ("d.csv", csv_data, "text/csv")})
    audit_id = resp.json()["audit_run_id"]
    client.post(f"/audits/{audit_id}/generate")
    resp = client.post(
        f"/audits/{audit_id}/replay/run",
        json={"candidate_models": ["gpt-4o-mini"], "max_records": 2},
    )
    return resp.json()["replay_run_id"]


def _export_csv(client, run_id):
    return client.get(f"/replay/{run_id}/review/export").text


def _make_import(export_text, label="candidate_better"):
    reader = csv.DictReader(io.StringIO(export_text))
    rows = list(reader)
    if not rows:
        return b""
    fields = (reader.fieldnames or []) + ["reviewer_label", "reviewer_notes"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for r in rows:
        r["reviewer_label"] = label
        r["reviewer_notes"] = ""
        w.writerow(r)
    return buf.getvalue().encode()


def test_cross_run_import_rejected(client):
    run_a = _upload_and_replay(client)
    run_b = _upload_and_replay(client)

    # Export from run_a, import into run_b — should reject since IDs don't match
    export_a = _export_csv(client, run_a)
    import_csv = _make_import(export_a, "candidate_better")

    resp = client.post(
        f"/replay/{run_b}/review/import",
        files={"file": ("review.csv", import_csv, "text/csv")},
    )
    assert resp.status_code == 422
    assert "do not belong to replay run" in resp.json()["detail"]


def test_unknown_replay_id_in_import_rejected(client):
    run_id = _upload_and_replay(client)
    fake_csv = (
        "replay_result_id,replay_run_id,original_record_id,candidate_model,"
        "task_type,original_response,candidate_response,quality_score,"
        "quality_method,evaluator_explanation,estimated_cost,cost_source,"
        "latency_ms,latency_source,reviewer_label,reviewer_notes\n"
        "nonexistent-uuid," + run_id + ",1,gpt-4o-mini,summarization,"
        "orig,cand,0.8,heuristic,ok,0.001,estimated_catalog,200,fake,"
        "candidate_better,\n"
    )
    resp = client.post(
        f"/replay/{run_id}/review/import",
        files={"file": ("review.csv", fake_csv.encode(), "text/csv")},
    )
    assert resp.status_code == 422
    assert "do not belong to replay run" in resp.json()["detail"]


def test_valid_import_for_correct_run_succeeds(client):
    run_id = _upload_and_replay(client)
    export_text = _export_csv(client, run_id)
    import_csv = _make_import(export_text, "candidate_equivalent")

    resp = client.post(
        f"/replay/{run_id}/review/import",
        files={"file": ("review.csv", import_csv, "text/csv")},
    )
    assert resp.status_code == 200
    assert resp.json()["reviews_applied"] > 0


def test_update_evidence_scoped_to_replay_run(db):
    """update_replay_result_evidence cannot update a row outside the provided replay_run_id."""
    from app.database import update_replay_result_evidence
    from app.replay.replay_models import ReplayResult
    from app.replay.replay_store import get_replay_results, save_replay_results, save_replay_run

    run_a = "scope-run-A"
    run_b = "scope-run-B"
    rr_id = str(uuid.uuid4())

    save_replay_run(run_a, None, ["gpt-4o-mini"], record_count=1)
    save_replay_run(run_b, None, ["gpt-4o-mini"], record_count=1)

    result = ReplayResult(
        replay_id=rr_id,
        original_record_id="1",
        candidate_provider="openai",
        candidate_model="gpt-4o-mini",
        candidate_response="ok",
        estimated_cost=0.001,
        latency_ms=100.0,
        quality_score=0.8,
        quality_method="heuristic",
        evidence_level=EvidenceLevel.OBSERVED_REPLAY,
        validation_status=ValidationStatus.EVALUATOR_SCORED,
        confidence_score=0.70,
    )
    save_replay_results(run_a, [result])

    # Try to update using run_b — should be a no-op (wrong run_id)
    update_replay_result_evidence(
        replay_id=rr_id,
        replay_run_id=run_b,  # wrong run
        evidence_level="human_reviewed",
        validation_status="human_reviewed",
        confidence_score=0.99,
    )

    rows = get_replay_results(run_a)
    assert rows[0]["evidence_level"] == "observed_replay"  # unchanged


# ════════════════════════════════════════════════════════════════════════════════
# Fix 4 — Stale simulation detection (Option C)
# ════════════════════════════════════════════════════════════════════════════════

def test_report_returns_409_when_reviews_newer_than_simulations(client):
    run_id = _upload_and_replay(client)

    # Run simulate
    client.post(f"/replay/{run_id}/simulate")

    # Now import human reviews (newer than simulate)
    export_text = _export_csv(client, run_id)
    import_csv = _make_import(export_text, "candidate_better")
    client.post(
        f"/replay/{run_id}/review/import",
        files={"file": ("review.csv", import_csv, "text/csv")},
    )

    # Report should now return 409
    resp = client.get(f"/replay/{run_id}/report")
    assert resp.status_code == 409
    assert "Re-run" in resp.json()["detail"]
    assert "simulate" in resp.json()["detail"]


def test_report_returns_409_for_markdown_when_stale(client):
    run_id = _upload_and_replay(client)
    client.post(f"/replay/{run_id}/simulate")
    export_text = _export_csv(client, run_id)
    import_csv = _make_import(export_text, "candidate_equivalent")
    client.post(
        f"/replay/{run_id}/review/import",
        files={"file": ("review.csv", import_csv, "text/csv")},
    )
    resp = client.get(f"/replay/{run_id}/report/markdown")
    assert resp.status_code == 409


def test_report_ok_after_resimulate_following_review(client):
    run_id = _upload_and_replay(client)
    client.post(f"/replay/{run_id}/simulate")

    export_text = _export_csv(client, run_id)
    import_csv = _make_import(export_text, "candidate_better")
    client.post(
        f"/replay/{run_id}/review/import",
        files={"file": ("review.csv", import_csv, "text/csv")},
    )

    # Re-run simulate to make it fresh
    client.post(f"/replay/{run_id}/simulate")

    resp = client.get(f"/replay/{run_id}/report")
    assert resp.status_code == 200


def test_report_without_any_reviews_is_not_stale(client):
    run_id = _upload_and_replay(client)
    client.post(f"/replay/{run_id}/simulate")
    resp = client.get(f"/replay/{run_id}/report")
    assert resp.status_code == 200


def test_report_reflects_human_reviewed_evidence_after_resimulate(client):
    """After review + re-simulate, the simulation response reflects HUMAN_REVIEWED evidence."""
    run_id = _upload_and_replay(client)
    client.post(f"/replay/{run_id}/simulate")
    export_text = _export_csv(client, run_id)
    import_csv = _make_import(export_text, "candidate_better")
    client.post(
        f"/replay/{run_id}/review/import",
        files={"file": ("review.csv", import_csv, "text/csv")},
    )

    # Re-run simulation to pick up human review evidence from replay_results
    sim_resp = client.post(f"/replay/{run_id}/simulate")
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    # top_scenarios are the top-N by savings regardless of recommendation type
    top_levels = [s["evidence_level"] for s in sim_data.get("top_scenarios", [])]
    assert "human_reviewed" in top_levels


# ════════════════════════════════════════════════════════════════════════════════
# Fix 5 — ReplayResult validators for cost/latency provenance fields
# ════════════════════════════════════════════════════════════════════════════════

from app.replay.replay_models import ReplayResult


def _base_result(**overrides) -> dict:
    defaults = dict(
        replay_id=str(uuid.uuid4()),
        original_record_id="1",
        candidate_provider="openai",
        candidate_model="gpt-4o-mini",
        candidate_response="ok",
        estimated_cost=0.001,
        latency_ms=100.0,
        quality_score=0.85,
        quality_method="heuristic",
        input_tokens=10,
        output_tokens=5,
        cost_source="estimated_catalog",
        latency_source="fake",
    )
    return {**defaults, **overrides}


def test_valid_cost_source_values():
    for source in ("observed", "estimated_catalog", "fake", "missing"):
        r = ReplayResult(**_base_result(cost_source=source))
        assert r.cost_source == source


def test_invalid_cost_source_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="cost_source"):
        ReplayResult(**_base_result(cost_source="unknown_source"))


def test_valid_latency_source_values():
    for source in ("observed", "fake", "missing"):
        r = ReplayResult(**_base_result(latency_source=source))
        assert r.latency_source == source


def test_invalid_latency_source_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="latency_source"):
        ReplayResult(**_base_result(latency_source="estimated"))


def test_negative_input_tokens_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="input_tokens"):
        ReplayResult(**_base_result(input_tokens=-1))


def test_negative_output_tokens_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="output_tokens"):
        ReplayResult(**_base_result(output_tokens=-5))


def test_zero_tokens_is_valid():
    r = ReplayResult(**_base_result(input_tokens=0, output_tokens=0))
    assert r.input_tokens == 0
    assert r.output_tokens == 0


def test_cost_source_estimated_catalog_is_default():
    r = ReplayResult(**_base_result())
    assert r.cost_source == "estimated_catalog"


def test_latency_source_fake_is_default():
    r = ReplayResult(**_base_result())
    assert r.latency_source == "fake"


def test_missing_source_valid_for_error_paths():
    r = ReplayResult(**_base_result(cost_source="missing", latency_source="missing"))
    assert r.cost_source == "missing"
    assert r.latency_source == "missing"
