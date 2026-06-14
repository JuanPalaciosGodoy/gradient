"""
Phase 2.5 evidence model tests.

Covers:
- EvidenceLevel / ValidationStatus on all recommendation types
- confidence_score (evidence-adjusted) on Opportunity, ReplayResult, ScenarioOutcome,
  SimulationTopRecommendation, MigrationSimulationResult
- base_confidence_score (raw) on MigrationSimulationResult
- adjust_confidence_for_evidence ordering
- Failed replay uses HEURISTIC (not OBSERVED_REPLAY)
- All-failed simulation uses HEURISTIC (not OBSERVED_REPLAY)
- ScenarioOutcome.limitations uses Field(default_factory=list)
- Conservative Pydantic defaults (no model defaults to OBSERVED_REPLAY/EVALUATOR_SCORED)
- Markdown reports separate evidence tiers into named sections, including LLM_JUDGE and
  PRODUCTION_VALIDATED
- Legacy DB rows get conservative defaults
- Field validators reject out-of-range confidence values
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.audit.opportunity_analyzer import find_opportunities
from app.replay.migration_simulator import simulate_from_replay_data, simulate_migration
from app.replay.replay_models import (
    MigrationScenario,
    MigrationSimulationResult,
    REPLAY_CANDIDATES,
    ReplayResult,
)
from app.replay.replay_report import ScenarioOutcome, _sim_to_outcome
from app.replay.replay_runner import ReplayRunner, build_replay_requests
from app.providers.fake import FakeProvider
from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.reports.replay_markdown import render_replay_markdown_report
from app.schemas import (
    EvidenceLevel,
    SimulationTopRecommendation,
    TaskType,
    UsageRecord,
    ValidationStatus,
)
from app.utils.evidence import adjust_confidence_for_evidence


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_record(
    model: str = "gpt-4o",
    task_type: str = "classification",
    cost: float = 0.05,
    days_ago: int = 0,
) -> UsageRecord:
    return UsageRecord(
        prompt="Classify this customer ticket",
        response="Category: billing",
        timestamp=datetime(2024, 3, 31, tzinfo=timezone.utc) - timedelta(days=days_ago),
        model=model,
        cost=cost,
        task_type=TaskType(task_type),
    )


def _make_usage_row(
    record_id: int,
    model: str = "gpt-4o",
    task_type: str = "classification",
    cost: float = 0.05,
) -> dict:
    return {
        "id": record_id,
        "model": model,
        "task_type": task_type,
        "cost": cost,
        "prompt": "Classify this customer ticket",
        "response": "Category: billing",
        "timestamp": "2024-03-31T00:00:00+00:00",
    }


def _make_replay_result(
    original_record_id: int,
    candidate_model: str = "gpt-4o-mini",
    quality_score: float = 0.92,
    quality_confidence: float = 0.85,
    error_message: str | None = None,
) -> dict:
    return {
        "replay_id": str(uuid.uuid4()),
        "original_record_id": original_record_id,
        "candidate_model": candidate_model,
        "candidate_provider": "openai",
        "candidate_response": "Category: billing",
        "estimated_cost": 0.002,
        "latency_ms": 250.0,
        "quality_score": quality_score,
        "quality_method": "heuristic",
        "quality_explanation": "Good classification output.",
        "quality_confidence": quality_confidence,
        "quality_flags": [],
        "error_message": error_message,
    }


def _make_sim(
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVED_REPLAY,
    validation_status: ValidationStatus = ValidationStatus.EVALUATOR_SCORED,
    recommendation: str = "migrate",
    base_confidence_score: float = 0.80,
) -> MigrationSimulationResult:
    adj = adjust_confidence_for_evidence(base_confidence_score, evidence_level, validation_status)
    return MigrationSimulationResult(
        scenario_name="gpt-4o→gpt-4o-mini:classification",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=1000.0,
        simulated_annualized_cost=300.0,
        estimated_annual_savings=700.0,
        estimated_savings_pct=70.0,
        average_quality_delta=-0.01,
        base_confidence_score=base_confidence_score,
        confidence_score=adj,
        recommendation=recommendation,
        rationale="Low quality loss, high savings.",
        records_analyzed=50,
        evidence_level=evidence_level,
        evidence_summary="Test summary.",
        limitations=["Test limitation."],
        validation_status=validation_status,
    )


def _executive_report(scenarios: list[ScenarioOutcome]):
    from app.replay.replay_report import ExecutiveReplayReport
    return ExecutiveReplayReport(
        replay_run_id="test-run",
        generated_at=datetime.now(timezone.utc),
        total_requests=10,
        total_results=60,
        current_annualized_spend=1000.0,
        simulated_annualized_spend=300.0,
        estimated_annual_savings=700.0,
        avg_quality_impact=-0.01,
        overall_confidence=80,
        executive_summary="Test summary.",
        recommended_migrations=scenarios,
        do_not_migrate=[],
        risk_notes=[],
        next_actions=["Act now."],
    )


# ── adjust_confidence_for_evidence ordering ───────────────────────────────────

def test_adjust_confidence_heuristic_is_lower_than_estimated():
    h = adjust_confidence_for_evidence(0.90, EvidenceLevel.HEURISTIC, ValidationStatus.NOT_VALIDATED)
    e = adjust_confidence_for_evidence(0.90, EvidenceLevel.ESTIMATED, ValidationStatus.NOT_VALIDATED)
    assert h < e


def test_adjust_confidence_estimated_is_lower_than_observed_replay():
    e = adjust_confidence_for_evidence(0.90, EvidenceLevel.ESTIMATED, ValidationStatus.NOT_VALIDATED)
    o = adjust_confidence_for_evidence(0.90, EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED)
    assert e < o


def test_adjust_confidence_observed_replay_is_lower_than_human_reviewed():
    o = adjust_confidence_for_evidence(0.90, EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED)
    h = adjust_confidence_for_evidence(0.90, EvidenceLevel.HUMAN_REVIEWED, ValidationStatus.HUMAN_REVIEWED)
    assert o < h


def test_adjust_confidence_human_reviewed_is_lower_than_production_validated():
    h = adjust_confidence_for_evidence(0.90, EvidenceLevel.HUMAN_REVIEWED, ValidationStatus.HUMAN_REVIEWED)
    p = adjust_confidence_for_evidence(0.90, EvidenceLevel.PRODUCTION_VALIDATED, ValidationStatus.PRODUCTION_VALIDATED)
    assert h < p


def test_adjust_confidence_result_bounded_at_zero_and_095():
    assert adjust_confidence_for_evidence(0.0, EvidenceLevel.HEURISTIC, ValidationStatus.NOT_VALIDATED) == 0.0
    assert adjust_confidence_for_evidence(1.5, EvidenceLevel.PRODUCTION_VALIDATED, ValidationStatus.PRODUCTION_VALIDATED) == 0.95


def test_adjust_confidence_same_base_different_levels_produce_different_scores():
    scores = [
        adjust_confidence_for_evidence(0.80, EvidenceLevel.HEURISTIC, ValidationStatus.NOT_VALIDATED),
        adjust_confidence_for_evidence(0.80, EvidenceLevel.ESTIMATED, ValidationStatus.NOT_VALIDATED),
        adjust_confidence_for_evidence(0.80, EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED),
        adjust_confidence_for_evidence(0.80, EvidenceLevel.HUMAN_REVIEWED, ValidationStatus.HUMAN_REVIEWED),
        adjust_confidence_for_evidence(0.80, EvidenceLevel.PRODUCTION_VALIDATED, ValidationStatus.PRODUCTION_VALIDATED),
    ]
    assert scores == sorted(scores), "Scores must be strictly ordered by evidence tier"


# ── Conservative Pydantic defaults ───────────────────────────────────────────

def test_replay_result_default_evidence_is_heuristic():
    r = ReplayResult(
        replay_id=str(uuid.uuid4()),
        original_record_id="1",
        candidate_provider="openai",
        candidate_model="gpt-4o-mini",
        candidate_response="ok",
        estimated_cost=0.001,
        latency_ms=100.0,
        quality_score=0.9,
        quality_method="heuristic",
    )
    assert r.evidence_level == EvidenceLevel.HEURISTIC
    assert r.validation_status == ValidationStatus.NOT_VALIDATED
    assert r.confidence_score == 0.0


def test_scenario_outcome_default_evidence_is_heuristic():
    o = ScenarioOutcome(
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_type="classification",
        estimated_annual_savings=500.0,
        quality_loss_pct=1.0,
        avg_quality_delta=-0.01,
        avg_latency_ms=200.0,
        confidence_pct=70,
        recommendation="migrate",
        rationale="test",
        records_analyzed=10,
    )
    assert o.evidence_level == EvidenceLevel.HEURISTIC
    assert o.validation_status == ValidationStatus.NOT_VALIDATED
    assert o.confidence_score == 0.0


def test_simulation_top_recommendation_default_evidence_is_heuristic():
    r = SimulationTopRecommendation(
        scenario_name="gpt-4o→gpt-4o-mini:classification",
        recommendation="migrate",
        estimated_annual_savings=500.0,
        confidence_pct=70,
    )
    assert r.evidence_level == EvidenceLevel.HEURISTIC
    assert r.validation_status == ValidationStatus.NOT_VALIDATED
    assert r.confidence_score == 0.0


# ── Field validators — confidence_score ──────────────────────────────────────

def test_replay_result_confidence_score_rejects_negative():
    with pytest.raises(Exception):
        ReplayResult(
            replay_id=str(uuid.uuid4()),
            original_record_id="1",
            candidate_provider="openai",
            candidate_model="gpt-4o-mini",
            candidate_response="ok",
            estimated_cost=0.001,
            latency_ms=100.0,
            quality_score=0.9,
            quality_method="heuristic",
            confidence_score=-0.01,
        )


def test_replay_result_confidence_score_rejects_above_one():
    with pytest.raises(Exception):
        ReplayResult(
            replay_id=str(uuid.uuid4()),
            original_record_id="1",
            candidate_provider="openai",
            candidate_model="gpt-4o-mini",
            candidate_response="ok",
            estimated_cost=0.001,
            latency_ms=100.0,
            quality_score=0.9,
            quality_method="heuristic",
            confidence_score=1.01,
        )


def test_replay_result_confidence_score_boundary_values_accepted():
    for v in (0.0, 0.5, 1.0):
        r = ReplayResult(
            replay_id=str(uuid.uuid4()),
            original_record_id="1",
            candidate_provider="openai",
            candidate_model="gpt-4o-mini",
            candidate_response="ok",
            estimated_cost=0.001,
            latency_ms=100.0,
            quality_score=0.9,
            quality_method="heuristic",
            confidence_score=v,
        )
        assert r.confidence_score == v


def test_scenario_outcome_confidence_pct_rejects_above_100():
    with pytest.raises(Exception):
        ScenarioOutcome(
            source_model="gpt-4o",
            target_model="gpt-4o-mini",
            task_type="classification",
            estimated_annual_savings=500.0,
            quality_loss_pct=1.0,
            avg_quality_delta=-0.01,
            avg_latency_ms=200.0,
            confidence_pct=101,
            recommendation="migrate",
            rationale="test",
            records_analyzed=10,
        )


def test_scenario_outcome_confidence_pct_rejects_negative():
    with pytest.raises(Exception):
        ScenarioOutcome(
            source_model="gpt-4o",
            target_model="gpt-4o-mini",
            task_type="classification",
            estimated_annual_savings=500.0,
            quality_loss_pct=1.0,
            avg_quality_delta=-0.01,
            avg_latency_ms=200.0,
            confidence_pct=-1,
            recommendation="migrate",
            rationale="test",
            records_analyzed=10,
        )


def test_simulation_top_rec_confidence_pct_rejects_above_100():
    with pytest.raises(Exception):
        SimulationTopRecommendation(
            scenario_name="test",
            recommendation="migrate",
            estimated_annual_savings=100.0,
            confidence_pct=101,
        )


def test_migration_sim_result_base_confidence_score_rejects_above_one():
    with pytest.raises(Exception):
        MigrationSimulationResult(
            scenario_name="test",
            source_model="gpt-4o",
            target_model="gpt-4o-mini",
            current_annualized_cost=1000.0,
            simulated_annualized_cost=300.0,
            estimated_annual_savings=700.0,
            estimated_savings_pct=70.0,
            average_quality_delta=-0.01,
            base_confidence_score=1.5,
            confidence_score=0.0,
            recommendation="migrate",
            rationale="test",
        )


def test_migration_sim_result_confidence_score_rejects_negative():
    with pytest.raises(Exception):
        MigrationSimulationResult(
            scenario_name="test",
            source_model="gpt-4o",
            target_model="gpt-4o-mini",
            current_annualized_cost=1000.0,
            simulated_annualized_cost=300.0,
            estimated_annual_savings=700.0,
            estimated_savings_pct=70.0,
            average_quality_delta=-0.01,
            base_confidence_score=0.8,
            confidence_score=-0.01,
            recommendation="migrate",
            rationale="test",
        )


# ── Opportunity evidence + confidence_score ───────────────────────────────────

def test_opportunity_evidence_level_is_heuristic():
    records = [_make_record() for _ in range(5)]
    opps = find_opportunities(records, date_range_days=30)
    assert opps
    for opp in opps:
        assert opp.evidence_level == EvidenceLevel.HEURISTIC


def test_opportunity_validation_status_is_not_validated():
    records = [_make_record() for _ in range(5)]
    opps = find_opportunities(records, date_range_days=30)
    assert opps
    for opp in opps:
        assert opp.validation_status == ValidationStatus.NOT_VALIDATED


def test_opportunity_has_confidence_score():
    records = [_make_record() for _ in range(5)]
    opps = find_opportunities(records, date_range_days=30)
    assert opps
    for opp in opps:
        assert isinstance(opp.confidence_score, float)
        assert 0.0 <= opp.confidence_score <= 0.95


def test_opportunity_confidence_score_lower_than_raw_confidence():
    """Evidence adjustment must discount raw confidence for HEURISTIC/NOT_VALIDATED."""
    records = [_make_record() for _ in range(5)]
    opps = find_opportunities(records, date_range_days=30)
    assert opps
    for opp in opps:
        assert opp.confidence_score < opp.confidence


def test_opportunity_evidence_summary_mentions_heuristic():
    records = [_make_record() for _ in range(5)]
    opps = find_opportunities(records, date_range_days=30)
    assert opps
    for opp in opps:
        assert "heuristic" in opp.evidence_summary.lower()


# ── Replay result evidence + confidence_score ─────────────────────────────────

def test_replay_result_success_has_observed_replay_evidence():
    records = [_make_record(model="gpt-4o")]
    requests = build_replay_requests(records)
    candidates = [c for c in REPLAY_CANDIDATES if c.model == "gpt-4o-mini"]
    runner = ReplayRunner(provider=FakeProvider(), evaluator=HeuristicEvaluator())
    results = runner.run(requests, candidates)
    assert results
    for r in results:
        if not r.error_message:
            assert r.evidence_level == EvidenceLevel.OBSERVED_REPLAY


def test_replay_result_success_has_confidence_score():
    records = [_make_record(model="gpt-4o")]
    requests = build_replay_requests(records)
    candidates = [c for c in REPLAY_CANDIDATES if c.model == "gpt-4o-mini"]
    runner = ReplayRunner(provider=FakeProvider(), evaluator=HeuristicEvaluator())
    results = runner.run(requests, candidates)
    successful = [r for r in results if not r.error_message]
    assert successful
    for r in successful:
        assert isinstance(r.confidence_score, float)
        assert 0.0 <= r.confidence_score <= 0.95


def test_replay_result_failed_uses_heuristic_not_observed_replay():
    """A failed replay must not claim OBSERVED_REPLAY evidence."""
    from app.providers.base import GenerationProvider

    class FailingProvider(GenerationProvider):
        def generate(self, prompt, model):
            raise RuntimeError("network timeout")

    records = [_make_record(model="gpt-4o")]
    requests = build_replay_requests(records)
    candidates = [c for c in REPLAY_CANDIDATES if c.model == "gpt-4o-mini"]
    runner = ReplayRunner(provider=FailingProvider(), evaluator=HeuristicEvaluator())
    results = runner.run(requests, candidates)
    failed = [r for r in results if r.error_message]
    assert failed
    for r in failed:
        assert r.evidence_level == EvidenceLevel.HEURISTIC
        assert r.validation_status == ValidationStatus.NOT_VALIDATED
        assert r.confidence_score == 0.0


# ── Phase 1 (ESTIMATED) simulation evidence ───────────────────────────────────

def test_phase1_simulation_evidence_level_is_estimated():
    records = [_make_record() for _ in range(10)]
    scenario = MigrationScenario(
        scenario_name="gpt4o-to-mini:classification",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=["classification"],
        migration_percentage=1.0,
    )
    result = simulate_migration(records, scenario, date_range_days=30)
    assert result.evidence_level == EvidenceLevel.ESTIMATED


def test_phase1_simulation_confidence_score_is_evidence_adjusted():
    """confidence_score must be < base_confidence_score for ESTIMATED/NOT_VALIDATED."""
    records = [_make_record() for _ in range(10)]
    scenario = MigrationScenario(
        scenario_name="gpt4o-to-mini:classification",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=["classification"],
        migration_percentage=1.0,
    )
    result = simulate_migration(records, scenario, date_range_days=30)
    assert 0.0 <= result.confidence_score <= 0.95
    assert result.confidence_score < result.base_confidence_score


def test_phase1_simulation_confidence_score_matches_adjustment():
    """confidence_score == adjust_confidence_for_evidence(base, ESTIMATED, NOT_VALIDATED)."""
    records = [_make_record() for _ in range(10)]
    scenario = MigrationScenario(
        scenario_name="gpt4o-to-mini:classification",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=["classification"],
        migration_percentage=1.0,
    )
    result = simulate_migration(records, scenario, date_range_days=30)
    expected = adjust_confidence_for_evidence(
        result.base_confidence_score, EvidenceLevel.ESTIMATED, ValidationStatus.NOT_VALIDATED
    )
    assert result.confidence_score == expected


# ── Phase 2 (OBSERVED_REPLAY) simulation evidence ─────────────────────────────

def test_phase2_simulation_evidence_level_is_observed_replay():
    rows = [_make_usage_row(i) for i in range(1, 11)]
    replay_results = [_make_replay_result(i) for i in range(1, 11)]
    sims = simulate_from_replay_data(rows, replay_results, date_range_days=30)
    assert sims
    for s in sims:
        if s.records_analyzed > 0:
            assert s.evidence_level == EvidenceLevel.OBSERVED_REPLAY


def test_phase2_simulation_confidence_score_is_evidence_adjusted():
    rows = [_make_usage_row(i) for i in range(1, 11)]
    replay_results = [_make_replay_result(i) for i in range(1, 11)]
    sims = simulate_from_replay_data(rows, replay_results, date_range_days=30)
    successful = [s for s in sims if s.records_analyzed > 0]
    assert successful
    for s in successful:
        assert 0.0 <= s.confidence_score <= 0.95
        expected = adjust_confidence_for_evidence(
            s.base_confidence_score, EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED
        )
        assert s.confidence_score == expected


def test_phase2_confidence_score_greater_than_phase1():
    """OBSERVED_REPLAY must yield higher adjusted confidence than ESTIMATED for same base."""
    records_p1 = [_make_record(days_ago=i) for i in range(10)]
    scenario = MigrationScenario(
        scenario_name="gpt4o-to-mini:classification",
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_types_included=["classification"],
        migration_percentage=1.0,
    )
    phase1 = simulate_migration(records_p1, scenario, date_range_days=30)

    rows = [_make_usage_row(i) for i in range(1, 51)]
    replay_results = [_make_replay_result(i, quality_confidence=1.0) for i in range(1, 51)]
    phase2_sims = simulate_from_replay_data(rows, replay_results, date_range_days=30)
    phase2 = next(s for s in phase2_sims if s.records_analyzed > 0)

    assert phase2.confidence_score > phase1.confidence_score


def test_all_failed_replays_have_heuristic_evidence_level():
    """All-failed simulation must not claim OBSERVED_REPLAY."""
    rows = [_make_usage_row(1)]
    replay_results = [_make_replay_result(1, error_message="Provider timed out")]
    sims = simulate_from_replay_data(rows, replay_results, date_range_days=30)
    assert sims
    failed_sims = [s for s in sims if s.records_analyzed == 0]
    assert failed_sims
    for s in failed_sims:
        assert s.evidence_level == EvidenceLevel.HEURISTIC
        assert s.validation_status == ValidationStatus.NOT_VALIDATED
        assert s.confidence_score == 0.0
        assert s.base_confidence_score == 0.0


# ── ScenarioOutcome evidence + confidence_score ───────────────────────────────

def test_scenario_outcome_inherits_evidence_from_simulation():
    sim = _make_sim(
        evidence_level=EvidenceLevel.OBSERVED_REPLAY,
        validation_status=ValidationStatus.EVALUATOR_SCORED,
    )
    outcome = _sim_to_outcome(sim)
    assert outcome.evidence_level == EvidenceLevel.OBSERVED_REPLAY
    assert outcome.validation_status == ValidationStatus.EVALUATOR_SCORED
    assert outcome.evidence_summary == "Test summary."
    assert outcome.limitations == ["Test limitation."]


def test_scenario_outcome_confidence_score_equals_sim_confidence_score():
    """_sim_to_outcome must propagate confidence_score directly — no double-adjustment."""
    sim = _make_sim(
        evidence_level=EvidenceLevel.OBSERVED_REPLAY,
        validation_status=ValidationStatus.EVALUATOR_SCORED,
        base_confidence_score=0.90,
    )
    outcome = _sim_to_outcome(sim)
    assert outcome.confidence_score == sim.confidence_score


def test_scenario_outcome_confidence_pct_matches_confidence_score():
    sim = _make_sim(base_confidence_score=0.80)
    outcome = _sim_to_outcome(sim)
    assert outcome.confidence_pct == round(sim.confidence_score * 100)


def test_scenario_outcome_limitations_instances_do_not_share_list():
    """Field(default_factory=list) fix: two instances must have independent lists."""
    o1 = ScenarioOutcome(
        source_model="gpt-4o", target_model="gpt-4o-mini", task_type="classification",
        estimated_annual_savings=700.0, quality_loss_pct=1.0, avg_quality_delta=-0.01,
        avg_latency_ms=250.0, confidence_pct=80, recommendation="migrate", rationale="good",
        records_analyzed=50,
    )
    o2 = ScenarioOutcome(
        source_model="gpt-4o", target_model="gpt-4o-mini", task_type="summarization",
        estimated_annual_savings=500.0, quality_loss_pct=2.0, avg_quality_delta=-0.02,
        avg_latency_ms=200.0, confidence_pct=75, recommendation="controlled_pilot", rationale="ok",
        records_analyzed=30,
    )
    o1.limitations.append("something")
    assert o2.limitations == [], "Shared mutable default would have been mutated"


# ── SimulationTopRecommendation evidence fields via API ───────────────────────

def test_simulate_top_scenarios_includes_evidence_fields(client):
    content = b"""prompt,response,timestamp,model,cost
"Classify this ticket","Category: billing",2024-01-15 09:00:00,gpt-4o,0.05
"Classify another ticket","Category: support",2024-01-16 09:00:00,gpt-4o,0.04
"""
    upload_resp = client.post(
        "/audits/upload",
        files={"file": ("data.csv", content, "text/csv")},
    )
    audit_run_id = upload_resp.json()["audit_run_id"]

    run_resp = client.post(f"/audits/{audit_run_id}/replay/run")
    assert run_resp.status_code == 200
    replay_run_id = run_resp.json()["replay_run_id"]

    sim_resp = client.post(f"/replay/{replay_run_id}/simulate")
    assert sim_resp.status_code == 200
    data = sim_resp.json()

    for scenario in data.get("top_scenarios", []):
        assert "evidence_level" in scenario
        assert "validation_status" in scenario
        assert "confidence_score" in scenario
        assert isinstance(scenario["confidence_score"], float)
        assert 0.0 <= scenario["confidence_score"] <= 1.0


# ── Markdown Phase 1 section separation ──────────────────────────────────────

def test_audit_markdown_has_heuristic_opportunities_section(client):
    content = b"""prompt,response,timestamp,model,cost
"Summarize this report","Summary here",2024-01-15 09:00:00,gpt-4o,0.0342
"Summarize another report","Short summary",2024-01-16 09:00:00,gpt-4o,0.0312
"""
    from tests.test_api import _upload_and_generate
    run_id = _upload_and_generate(client, content)
    resp = client.get(f"/audits/{run_id}/report/markdown")
    assert resp.status_code == 200
    assert "### Heuristic Opportunities" in resp.text


def test_audit_markdown_heuristic_opps_under_top_opportunities_header(client):
    content = b"""prompt,response,timestamp,model,cost
"Summarize this report","Summary here",2024-01-15 09:00:00,gpt-4o,0.0342
"""
    from tests.test_api import _upload_and_generate
    run_id = _upload_and_generate(client, content)
    resp = client.get(f"/audits/{run_id}/report/markdown")
    text = resp.text
    assert "## Top Opportunities" in text
    top_idx = text.index("## Top Opportunities")
    heuristic_idx = text.index("### Heuristic Opportunities")
    assert top_idx < heuristic_idx


# ── Replay markdown evidence tier subsections ─────────────────────────────────

def test_replay_markdown_has_replay_validated_subsection():
    sim = _make_sim(EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED)
    md = render_replay_markdown_report(_executive_report([_sim_to_outcome(sim)]))
    assert "## Recommended Migrations" in md
    assert "### Replay-Validated Opportunities" in md


def test_replay_markdown_llm_judge_scenario_in_llm_judged_section():
    sim = _make_sim(EvidenceLevel.LLM_JUDGE, ValidationStatus.EVALUATOR_SCORED)
    md = render_replay_markdown_report(_executive_report([_sim_to_outcome(sim)]))
    assert "### LLM-Judged Opportunities" in md
    assert "### Replay-Validated Opportunities" not in md


def test_replay_markdown_production_validated_in_production_section():
    sim = _make_sim(EvidenceLevel.PRODUCTION_VALIDATED, ValidationStatus.PRODUCTION_VALIDATED)
    md = render_replay_markdown_report(_executive_report([_sim_to_outcome(sim)]))
    assert "### Production-Validated Opportunities" in md
    assert "### Replay-Validated Opportunities" not in md
    assert "### LLM-Judged Opportunities" not in md


def test_replay_markdown_human_reviewed_in_correct_section():
    sim = _make_sim(EvidenceLevel.HUMAN_REVIEWED, ValidationStatus.HUMAN_REVIEWED)
    md = render_replay_markdown_report(_executive_report([_sim_to_outcome(sim)]))
    assert "### Human-Reviewed Opportunities" in md
    assert "### Replay-Validated Opportunities" not in md


def test_replay_markdown_heuristic_scenario_in_correct_subsection():
    sim = _make_sim(EvidenceLevel.ESTIMATED, ValidationStatus.NOT_VALIDATED)
    md = render_replay_markdown_report(_executive_report([_sim_to_outcome(sim)]))
    assert "### Heuristic / Estimated Opportunities" in md
    assert "### Replay-Validated Opportunities" not in md


def test_replay_markdown_empty_tiers_omitted():
    sim = _make_sim(EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED)
    md = render_replay_markdown_report(_executive_report([_sim_to_outcome(sim)]))
    assert "### Human-Reviewed Opportunities" not in md
    assert "### LLM-Judged Opportunities" not in md
    assert "### Production-Validated Opportunities" not in md


def test_replay_markdown_all_five_tiers_present_when_all_evidence_levels_present():
    scenarios = [
        _sim_to_outcome(_make_sim(EvidenceLevel.PRODUCTION_VALIDATED, ValidationStatus.PRODUCTION_VALIDATED)),
        _sim_to_outcome(_make_sim(EvidenceLevel.HUMAN_REVIEWED, ValidationStatus.HUMAN_REVIEWED)),
        _sim_to_outcome(_make_sim(EvidenceLevel.LLM_JUDGE, ValidationStatus.EVALUATOR_SCORED)),
        _sim_to_outcome(_make_sim(EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED)),
        _sim_to_outcome(_make_sim(EvidenceLevel.HEURISTIC, ValidationStatus.NOT_VALIDATED)),
    ]
    md = render_replay_markdown_report(_executive_report(scenarios))
    assert "### Production-Validated Opportunities" in md
    assert "### Human-Reviewed Opportunities" in md
    assert "### LLM-Judged Opportunities" in md
    assert "### Replay-Validated Opportunities" in md
    assert "### Heuristic / Estimated Opportunities" in md


def test_replay_markdown_tiers_appear_in_descending_evidence_order():
    """Higher evidence tiers must appear before lower ones in the output."""
    scenarios = [
        _sim_to_outcome(_make_sim(EvidenceLevel.PRODUCTION_VALIDATED, ValidationStatus.PRODUCTION_VALIDATED)),
        _sim_to_outcome(_make_sim(EvidenceLevel.OBSERVED_REPLAY, ValidationStatus.EVALUATOR_SCORED)),
        _sim_to_outcome(_make_sim(EvidenceLevel.HEURISTIC, ValidationStatus.NOT_VALIDATED)),
    ]
    md = render_replay_markdown_report(_executive_report(scenarios))
    prod_idx = md.index("### Production-Validated Opportunities")
    replay_idx = md.index("### Replay-Validated Opportunities")
    heuristic_idx = md.index("### Heuristic / Estimated Opportunities")
    assert prod_idx < replay_idx < heuristic_idx


def test_replay_markdown_shows_evidence_label(client):
    content = b"""prompt,response,timestamp,model,cost
"Classify this ticket","Category: billing",2024-01-15 09:00:00,gpt-4o,0.05
"Classify another ticket","Category: support",2024-01-16 09:00:00,gpt-4o,0.04
"""
    upload_resp = client.post(
        "/audits/upload",
        files={"file": ("data.csv", content, "text/csv")},
    )
    audit_run_id = upload_resp.json()["audit_run_id"]
    run_resp = client.post(f"/audits/{audit_run_id}/replay/run")
    assert run_resp.status_code == 200
    replay_run_id = run_resp.json()["replay_run_id"]
    client.post(f"/replay/{replay_run_id}/simulate")
    report_resp = client.get(f"/replay/{replay_run_id}/report/markdown")
    assert report_resp.status_code == 200
    assert "heuristic" in report_resp.text.lower()
