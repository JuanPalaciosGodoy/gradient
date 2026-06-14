"""Tests for coverage-aware evidence promotion policy (Phase 2.5)."""
from collections import Counter

import pytest

from app.replay.migration_simulator import (
    _PROMOTION_THRESHOLD,
    _build_evidence_summary,
    _compute_sim_confidence,
    _derive_scenario_evidence,
)
from app.schemas import EvidenceLevel, ValidationStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _results(counts: dict[str, int]) -> list[dict]:
    """Build minimal result dicts with the given evidence level distribution."""
    out = []
    for level, n in counts.items():
        for _ in range(n):
            out.append({"evidence_level": level, "validation_status": "evaluator_scored"})
    return out


# ── Core threshold tests ──────────────────────────────────────────────────────

def test_1hr_99obs_stays_at_observed_replay():
    results = _results({"human_reviewed": 1, "observed_replay": 99})
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(results)
    assert level == EvidenceLevel.OBSERVED_REPLAY
    assert status == ValidationStatus.EVALUATOR_SCORED
    assert counts["human_reviewed"] == 1
    assert counts["observed_replay"] == 99
    # evidence_coverage_pct is fraction at or above observed_replay = 100/100 = 1.0
    assert abs(coverage_pct - 1.0) < 0.01
    # summary should mention partial human review coverage
    assert "human review coverage" in summary
    assert "1%" in summary


def test_80hr_20obs_promotes_to_human_reviewed():
    results = _results({"human_reviewed": 80, "observed_replay": 20})
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(results)
    assert level == EvidenceLevel.HUMAN_REVIEWED
    assert status == ValidationStatus.HUMAN_REVIEWED
    assert abs(coverage_pct - 0.80) < 0.01
    assert partial == 0.0


def test_79hr_21obs_stays_at_observed_replay():
    results = _results({"human_reviewed": 79, "observed_replay": 21})
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(results)
    assert level == EvidenceLevel.OBSERVED_REPLAY
    assert "79%" in summary or "human review coverage" in summary


def test_80llm_20obs_promotes_to_llm_judge():
    results = _results({"llm_judge": 80, "observed_replay": 20})
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(results)
    assert level == EvidenceLevel.LLM_JUDGE
    assert status == ValidationStatus.EVALUATOR_SCORED
    assert abs(coverage_pct - 0.80) < 0.01
    assert partial == 0.0


def test_79llm_21obs_stays_at_observed_replay():
    results = _results({"llm_judge": 79, "observed_replay": 21})
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(results)
    assert level == EvidenceLevel.OBSERVED_REPLAY
    assert "LLM-judge coverage" in summary


def test_all_observed_replay_stays_at_observed_replay():
    results = _results({"observed_replay": 50})
    level, status, coverage_pct, counts, partial, summary = _derive_scenario_evidence(results)
    assert level == EvidenceLevel.OBSERVED_REPLAY
    # 100% of results are at or above observed_replay
    assert abs(coverage_pct - 1.0) < 0.01
    assert partial == 0.0  # nothing above observed_replay tier


def test_exactly_at_threshold_promotes():
    """The boundary case: exactly 80% is a promotion."""
    results = _results({"human_reviewed": 4, "observed_replay": 1})
    level, _, _, _, _, _ = _derive_scenario_evidence(results)
    assert level == EvidenceLevel.HUMAN_REVIEWED


def test_one_below_threshold_does_not_promote():
    """79.something% rounds below 80% and must not promote."""
    # 79 HR + 22 OBS = 79/101 = 78.2%
    results = _results({"human_reviewed": 79, "observed_replay": 22})
    level, _, _, _, _, _ = _derive_scenario_evidence(results)
    assert level == EvidenceLevel.OBSERVED_REPLAY


def test_production_validated_threshold():
    results = _results({"production_validated": 80, "observed_replay": 20})
    level, status, coverage_pct, _, _, _ = _derive_scenario_evidence(results)
    assert level == EvidenceLevel.PRODUCTION_VALIDATED
    assert status == ValidationStatus.PRODUCTION_VALIDATED
    assert abs(coverage_pct - 0.80) < 0.01


# ── evidence_counts dictionary ────────────────────────────────────────────────

def test_evidence_counts_dict_structure():
    results = _results({"human_reviewed": 3, "observed_replay": 7})
    _, _, _, counts, _, _ = _derive_scenario_evidence(results)
    assert isinstance(counts, dict)
    assert counts["human_reviewed"] == 3
    assert counts["observed_replay"] == 7


def test_evidence_counts_single_level():
    results = _results({"observed_replay": 10})
    _, _, _, counts, _, _ = _derive_scenario_evidence(results)
    assert counts == {"observed_replay": 10}


# ── partial_higher_pct and confidence boost ───────────────────────────────────

def test_partial_higher_pct_is_zero_when_all_obs():
    results = _results({"observed_replay": 100})
    _, _, _, _, partial, _ = _derive_scenario_evidence(results)
    assert partial == 0.0


def test_partial_higher_pct_nonzero_for_partial_hr():
    results = _results({"human_reviewed": 1, "observed_replay": 99})
    _, _, _, _, partial, _ = _derive_scenario_evidence(results)
    assert partial > 0.0
    assert abs(partial - 0.01) < 0.001


def test_partial_higher_pct_zero_when_promoted():
    results = _results({"human_reviewed": 80, "observed_replay": 20})
    _, _, _, _, partial, _ = _derive_scenario_evidence(results)
    assert partial == 0.0


def test_confidence_ordering_full_hr_gt_partial_hr_gt_pure_obs():
    """Full HR > partial HR > pure OBSERVED_REPLAY."""
    base = 0.70

    results_obs = _results({"observed_replay": 100})
    ev_obs, st_obs, _, _, p_obs, _ = _derive_scenario_evidence(results_obs)
    conf_obs = _compute_sim_confidence(base, ev_obs, st_obs, p_obs)

    results_partial = _results({"human_reviewed": 1, "observed_replay": 99})
    ev_p, st_p, _, _, p_p, _ = _derive_scenario_evidence(results_partial)
    conf_partial = _compute_sim_confidence(base, ev_p, st_p, p_p)

    results_full = _results({"human_reviewed": 80, "observed_replay": 20})
    ev_full, st_full, _, _, p_full, _ = _derive_scenario_evidence(results_full)
    conf_full = _compute_sim_confidence(base, ev_full, st_full, p_full)

    assert conf_full > conf_partial > conf_obs


def test_partial_boost_bounded_below_full_promotion():
    """Even at 79% partial coverage, confidence stays below full HUMAN_REVIEWED level."""
    base = 0.90  # use high base to stress-test the invariant

    # 79% HR — threshold not met
    results_partial = _results({"human_reviewed": 79, "observed_replay": 21})
    ev_p, st_p, _, _, p_p, _ = _derive_scenario_evidence(results_partial)
    conf_partial = _compute_sim_confidence(base, ev_p, st_p, p_p)

    # 80% HR — threshold met
    results_full = _results({"human_reviewed": 80, "observed_replay": 20})
    ev_full, st_full, _, _, p_full, _ = _derive_scenario_evidence(results_full)
    conf_full = _compute_sim_confidence(base, ev_full, st_full, p_full)

    assert conf_full > conf_partial


# ── Evidence summary text ─────────────────────────────────────────────────────

def test_summary_notes_partial_human_review():
    results = _results({"human_reviewed": 12, "observed_replay": 88})
    _, _, _, counts, _, summary = _derive_scenario_evidence(results)
    assert "12%" in summary
    assert "human review coverage" in summary
    assert "80%" in summary  # threshold reminder


def test_summary_notes_partial_llm_judge():
    results = _results({"llm_judge": 30, "observed_replay": 70})
    _, _, _, _, _, summary = _derive_scenario_evidence(results)
    assert "LLM-judge coverage" in summary
    assert "30%" in summary


def test_summary_no_partial_note_when_promoted():
    results = _results({"human_reviewed": 80, "observed_replay": 20})
    _, _, _, _, _, summary = _derive_scenario_evidence(results)
    assert "threshold" not in summary


def test_summary_no_partial_note_when_all_observed():
    results = _results({"observed_replay": 100})
    _, _, _, _, _, summary = _derive_scenario_evidence(results)
    assert "coverage" not in summary


# ── Markdown tier routing ─────────────────────────────────────────────────────

def test_markdown_79hr_stays_in_replay_tier_not_human_reviewed():
    """Scenarios below the 80% threshold must appear in Replay-Validated, not Human-Reviewed."""
    from datetime import datetime, timezone

    from app.replay.replay_models import MigrationSimulationResult
    from app.replay.replay_report import ScenarioOutcome, ExecutiveReplayReport
    from app.reports.replay_markdown import render_replay_markdown_report
    from app.schemas import EvidenceLevel, ValidationStatus

    scenario = ScenarioOutcome(
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_type="summarization",
        estimated_annual_savings=5000.0,
        quality_loss_pct=1.0,
        avg_quality_delta=-0.01,
        avg_latency_ms=150.0,
        confidence_pct=63,
        recommendation="migrate",
        rationale="Good savings.",
        records_analyzed=100,
        evidence_level=EvidenceLevel.OBSERVED_REPLAY,  # 79 HR didn't meet threshold
        evidence_summary="79 human-reviewed, 21 evaluator-scored.",
        evidence_coverage_pct=1.0,
        evidence_counts={"human_reviewed": 79, "observed_replay": 21},
        validation_status=ValidationStatus.EVALUATOR_SCORED,
        confidence_score=0.63,
    )

    report = ExecutiveReplayReport(
        replay_run_id="test-run",
        generated_at=datetime.now(timezone.utc),
        total_requests=100,
        total_results=100,
        current_annualized_spend=50000.0,
        simulated_annualized_spend=45000.0,
        estimated_annual_savings=5000.0,
        avg_quality_impact=-0.01,
        overall_confidence=63,
        executive_summary="Test.",
        recommended_migrations=[scenario],
        do_not_migrate=[],
        risk_notes=[],
        next_actions=["Test action."],
    )

    md = render_replay_markdown_report(report)

    # Must appear under Replay-Validated, NOT Human-Reviewed
    assert "### Replay-Validated Opportunities" in md
    # Human-Reviewed section should not have this scenario's models
    hr_section_start = md.find("### Human-Reviewed Opportunities")
    replay_section_start = md.find("### Replay-Validated Opportunities")
    # If HR section is present, the scenario line must NOT be in it
    if hr_section_start != -1:
        hr_content = md[hr_section_start:replay_section_start] if replay_section_start > hr_section_start else md[hr_section_start:]
        assert "gpt-4o-mini" not in hr_content


def test_markdown_80hr_appears_in_human_reviewed_tier():
    """Scenarios meeting the 80% threshold must appear in Human-Reviewed."""
    from datetime import datetime, timezone

    from app.replay.replay_report import ScenarioOutcome, ExecutiveReplayReport
    from app.reports.replay_markdown import render_replay_markdown_report
    from app.schemas import EvidenceLevel, ValidationStatus

    scenario = ScenarioOutcome(
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_type="summarization",
        estimated_annual_savings=5000.0,
        quality_loss_pct=1.0,
        avg_quality_delta=-0.01,
        avg_latency_ms=150.0,
        confidence_pct=72,
        recommendation="migrate",
        rationale="Good savings.",
        records_analyzed=100,
        evidence_level=EvidenceLevel.HUMAN_REVIEWED,  # 80 HR met threshold
        evidence_summary="100 replayed: 80 human-reviewed, 20 evaluator-scored.",
        evidence_coverage_pct=0.80,
        evidence_counts={"human_reviewed": 80, "observed_replay": 20},
        validation_status=ValidationStatus.HUMAN_REVIEWED,
        confidence_score=0.72,
    )

    report = ExecutiveReplayReport(
        replay_run_id="test-run",
        generated_at=datetime.now(timezone.utc),
        total_requests=100,
        total_results=100,
        current_annualized_spend=50000.0,
        simulated_annualized_spend=45000.0,
        estimated_annual_savings=5000.0,
        avg_quality_impact=-0.01,
        overall_confidence=72,
        executive_summary="Test.",
        recommended_migrations=[scenario],
        do_not_migrate=[],
        risk_notes=[],
        next_actions=["Test action."],
    )

    md = render_replay_markdown_report(report)
    assert "### Human-Reviewed Opportunities" in md
    # Scenario must be in the human-reviewed section
    hr_idx = md.find("### Human-Reviewed Opportunities")
    replay_idx = md.find("### LLM-Judged Opportunities")
    hr_section = md[hr_idx:replay_idx] if replay_idx > hr_idx else md[hr_idx:]
    assert "gpt-4o-mini" in hr_section


def test_markdown_coverage_note_shown_in_scenario_line():
    """Coverage percentage must appear in each scenario's evidence line."""
    from datetime import datetime, timezone

    from app.replay.replay_report import ScenarioOutcome, ExecutiveReplayReport
    from app.reports.replay_markdown import render_replay_markdown_report
    from app.schemas import EvidenceLevel, ValidationStatus

    scenario = ScenarioOutcome(
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        task_type="summarization",
        estimated_annual_savings=5000.0,
        quality_loss_pct=1.0,
        avg_quality_delta=-0.01,
        avg_latency_ms=150.0,
        confidence_pct=72,
        recommendation="migrate",
        rationale="Good savings.",
        records_analyzed=100,
        evidence_level=EvidenceLevel.HUMAN_REVIEWED,
        evidence_summary="80 human-reviewed, 20 evaluator-scored.",
        evidence_coverage_pct=0.80,
        evidence_counts={"human_reviewed": 80, "observed_replay": 20},
        validation_status=ValidationStatus.HUMAN_REVIEWED,
        confidence_score=0.72,
    )

    report = ExecutiveReplayReport(
        replay_run_id="test-run",
        generated_at=datetime.now(timezone.utc),
        total_requests=100,
        total_results=100,
        current_annualized_spend=50000.0,
        simulated_annualized_spend=45000.0,
        estimated_annual_savings=5000.0,
        avg_quality_impact=-0.01,
        overall_confidence=72,
        executive_summary="Test.",
        recommended_migrations=[scenario],
        do_not_migrate=[],
        risk_notes=[],
        next_actions=["Test action."],
    )

    md = render_replay_markdown_report(report)
    # Should show "80% coverage" in the evidence line
    assert "80% coverage" in md


# ── Integration: simulate_from_replay_data respects thresholds ────────────────

def test_simulate_from_replay_data_respects_80pct_threshold(db):
    """End-to-end: 80 HR + 20 OBS → HUMAN_REVIEWED scenario."""
    from app.replay.migration_simulator import simulate_from_replay_data

    orig = [{"id": str(i), "cost": 0.01, "model": "gpt-4o", "task_type": "summarization"} for i in range(100)]

    def _rr(i: int, evidence: str) -> dict:
        return {
            "original_record_id": str(i),
            "candidate_model": "gpt-4o-mini",
            "error_message": None,
            "estimated_cost": 0.001,
            "quality_score": 0.9,
            "quality_confidence": 0.85,
            "latency_ms": 100.0,
            "evidence_level": evidence,
            "validation_status": "evaluator_scored" if evidence != "human_reviewed" else "human_reviewed",
        }

    replay_results = [_rr(i, "human_reviewed") for i in range(80)] + [_rr(i, "observed_replay") for i in range(80, 100)]

    sims = simulate_from_replay_data(orig, replay_results, date_range_days=30)
    assert len(sims) == 1
    sim = sims[0]
    assert sim.evidence_level == EvidenceLevel.HUMAN_REVIEWED
    assert abs(sim.evidence_coverage_pct - 0.80) < 0.01
    assert sim.evidence_counts["human_reviewed"] == 80
    assert sim.evidence_counts["observed_replay"] == 20


def test_simulate_from_replay_data_79pct_stays_observed_replay(db):
    """79 HR + 21 OBS → OBSERVED_REPLAY, partial boost in evidence_summary."""
    from app.replay.migration_simulator import simulate_from_replay_data

    orig = [{"id": str(i), "cost": 0.01, "model": "gpt-4o", "task_type": "summarization"} for i in range(100)]

    replay_results = [
        {
            "original_record_id": str(i),
            "candidate_model": "gpt-4o-mini",
            "error_message": None,
            "estimated_cost": 0.001,
            "quality_score": 0.9,
            "quality_confidence": 0.85,
            "latency_ms": 100.0,
            "evidence_level": "human_reviewed" if i < 79 else "observed_replay",
            "validation_status": "human_reviewed" if i < 79 else "evaluator_scored",
        }
        for i in range(100)
    ]

    sims = simulate_from_replay_data(orig, replay_results, date_range_days=30)
    sim = sims[0]
    assert sim.evidence_level == EvidenceLevel.OBSERVED_REPLAY
    assert "human review coverage" in sim.evidence_summary
    assert sim.evidence_counts["human_reviewed"] == 79


def test_simulate_stores_and_loads_coverage_fields(db):
    """evidence_coverage_pct and evidence_counts survive a DB roundtrip."""
    from app.replay.migration_simulator import simulate_from_replay_data
    from app.replay.replay_store import (
        get_simulations_for_replay_run,
        replace_migration_simulations,
        save_replay_run,
    )

    run_id = "cov-test-run"
    save_replay_run(run_id, None, ["gpt-4o-mini"], record_count=100)

    orig = [{"id": str(i), "cost": 0.01, "model": "gpt-4o", "task_type": "summarization"} for i in range(100)]
    replay_results = [
        {
            "original_record_id": str(i),
            "candidate_model": "gpt-4o-mini",
            "error_message": None,
            "estimated_cost": 0.001,
            "quality_score": 0.9,
            "quality_confidence": 0.85,
            "latency_ms": 100.0,
            "evidence_level": "human_reviewed" if i < 80 else "observed_replay",
            "validation_status": "human_reviewed" if i < 80 else "evaluator_scored",
        }
        for i in range(100)
    ]

    sims = simulate_from_replay_data(orig, replay_results, date_range_days=30)
    replace_migration_simulations(run_id, None, sims)

    loaded = get_simulations_for_replay_run(run_id)
    assert len(loaded) == 1
    loaded_sim = loaded[0]
    assert loaded_sim.evidence_level == EvidenceLevel.HUMAN_REVIEWED
    assert abs(loaded_sim.evidence_coverage_pct - 0.80) < 0.01
    assert loaded_sim.evidence_counts.get("human_reviewed") == 80
    assert loaded_sim.evidence_counts.get("observed_replay") == 20
