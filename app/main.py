import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.audit.report_builder import build_report
from app.database import (
    audit_run_exists,
    get_replay_run,
    get_report,
    get_usage_records,
    init_db,
    save_audit_run,
    save_report,
    save_usage_records,
)
from app.audit.task_classifier import classify_task
from app.evaluation.factory import get_evaluator
from app.ingestion.csv_loader import load_csv
from app.providers.fake import FakeProvider
from app.replay.migration_simulator import simulate_from_replay_data
from app.utils.date_range import calculate_date_range_days
from app.replay.replay_models import REPLAY_CANDIDATES
from app.replay.replay_report import (
    build_executive_replay_report_from_simulations,
)
from app.replay.replay_runner import ReplayRunner, build_replay_requests_from_rows
from app.replay.replay_store import (
    get_replay_results,
    get_simulations_for_replay_run,
    replace_migration_simulations,
    save_replay_results,
    save_replay_run,
)
from app.reports.markdown import render_markdown_report
from app.reports.replay_markdown import render_replay_markdown_report
from app.schemas import (
    AuditReport,
    AuditRunStatus,
    GenerateResponse,
    ReplayRunRequest,
    ReplayRunResponse,
    SimulationResponse,
    SimulationTopRecommendation,
    TaskType,
    UploadResponse,
    UsageRecord,
    ValidationSummary,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Gradient",
    description="AI Procurement Audit Platform",
    version="0.1.0",
    lifespan=lifespan,
)


def _build_records(raw: list[dict]) -> list[UsageRecord]:
    records = []
    for r in raw:
        task_type = TaskType(r["task_type"]) if r.get("task_type") else None
        records.append(UsageRecord(
            prompt=r["prompt"],
            response=r["response"],
            timestamp=datetime.fromisoformat(r["timestamp"]),
            model=r["model"],
            cost=r["cost"],
            feedback=r.get("feedback"),
            task_type=task_type,
        ))
    return records


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/audits/upload", response_model=UploadResponse)
async def upload_audit(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()

    try:
        records = load_csv(content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    audit_run_id = str(uuid.uuid4())
    save_audit_run(run_id=audit_run_id, file_name=file.filename, record_count=len(records))

    # Classify task_type for any record where it wasn't explicitly provided in the CSV
    records_dicts = []
    for r in records:
        d = r.model_dump(mode="json")
        if d.get("task_type") is None:
            d["task_type"] = classify_task(r.prompt).value
        records_dicts.append(d)
    save_usage_records(audit_run_id=audit_run_id, records=records_dicts)

    return UploadResponse(
        audit_run_id=audit_run_id,
        record_count=len(records),
        status=AuditRunStatus.COMPLETE,
        validation_summary=ValidationSummary(
            total_rows=len(records),
            valid_rows=len(records),
            invalid_rows=0,
            error_samples=[],
        ),
    )


@app.post("/audits/{audit_run_id}/generate", response_model=GenerateResponse)
def generate_report(audit_run_id: str):
    raw = get_usage_records(audit_run_id)
    if not raw:
        raise HTTPException(status_code=404, detail="Audit run not found.")

    records = _build_records(raw)
    report = build_report(audit_run_id, records)
    save_report(report)

    return GenerateResponse(
        audit_run_id=audit_run_id,
        report_id=report.report_id,
        status="complete",
    )


@app.get("/audits/{audit_run_id}/report", response_model=AuditReport)
def fetch_report(audit_run_id: str):
    if not audit_run_exists(audit_run_id):
        raise HTTPException(status_code=404, detail="Audit run not found.")
    report = get_report(audit_run_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Report not found. Call POST /audits/{audit_run_id}/generate first.",
        )
    return report


@app.get("/audits/{audit_run_id}/report/markdown", response_class=PlainTextResponse)
def fetch_report_markdown(audit_run_id: str):
    if not audit_run_exists(audit_run_id):
        raise HTTPException(status_code=404, detail="Audit run not found.")
    report = get_report(audit_run_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Report not found. Call POST /audits/{audit_run_id}/generate first.",
        )
    return render_markdown_report(report)


# ── Replay endpoints ──────────────────────────────────────────────────────────

@app.post("/audits/{audit_run_id}/replay/run", response_model=ReplayRunResponse)
def run_replay(audit_run_id: str, body: ReplayRunRequest = ReplayRunRequest()):
    raw = get_usage_records(audit_run_id)
    if not raw:
        raise HTTPException(status_code=404, detail="Audit run not found.")

    # Optional filter by task type
    if body.task_types:
        raw = [r for r in raw if r.get("task_type") in body.task_types]

    # Optional record cap
    if body.max_records and body.max_records > 0:
        raw = raw[:body.max_records]

    if not raw:
        raise HTTPException(
            status_code=422,
            detail="No records match the specified criteria.",
        )

    replay_requests = build_replay_requests_from_rows(raw)

    # Resolve candidates
    if body.candidate_models:
        candidates = [c for c in REPLAY_CANDIDATES if c.model in body.candidate_models and c.enabled]
    else:
        candidates = [c for c in REPLAY_CANDIDATES if c.enabled]

    if not candidates:
        raise HTTPException(status_code=422, detail="No valid candidate models found.")

    evaluator = get_evaluator(body.evaluator_mode)
    runner = ReplayRunner(provider=FakeProvider(), evaluator=evaluator)
    results = runner.run(replay_requests, candidates)

    if not results:
        raise HTTPException(
            status_code=422,
            detail=(
                "No replay results were produced. All selected candidate models "
                "match the source model(s) in this audit run and were skipped."
            ),
        )

    replay_run_id = str(uuid.uuid4())
    save_replay_run(
        replay_run_id=replay_run_id,
        audit_run_id=audit_run_id,
        candidate_models=[c.model for c in candidates],
        record_count=len(replay_requests),
    )
    save_replay_results(replay_run_id, results)

    return ReplayRunResponse(
        replay_run_id=replay_run_id,
        status="complete",
        records_selected=len(replay_requests),
        candidates_selected=[c.model for c in candidates],
    )


@app.post("/replay/{replay_run_id}/simulate", response_model=SimulationResponse)
def simulate_replay(replay_run_id: str):
    replay_run = get_replay_run(replay_run_id)
    if replay_run is None:
        raise HTTPException(status_code=404, detail="Replay run not found.")

    audit_run_id = replay_run.get("audit_run_id")
    if not audit_run_id:
        raise HTTPException(status_code=422, detail="Replay run has no associated audit.")

    original_records = get_usage_records(audit_run_id)
    replay_results_raw = get_replay_results(replay_run_id)

    timestamps = [
        datetime.fromisoformat(r["timestamp"])
        for r in original_records if r.get("timestamp")
    ]
    date_range_days = calculate_date_range_days(timestamps)

    simulations = simulate_from_replay_data(original_records, replay_results_raw, date_range_days)

    replace_migration_simulations(replay_run_id, audit_run_id, simulations)

    top = sorted(simulations, key=lambda s: -s.estimated_annual_savings)[:3]

    return SimulationResponse(
        replay_run_id=replay_run_id,
        simulations_count=len(simulations),
        top_scenarios=[
            SimulationTopRecommendation(
                scenario_name=s.scenario_name,
                recommendation=s.recommendation,
                estimated_annual_savings=s.estimated_annual_savings,
                confidence_pct=round(s.confidence_score * 100),
            )
            for s in top
        ],
    )


def _load_replay_report(replay_run_id: str):
    """Shared helper: load persisted simulations and compute totals for the report."""
    replay_run = get_replay_run(replay_run_id)
    if replay_run is None:
        raise HTTPException(status_code=404, detail="Replay run not found.")

    simulations = get_simulations_for_replay_run(replay_run_id)
    if not simulations:
        raise HTTPException(
            status_code=422,
            detail="No simulations found. Run POST /replay/{replay_run_id}/simulate first.",
        )

    audit_run_id = replay_run.get("audit_run_id")
    original_records = get_usage_records(audit_run_id) if audit_run_id else []
    replay_results_raw = get_replay_results(replay_run_id)

    timestamps = [
        datetime.fromisoformat(r["timestamp"])
        for r in original_records if r.get("timestamp")
    ]
    date_range_days = calculate_date_range_days(timestamps)
    annualization_factor = 365 / max(date_range_days, 1)

    replayed_ids = {str(rr["original_record_id"]) for rr in replay_results_raw}
    current_annualized = (
        sum(r["cost"] for r in original_records if str(r["id"]) in replayed_ids)
        * annualization_factor
    )

    return build_executive_replay_report_from_simulations(
        replay_run_id=replay_run_id,
        simulations=simulations,
        current_annualized_spend=current_annualized,
        total_requests=replay_run.get("record_count", len(original_records)),
        total_results=len(replay_results_raw),
    )


@app.get("/replay/{replay_run_id}/report")
def fetch_replay_report(replay_run_id: str):
    return _load_replay_report(replay_run_id)


@app.get("/replay/{replay_run_id}/report/markdown", response_class=PlainTextResponse)
def fetch_replay_report_markdown(replay_run_id: str):
    report = _load_replay_report(replay_run_id)
    return render_replay_markdown_report(report)
