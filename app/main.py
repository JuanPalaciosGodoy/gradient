import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.audit.report_builder import build_report
from app.database import (
    audit_run_exists,
    get_human_reviews_for_run,
    get_latest_human_review_time,
    get_latest_simulation_created_at,
    get_replay_run,
    get_report,
    get_usage_records,
    init_db,
    save_audit_run,
    save_human_review,
    save_report,
    save_usage_records,
    update_replay_result_evidence,
)
from app.audit.task_classifier import classify_task
from app.ingestion.validators import METADATA_ONLY_SENTINEL
from app.evaluation.factory import get_evaluator
from app.ingestion.csv_loader import load_csv
from app.providers.router import get_generation_provider
from app.replay.human_review import apply_human_reviews, export_replay_results_csv, parse_import_csv
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
from app.sdk_ingestion import (
    FeedbackRequest,
    SDKEventsRequest,
    SDKEventsResponse,
    save_sdk_event,
    update_sdk_event_feedback,
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

    # Classify task_type for any record where it wasn't explicitly provided in the CSV.
    # Metadata-only records (sentinel prompt) are tagged as OTHER rather than classified.
    records_dicts = []
    for r in records:
        d = r.model_dump(mode="json")
        if d.get("task_type") is None:
            if r.prompt == METADATA_ONLY_SENTINEL:
                d["task_type"] = TaskType.OTHER.value
            else:
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

    # Filter out metadata-only records (prompt/response not captured).
    replayable = [
        r for r in raw
        if r.get("prompt") != METADATA_ONLY_SENTINEL
        and r.get("response") != METADATA_ONLY_SENTINEL
    ]
    if not replayable:
        raise HTTPException(
            status_code=422,
            detail=(
                "Replay requires captured prompt and response content. "
                "Re-run the SDK with capture_prompt=True and capture_response=True "
                "for this workflow, then upload a new audit."
            ),
        )
    raw = replayable

    replay_requests = build_replay_requests_from_rows(raw)

    # Resolve candidates
    if body.candidate_models:
        candidates = [c for c in REPLAY_CANDIDATES if c.model in body.candidate_models and c.enabled]
    else:
        candidates = [c for c in REPLAY_CANDIDATES if c.enabled]

    if not candidates:
        raise HTTPException(status_code=422, detail="No valid candidate models found.")

    evaluator = get_evaluator(body.evaluator_mode)
    runner = ReplayRunner(provider_factory=get_generation_provider, evaluator=evaluator)
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
                evidence_level=s.evidence_level,
                evidence_summary=s.evidence_summary,
                validation_status=s.validation_status,
                confidence_score=s.confidence_score,
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

    # Detect stale simulations: if human reviews were imported after the last simulation
    # was computed, the report would show pre-review evidence levels. Return 409 so the
    # caller knows to re-run /simulate before fetching the report.
    latest_sim_time = get_latest_simulation_created_at(replay_run_id)
    latest_review_time = get_latest_human_review_time(replay_run_id)
    if latest_sim_time and latest_review_time and latest_review_time > latest_sim_time:
        raise HTTPException(
            status_code=409,
            detail=(
                "Human reviews were imported after the last simulation run. "
                f"Re-run POST /replay/{replay_run_id}/simulate to incorporate "
                "human review evidence into migration simulations."
            ),
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


# ── Human review endpoints ────────────────────────────────────────────────────

@app.get("/replay/{replay_run_id}/review/export")
def export_review_csv(replay_run_id: str):
    replay_run = get_replay_run(replay_run_id)
    if replay_run is None:
        raise HTTPException(status_code=404, detail="Replay run not found.")

    audit_run_id = replay_run.get("audit_run_id")
    usage_records = get_usage_records(audit_run_id) if audit_run_id else []
    replay_results = get_replay_results(replay_run_id)

    if not replay_results:
        raise HTTPException(status_code=422, detail="No replay results to export.")

    csv_content = export_replay_results_csv(replay_run_id, replay_results, usage_records)
    filename = f"review_{replay_run_id[:8]}.csv"
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/replay/{replay_run_id}/review/import")
async def import_review_csv(replay_run_id: str, file: UploadFile = File(...)):
    replay_run = get_replay_run(replay_run_id)
    if replay_run is None:
        raise HTTPException(status_code=404, detail="Replay run not found.")

    content = await file.read()
    try:
        csv_text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="File must be UTF-8 encoded.")

    try:
        rows = parse_import_csv(csv_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not rows:
        raise HTTPException(status_code=422, detail="No valid review rows found in CSV.")

    replay_results = get_replay_results(replay_run_id)
    valid_replay_ids = {rr["replay_id"] for rr in replay_results}
    unknown_ids = [r["replay_id"] for r in rows if r["replay_id"] not in valid_replay_ids]
    if unknown_ids:
        raise HTTPException(
            status_code=422,
            detail=(
                f"These replay_result_ids do not belong to replay run {replay_run_id}: "
                + ", ".join(unknown_ids)
            ),
        )

    updates = apply_human_reviews(rows, replay_results)

    for upd in updates:
        save_human_review(
            replay_run_id=replay_run_id,
            replay_id=upd["replay_id"],
            reviewer_label=upd["reviewer_label"],
            reviewer_notes=upd["reviewer_notes"],
            reviewed_at=upd["reviewed_at"],
        )
        update_replay_result_evidence(
            replay_id=upd["replay_id"],
            replay_run_id=replay_run_id,
            evidence_level=upd["evidence_level"],
            validation_status=upd["validation_status"],
            confidence_score=upd["confidence_score"],
        )

    return {
        "status": "complete",
        "reviews_applied": len(updates),
        "replay_run_id": replay_run_id,
    }


# ── SDK ingestion ─────────────────────────────────────────────────────────────

@app.post("/sdk/events", response_model=SDKEventsResponse, tags=["SDK"])
def ingest_sdk_events(body: SDKEventsRequest) -> SDKEventsResponse:
    """Accept a batch of AuditEvents from the Gradient SDK.

    Duplicate event_ids are counted separately and not double-counted as accepted.
    Returns accepted/duplicate/rejected counts and per-event error messages.
    """
    accepted = 0
    duplicate = 0
    rejected = 0
    errors: list[str] = []
    for event in body.events:
        try:
            inserted = save_sdk_event(event)
            if inserted:
                accepted += 1
            else:
                duplicate += 1
        except Exception as exc:
            rejected += 1
            errors.append(f"{event.event_id}: {exc}")
    return SDKEventsResponse(accepted=accepted, duplicate=duplicate, rejected=rejected, errors=errors)


@app.patch("/sdk/events/{event_id}/feedback", tags=["SDK"])
def update_event_feedback(event_id: str, body: FeedbackRequest):
    """Attach or update feedback on a previously ingested SDK event.

    Returns 404 if the event_id is unknown.
    """
    updated = update_sdk_event_feedback(
        event_id=event_id,
        feedback_score=body.feedback_score,
        feedback_label=body.feedback_label,
        notes=body.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"SDK event not found: {event_id}")
    return {"event_id": event_id, "updated": True}
