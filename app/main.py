import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.audit.report_builder import build_report
from app.database import (
    audit_run_exists,
    get_report,
    get_usage_records,
    init_db,
    save_audit_run,
    save_report,
    save_usage_records,
)
from app.ingestion.csv_loader import load_csv
from app.reports.markdown import render_markdown_report
from app.schemas import (
    AuditReport,
    AuditRunStatus,
    GenerateResponse,
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
    save_usage_records(audit_run_id=audit_run_id, records=[r.model_dump(mode="json") for r in records])

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
