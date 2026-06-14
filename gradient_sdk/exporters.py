"""Export captured AuditEvents to JSONL, CSV, or Phase 1 compatible CSV."""
import csv
import json
from pathlib import Path

from gradient_sdk.models import AuditEvent

_ALL_FIELDS = [
    "event_id", "timestamp", "provider", "model", "workflow", "task_type", "status",
    "latency_ms", "input_tokens", "output_tokens", "estimated_cost", "error_message",
    "user_id_hash", "team", "business_unit", "process_name", "tool_name", "environment",
    "risk_level", "value_metric_name", "value_metric_value", "feedback_score", "feedback_label",
    "prompt", "response",
]

# Metadata-only: same as above, without the sensitive fields
_METADATA_FIELDS = [f for f in _ALL_FIELDS if f not in ("prompt", "response")]

# Phase 1 upload format expected by Gradient's audit engine
_PHASE1_FIELDS = ["prompt", "response", "timestamp", "model", "cost", "feedback", "task_type"]


def export_jsonl(events: list[AuditEvent], path: str | Path) -> None:
    """Write events as newline-delimited JSON."""
    with Path(path).open("w", encoding="utf-8") as f:
        for event in events:
            f.write(event.model_dump_json() + "\n")


def export_csv(events: list[AuditEvent], path: str | Path) -> None:
    """Write all event fields to CSV (includes prompt and response if present)."""
    _write_csv(events, path, _ALL_FIELDS)


def export_metadata_csv(events: list[AuditEvent], path: str | Path) -> None:
    """Write event metadata to CSV, excluding prompt and response."""
    _write_csv(events, path, _METADATA_FIELDS)


def export_phase1_csv(events: list[AuditEvent], path: str | Path) -> None:
    """Write a CSV compatible with the Gradient Phase 1 audit upload format.

    Maps SDK fields → Phase 1 column names:
        estimated_cost → cost
        feedback_label → feedback
    Rows without a prompt or response will have those columns empty.
    """
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PHASE1_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            writer.writerow({
                "prompt": e.prompt or "",
                "response": e.response or "",
                "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                "model": e.model,
                "cost": e.estimated_cost if e.estimated_cost is not None else "",
                "feedback": e.feedback_label or "",
                "task_type": e.task_type,
            })


def load_jsonl(path: str | Path) -> list[AuditEvent]:
    """Load AuditEvents from a JSONL file (skips feedback update lines)."""
    events = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("_type") == "feedback":
                continue
            events.append(AuditEvent.model_validate(data))
    return events


def _write_csv(events: list[AuditEvent], path: str | Path, fields: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            row = e.model_dump()
            row["timestamp"] = row["timestamp"].isoformat() if row.get("timestamp") else ""
            writer.writerow({k: row.get(k, "") for k in fields})
