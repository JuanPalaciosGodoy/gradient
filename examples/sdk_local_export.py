"""
Gradient SDK — load events from JSONL and export for Phase 1 upload.

After collecting audit events locally, export to:
  - metadata CSV (for spend analysis without raw content)
  - Phase 1 CSV  (compatible with Gradient audit engine upload)
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_sdk import GradientClient, audit, AuditEvent
from gradient_sdk.exporters import (
    export_metadata_csv,
    export_phase1_csv,
    load_jsonl,
)

# ── Step 1: generate some events locally ─────────────────────────────────────

client = GradientClient(mode="local", local_path="demo_export.jsonl")


@audit(
    client=client,
    workflow="ticket_triage",
    task_type="classification",
    provider="openai",
    model="gpt-4o-mini",
    team="support",
)
def triage(ticket: str) -> dict:
    return {
        "text": "billing",
        "input_tokens": 80,
        "output_tokens": 5,
        "estimated_cost": 0.000014,
    }


if __name__ == "__main__":
    for i in range(5):
        triage(f"Ticket #{i}: question about my invoice.")

    # ── Step 2: load and export ───────────────────────────────────────────────
    events = load_jsonl("demo_export.jsonl")
    print(f"Loaded {len(events)} events from demo_export.jsonl")

    export_metadata_csv(events, "audit_metadata_export.csv")
    print("Metadata CSV (no content) → audit_metadata_export.csv")

    export_phase1_csv(events, "audit_phase1_upload.csv")
    print("Phase 1 upload CSV → audit_phase1_upload.csv")
    print()
    print("Upload audit_phase1_upload.csv to Gradient to run the audit engine.")
