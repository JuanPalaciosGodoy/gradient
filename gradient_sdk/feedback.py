"""Feedback capture helpers for associating quality signals with AuditEvents."""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def log_feedback(
    client,
    event_id: str,
    feedback_score: Optional[float] = None,
    feedback_label: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Attach a feedback signal to a previously captured AuditEvent. Never raises.

    In local mode, appends a feedback record to the same JSONL file.
    In cloud mode, sends a PATCH to /sdk/events/{event_id}/feedback.
    In dry_run mode, logs what would be sent.
    In disabled mode, does nothing.
    """
    try:
        from gradient_sdk.models import FeedbackUpdate
        update = FeedbackUpdate(
            event_id=event_id,
            feedback_score=feedback_score,
            feedback_label=feedback_label,
            notes=notes,
        )
        if client.mode == "disabled":
            return
        if client.mode == "dry_run":
            logger.info("[gradient_sdk dry_run] feedback %s", update.model_dump_json())
            return
        if client.mode == "local":
            _write_local_feedback(client, update)
            return
        if client.mode == "cloud":
            _send_cloud_feedback(client, update)
    except Exception:
        logger.debug("[gradient_sdk] log_feedback failed (non-fatal)", exc_info=True)


def _write_local_feedback(client, update) -> None:
    record = {"_type": "feedback", **update.model_dump(mode="json")}
    with client._lock:
        with client.local_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _send_cloud_feedback(client, update) -> None:
    import urllib.request
    payload = json.dumps(update.model_dump(mode="json")).encode()
    req = urllib.request.Request(
        f"{client.api_url}/sdk/events/{update.event_id}/feedback",
        data=payload,
        headers={"Content-Type": "application/json", "X-API-Key": client.api_key},
        method="PATCH",
    )
    with urllib.request.urlopen(req, timeout=5):
        pass
