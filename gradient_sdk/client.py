"""GradientClient: event sink with local/cloud/dry_run/disabled modes."""
import json
import logging
import threading
from pathlib import Path
from typing import Literal, Optional

from gradient_sdk.models import AuditEvent, FeedbackUpdate

logger = logging.getLogger(__name__)


class GradientClient:
    """Primary entry point for the Gradient SDK.

    Args:
        api_key:     API key for cloud mode (ignored in other modes).
        mode:        "local" | "cloud" | "dry_run" | "disabled"
        local_path:  File path for local JSONL storage (local mode only).
        api_url:     Gradient API base URL (cloud mode only).
        batch_size:  Number of events to accumulate before auto-flushing (cloud mode).
    """

    def __init__(
        self,
        api_key: str = "",
        mode: Literal["local", "cloud", "dry_run", "disabled"] = "local",
        local_path: str = "gradient_audit.jsonl",
        api_url: str = "http://localhost:8000",
        batch_size: int = 100,
    ) -> None:
        self.api_key = api_key
        self.mode = mode
        self.local_path = Path(local_path)
        self.api_url = api_url.rstrip("/")
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._pending: list[AuditEvent] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def capture(self, event: AuditEvent) -> None:
        """Record one audit event. Never raises."""
        try:
            if self.mode == "disabled":
                return
            if self.mode == "dry_run":
                logger.info("[gradient_sdk dry_run] %s", event.model_dump_json())
                return
            if self.mode == "local":
                self._write_local(event)
                return
            if self.mode == "cloud":
                self._enqueue_cloud(event)
        except Exception:
            logger.debug("[gradient_sdk] capture failed (non-fatal)", exc_info=True)

    def flush(self) -> None:
        """Force-send any buffered events to the cloud API (cloud mode only)."""
        try:
            with self._lock:
                self._flush_cloud()
        except Exception:
            logger.debug("[gradient_sdk] flush failed (non-fatal)", exc_info=True)

    def log_feedback(
        self,
        event_id: str,
        feedback_score: Optional[float] = None,
        feedback_label: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Attach a quality or outcome signal to a previously captured event."""
        from gradient_sdk.feedback import log_feedback
        log_feedback(
            client=self,
            event_id=event_id,
            feedback_score=feedback_score,
            feedback_label=feedback_label,
            notes=notes,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write_local(self, event: AuditEvent) -> None:
        with self._lock:
            with self.local_path.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")

    def _enqueue_cloud(self, event: AuditEvent) -> None:
        with self._lock:
            self._pending.append(event)
            if len(self._pending) >= self.batch_size:
                self._flush_cloud()

    def _flush_cloud(self) -> None:
        """Send pending events. Must be called with self._lock held.

        Events are only removed from _pending after a confirmed successful POST.
        If the request fails, _pending is left intact for a future flush attempt.
        """
        if not self._pending:
            return
        import urllib.request
        to_send = list(self._pending)
        payload = json.dumps({"events": [e.model_dump(mode="json") for e in to_send]}).encode()
        req = urllib.request.Request(
            f"{self.api_url}/sdk/events",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"API returned {resp.status}")
        # Clear only the events we confirmed were sent
        self._pending = self._pending[len(to_send):]
