"""Tests for GradientClient modes."""
import json
import os
import tempfile

import pytest

from gradient_sdk import GradientClient
from gradient_sdk.models import AuditEvent


def _event(**kwargs) -> AuditEvent:
    defaults = dict(provider="openai", model="gpt-4o-mini", workflow="w", task_type="t")
    defaults.update(kwargs)
    return AuditEvent(**defaults)


# ── local mode ────────────────────────────────────────────────────────────────

def test_local_mode_writes_jsonl():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        client = GradientClient(mode="local", local_path=path)
        client.capture(_event())
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["workflow"] == "w"
    finally:
        os.unlink(path)


def test_local_mode_appends_multiple_events():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        client = GradientClient(mode="local", local_path=path)
        for i in range(5):
            client.capture(_event(team=f"team-{i}"))
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 5
    finally:
        os.unlink(path)


def test_local_mode_event_is_valid_json():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        client = GradientClient(mode="local", local_path=path)
        e = _event(team="eng", latency_ms=125.5, input_tokens=80)
        client.capture(e)
        with open(path) as f:
            data = json.loads(f.read().strip())
        assert data["event_id"] == e.event_id
        assert data["team"] == "eng"
        assert data["latency_ms"] == pytest.approx(125.5)
    finally:
        os.unlink(path)


# ── disabled mode ─────────────────────────────────────────────────────────────

def test_disabled_mode_writes_nothing():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        client = GradientClient(mode="disabled", local_path=path)
        client.capture(_event())
        assert os.path.getsize(path) == 0
    finally:
        os.unlink(path)


def test_disabled_mode_does_not_raise():
    client = GradientClient(mode="disabled")
    client.capture(_event())  # must not raise


# ── dry_run mode ──────────────────────────────────────────────────────────────

def test_dry_run_mode_does_not_write_file():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        client = GradientClient(mode="dry_run", local_path=path)
        client.capture(_event())
        assert os.path.getsize(path) == 0
    finally:
        os.unlink(path)


def test_dry_run_mode_logs_event(caplog):
    import logging
    client = GradientClient(mode="dry_run")
    with caplog.at_level(logging.INFO, logger="gradient_sdk.client"):
        client.capture(_event(workflow="my_workflow"))
    assert "my_workflow" in caplog.text


# ── cloud mode ────────────────────────────────────────────────────────────────

def test_cloud_mode_buffers_events_up_to_batch_size():
    client = GradientClient(mode="cloud", batch_size=5, api_url="http://localhost:9999")
    for i in range(4):
        client.capture(_event())
    assert len(client._pending) == 4


def test_cloud_mode_auto_flushes_at_batch_size():
    """At batch_size events, client attempts flush (which fails for test server, but clears buffer)."""
    client = GradientClient(mode="cloud", batch_size=3, api_url="http://localhost:9999")
    # Manually mock flush to avoid real HTTP
    flushed = []

    def _fake_flush():
        flushed.extend(client._pending)
        client._pending = []

    client._flush_cloud = _fake_flush

    for _ in range(3):
        client.capture(_event())

    assert len(flushed) == 3
    assert len(client._pending) == 0


# ── capture never raises ──────────────────────────────────────────────────────

def test_capture_never_raises_on_broken_storage():
    """Even if the underlying write fails, capture() must not propagate the exception."""
    client = GradientClient(mode="local", local_path="/nonexistent/path/audit.jsonl")
    client.capture(_event())  # must not raise


# ── log_feedback ──────────────────────────────────────────────────────────────

def test_log_feedback_local_mode_writes_feedback_line():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        client = GradientClient(mode="local", local_path=path)
        client.log_feedback(event_id="abc-123", feedback_score=1.0, feedback_label="accepted")
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["_type"] == "feedback"
        assert data["event_id"] == "abc-123"
        assert data["feedback_label"] == "accepted"
    finally:
        os.unlink(path)


def test_log_feedback_disabled_does_nothing():
    client = GradientClient(mode="disabled")
    client.log_feedback(event_id="x", feedback_score=1.0)  # must not raise


# ── cloud flush atomicity ─────────────────────────────────────────────────────

def test_failed_cloud_flush_leaves_pending_unchanged():
    """If the HTTP POST fails, _pending must be left intact for retry."""
    import urllib.request

    client = GradientClient(mode="cloud", batch_size=100, api_url="http://localhost:9999")
    client._pending = [_event(), _event()]

    original_count = len(client._pending)

    def _raise(*a, **kw):
        raise OSError("connection refused")

    # Intercept at the urlopen level so the entire POST path fails
    import unittest.mock as mock
    with mock.patch.object(urllib.request, "urlopen", side_effect=_raise):
        # flush() catches SDK errors, so we call _flush_cloud directly to observe the state
        try:
            with client._lock:
                client._flush_cloud()
        except OSError:
            pass

    assert len(client._pending) == original_count


def test_successful_cloud_flush_clears_pending():
    """After a confirmed POST, _pending must be empty."""
    import urllib.request
    import unittest.mock as mock

    client = GradientClient(mode="cloud", batch_size=100, api_url="http://localhost:9999")
    client._pending = [_event(), _event(), _event()]

    fake_response = mock.MagicMock()
    fake_response.__enter__ = lambda s: s
    fake_response.__exit__ = mock.MagicMock(return_value=False)
    fake_response.status = 200

    with mock.patch.object(urllib.request, "urlopen", return_value=fake_response):
        with client._lock:
            client._flush_cloud()

    assert len(client._pending) == 0


def test_auto_flush_at_batch_size_preserves_events_when_http_fails():
    """When batch_size is reached and HTTP fails, _pending retains the events."""
    import urllib.request
    import unittest.mock as mock

    client = GradientClient(mode="cloud", batch_size=2, api_url="http://localhost:9999")

    def _raise(*a, **kw):
        raise OSError("connection refused")

    with mock.patch.object(urllib.request, "urlopen", side_effect=_raise):
        # Adding 2 events triggers auto-flush; flush fails; events stay buffered
        client.capture(_event())
        client.capture(_event())

    assert len(client._pending) == 2


# ── log_feedback cloud mode ────────────────────────────────────────────────────

def test_log_feedback_cloud_mode_hits_feedback_endpoint():
    """Cloud-mode log_feedback must PATCH /sdk/events/{event_id}/feedback."""
    import urllib.request
    import unittest.mock as mock

    client = GradientClient(mode="cloud", api_key="test-key", api_url="http://localhost:9999")

    captured_requests = []

    def _fake_urlopen(req, timeout=None):
        captured_requests.append(req)
        resp = mock.MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = mock.MagicMock(return_value=False)
        resp.status = 200
        return resp

    with mock.patch.object(urllib.request, "urlopen", side_effect=_fake_urlopen):
        client.log_feedback(
            event_id="evt-abc",
            feedback_score=0.9,
            feedback_label="accepted",
            notes="looks good",
        )

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert "/sdk/events/evt-abc/feedback" in req.full_url
    assert req.method == "PATCH"
