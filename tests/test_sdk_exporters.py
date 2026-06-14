"""Tests for gradient_sdk.exporters."""
import csv
import io
import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from gradient_sdk.exporters import (
    export_csv,
    export_jsonl,
    export_metadata_csv,
    export_phase1_csv,
    load_jsonl,
)
from gradient_sdk.models import AuditEvent


def _event(**kwargs) -> AuditEvent:
    defaults = dict(
        provider="openai",
        model="gpt-4o-mini",
        workflow="test_workflow",
        task_type="summarization",
        latency_ms=150.0,
        input_tokens=80,
        output_tokens=30,
        estimated_cost=0.000035,
        team="engineering",
    )
    defaults.update(kwargs)
    return AuditEvent(**defaults)


# ── JSONL export / load ───────────────────────────────────────────────────────

def test_export_jsonl_creates_file():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        export_jsonl([_event()], path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_export_jsonl_one_line_per_event():
    events = [_event() for _ in range(3)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        export_jsonl(events, path)
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3
    finally:
        os.unlink(path)


def test_load_jsonl_roundtrip():
    events = [_event(team=f"team-{i}") for i in range(5)]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        export_jsonl(events, path)
        loaded = load_jsonl(path)
        assert len(loaded) == 5
        for orig, back in zip(events, loaded):
            assert orig.event_id == back.event_id
            assert orig.workflow == back.workflow
            assert orig.team == back.team
    finally:
        os.unlink(path)


def test_load_jsonl_skips_feedback_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        path = f.name
        e = _event()
        f.write(e.model_dump_json() + "\n")
        f.write(json.dumps({"_type": "feedback", "event_id": e.event_id, "feedback_score": 1.0}) + "\n")
    try:
        loaded = load_jsonl(path)
        assert len(loaded) == 1
    finally:
        os.unlink(path)


# ── CSV export ────────────────────────────────────────────────────────────────

def test_export_csv_contains_all_fields():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        export_csv([_event()], path)
        with open(path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        assert "prompt" in fieldnames
        assert "response" in fieldnames
        assert "event_id" in fieldnames
        assert "workflow" in fieldnames
    finally:
        os.unlink(path)


def test_export_csv_row_values():
    e = _event(team="support", estimated_cost=0.001)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        export_csv([e], path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["team"] == "support"
        assert row["workflow"] == "test_workflow"
    finally:
        os.unlink(path)


# ── Metadata-only export ──────────────────────────────────────────────────────

def test_export_metadata_csv_excludes_prompt_and_response():
    e = _event(prompt="secret prompt", response="secret response")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        export_metadata_csv([e], path)
        with open(path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        assert "prompt" not in fieldnames
        assert "response" not in fieldnames
    finally:
        os.unlink(path)


def test_export_metadata_csv_includes_business_fields():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        export_metadata_csv([_event(team="ops", risk_level="high")], path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["team"] == "ops"
        assert row["risk_level"] == "high"
        assert row["model"] == "gpt-4o-mini"
    finally:
        os.unlink(path)


# ── Phase 1 compatible export ─────────────────────────────────────────────────

def test_export_phase1_csv_has_correct_columns():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        export_phase1_csv([_event()], path)
        with open(path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        for col in ("prompt", "response", "timestamp", "model", "cost", "feedback", "task_type"):
            assert col in fieldnames
    finally:
        os.unlink(path)


def test_export_phase1_csv_maps_estimated_cost_to_cost():
    e = _event(estimated_cost=0.00125, feedback_label="accepted")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        export_phase1_csv([e], path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["cost"] == "0.00125"
        assert row["feedback"] == "accepted"
    finally:
        os.unlink(path)


def test_export_phase1_csv_populates_prompt_and_response_when_captured():
    e = _event(prompt="my prompt", response="my response")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        export_phase1_csv([e], path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["prompt"] == "my prompt"
        assert row["response"] == "my response"
    finally:
        os.unlink(path)


def test_export_phase1_csv_empty_prompt_when_not_captured():
    e = _event()  # no prompt or response
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        export_phase1_csv([e], path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["prompt"] == ""
        assert row["response"] == ""
    finally:
        os.unlink(path)
