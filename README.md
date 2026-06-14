# Gradient

**AI Procurement Audit — Phase 1**

Gradient tells companies what they should actually do about their AI spend. Not how many tokens they used. Not which models are fastest. What they should *do* — switch models, consolidate vendors, reallocate budget.

---

## What it does

You upload your historical AI usage data (prompt, response, model, cost, timestamp). Gradient produces an executive procurement audit report:

- **Current annualized AI spend**
- **Potential annual savings** with specific model recommendations
- **Confidence score** based on data quality and coverage
- **Top optimization opportunities** ranked by dollar impact
- **Model and task cost breakdown**
- **Risk notes** — concentration risk, spend level, data gaps
- **Recommended next actions**

The output is designed to be dropped into a board deck or procurement review, not a developer dashboard.

---

## Why this is an audit, not a dashboard

Dashboards show you what happened. Audits tell you what to do next.

Gradient's job isn't to visualize token counts. It's to answer: *Are we using the right models for the right tasks? Where are we overpaying? What's the annualized exposure?*

A summarization task running on GPT-4o is a procurement problem, not an engineering metric. Gradient surfaces that.

---

## Local setup

**Requirements:** Python 3.12, `venv`

```bash
git clone <repo>
cd gradient
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run tests

```bash
.venv/bin/python -m pytest -q
```

All tests should pass. Test coverage includes:

- CSV validation (missing columns, bad cost, bad timestamp, edge cases)
- Task classifier (all 6 task types + unknown)
- Spend analyzer (totals, annualization, breakdowns)
- Report builder (fields, savings math, opportunity annualization)
- API endpoints via FastAPI TestClient (upload, errors, report retrieval)

---

## Start the API

```bash
.venv/bin/uvicorn app.main:app --reload
```

API runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Upload the sample CSV

The repo includes 41 realistic usage rows covering summarization, classification, extraction, research, coding, and customer support — across GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and their cheaper counterparts.

**Step 1 — Upload the CSV:**

```bash
curl -X POST http://localhost:8000/audits/upload \
  -F "file=@data/sample_usage.csv" | jq .
```

Response:

```json
{
  "audit_run_id": "3f2a1b...",
  "record_count": 41,
  "status": "complete",
  "validation_summary": {
    "total_rows": 41,
    "valid_rows": 41,
    "invalid_rows": 0,
    "error_samples": []
  }
}
```

**Step 2 — Generate the audit report:**

```bash
curl -X POST http://localhost:8000/audits/{audit_run_id}/generate | jq .
```

Response:

```json
{
  "audit_run_id": "3f2a1b...",
  "report_id": "9e1c4d...",
  "status": "complete"
}
```

Calling generate again regenerates and replaces the stored report for that audit run. The new report gets a new `report_id`.

**Step 3 — Retrieve the JSON report:**

```bash
curl http://localhost:8000/audits/{audit_run_id}/report | jq .
```

**Step 4 — Retrieve the Markdown report (canonical executive memo):**

```bash
curl http://localhost:8000/audits/{audit_run_id}/report/markdown
```

The Markdown report is the primary human-readable output. It contains the executive summary, ranked opportunities, spend concentration tables, risk notes, and recommended next actions in a format ready for a board deck or procurement review.

The JSON report (Step 3) exposes the same data as structured fields. Its top-level fields include:

| Field | Description |
|---|---|
| `potential_annual_savings` | Estimated annual savings if opportunities are adopted |
| `savings_rate` | Savings as a fraction of annualized spend (0.0–1.0) |
| `top_opportunities` | Ranked list of model-switch recommendations |
| `recommended_next_actions` | Procurement-oriented action items |
| `spend_summary.top_cost_driving_task_types` | Task types sorted by observed spend, descending |

`estimated_annual_savings` on each opportunity is annualized from the observed period — not raw period spend.

Example key numbers from the sample CSV:

```
Current Annual Spend:   $5.81
Potential Savings:      $2.47
Confidence:             73%
```

*(Numbers are small because the sample data covers 41 rows. Real enterprise data produces real numbers.)*

---

## CSV format

| Column | Required | Description |
|---|---|---|
| `prompt` | yes | The input sent to the model |
| `response` | yes | The model's output |
| `timestamp` | yes | ISO 8601 or common datetime string |
| `model` | yes | Model identifier (e.g. `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `cost` | yes | Cost in USD for this request |
| `feedback` | no | Optional signal (`positive`, `negative`, etc.) |

**Validation is fail-fast.** If any row fails validation (bad cost, unparseable timestamp, empty model), the entire upload is rejected with a 422. The `validation_summary.invalid_rows` and `error_samples` fields in the upload response are reserved for a future partial-ingestion mode and will always be `0` / `[]` in Phase 1.

---

## Current limitations

- **No authentication.** Every upload is open. Don't expose this publicly yet.
- **Reports are persisted as JSON in SQLite.** Calling `POST /generate` regenerates and replaces the stored report for that audit run. Report history and versioning are not implemented.
- **Task classification is keyword-based.** It covers the common cases well but won't be perfect on unusual prompts.
- **Phase 1 savings estimates are conservative and model-tier-based**, not derived from actual API pricing. They're a procurement signal, not a billing forecast. Phase 2.5 replay simulation uses real pricing via provider APIs.
- **Real provider APIs are supported but beta.** OpenAI, Anthropic, and Gemini connectors are wired. Replay calls cost real money and require valid API credentials. The default evaluator mode (`heuristic`) requires no external calls.
- **Metadata-only audits support spend and task analysis but not replay.** Records uploaded without captured prompt and response cannot be replayed. Use `capture_prompt=True` and `capture_response=True` on the `@audit` decorator to enable replay for a workflow.
- **SQLite only.** Fine for development and pilots. Not for multi-tenant production.

---

## Optional: Streamlit UI

The core product path is the API (upload → generate → retrieve). The Streamlit app is an optional local demo, not part of the core backend.

```bash
.venv/bin/streamlit run streamlit_app.py
```

It uses the audit engine directly (no separate API server needed). Upload a CSV, generate a report, view findings, and download the Markdown report. Useful for demos; not required for production use.

---

## Phase 2.5: Replay Engine + SDK + Evidence Model

Phase 1 savings estimates are heuristic: conservative rates derived from model tiers, not actual output quality data.

Phase 2.5 replaces those estimates with evidence. It runs a sample of your historical prompts against cheaper candidate models, compares output quality, and builds a structured evidence tier per migration scenario — so recommendations are backed by data, not assumptions.

**What's in Phase 2.5:**
- **Real provider replay** — OpenAI, Anthropic, and Gemini connectors send prompts to live APIs and return actual model responses. Requires credentials; incurs real cost.
- **Evidence levels** — each migration scenario is rated: `heuristic → estimated → observed_replay → llm_judge → human_reviewed → production_validated`. Evidence level gates how strongly a migration is recommended.
- **Coverage-aware promotion** — a scenario only claims `human_reviewed` evidence if ≥80% of its replay results were human-reviewed. Partial coverage gets a small confidence boost but does not promote the label.
- **SDK instrumentation** — the `@audit` decorator captures AI workflow metadata (and optionally prompt/response) without breaking production. Metadata-only mode requires no prompt capture.
- **Human review loop** — export a CSV of replay results, fill in `reviewer_label`, and re-import. Re-simulate to incorporate human evidence into migration recommendations.
- **Local JSONL → Phase 1 CSV export** — metadata-only events can be exported for Phase 1 spend analysis without any prompt/response content leaving the process.

---

## Phase 2.5: Replay Scaffold

Phase 2 is fully functional locally — no real provider API calls required. The components below work together as a deterministic simulation layer.

### Key components

**`FakeProvider`** (`app/providers/fake.py`)
Implements the `GenerationProvider` interface without any external API calls. Uses MD5 hashing of `(prompt, model)` to return deterministic responses across runs. Pricing and latency are scaled by model tier from the catalog.

**`HeuristicEvaluator`** (`app/evaluation/heuristic_evaluator.py`)
The default evaluator — no LLM calls, no external APIs. Uses task-specific heuristics for all seven task types (summarization, classification, extraction, research, coding, customer support, other). Returns a `QualityEvaluation` with four fields: `score` (0.0–1.0), `explanation` (human-readable rationale), `confidence` (how reliable the score is for this task type), and `flags` (machine-readable signals such as `empty_response`, `research_conservative`, `missing_code_structure`). Research and coding scores are conservatively capped.

Select via `EVALUATOR_MODE=heuristic` (default). Use `get_evaluator(mode)` from `app/evaluation/factory.py` for programmatic selection.

**LLM judge evaluator** (`app/evaluation/llm_judge_evaluator.py`)
Uses a secondary LLM call to score candidate responses. Requires valid API credentials for the judge model. Select via `evaluator_mode=llm_judge`. When the judge call fails (network error, missing credentials), the evaluator falls back to heuristic scoring — **fallback results are recorded as `observed_replay` evidence, not `llm_judge`**, so the evidence level in the report accurately reflects what actually scored each result.

**Prometheus evaluator** (`app/evaluation/prometheus_evaluator.py`)
Placeholder — raises `NotImplementedError`. Reserved for a future Prometheus-compatible endpoint. The API rejects `evaluator_mode=prometheus` with 422.

**`ReplayRunner`** (`app/replay/replay_runner.py`)
Accepts any `GenerationProvider` and `BaseEvaluator`. Runs every `ReplayRequest` against every enabled `ReplayCandidate`. Errors are captured per result and never abort the full run.

Self-model candidates are always skipped: a candidate whose model name matches the source record's model produces no result for that record. If **all** selected candidates match the source model(s), the endpoint returns 422 and nothing is persisted.

Two construction helpers:
- `build_replay_requests(records)` — builds from in-memory `UsageRecord` objects; assigns synthetic UUIDs (no DB traceability)
- `build_replay_requests_from_rows(rows)` — builds from SQLite `usage_records` row dicts; uses the DB `id` as `original_record_id` for full traceability

Both builders carry the original `feedback` signal (positive/negative/etc.) into `ReplayRequest.feedback`. The runner passes it to the evaluator, where it is available as context for future quality scoring improvements.

**`MigrationSimulator`** (`app/replay/migration_simulator.py`)
Two modes:
- `simulate_migration(records, scenario)` — Phase 1 heuristic using catalog pricing ratios.
- `simulate_from_replay_data(original_records, replay_results)` — **Phase 2 evidence-based**, using actual replay quality scores and costs. Groups results by `(source_model, target_model, task_type)` and applies conservative migration rules:
  - quality loss < 2% and savings > 20% → `migrate`
  - quality loss 2–5% and savings > 30% → `controlled_pilot`
  - quality loss > 5% → `no_migration`
  - research/coding: tighter thresholds; classification/summarization: looser
  - Confidence scored from: record count, evaluator confidence, failure rate, quality score variance, task risk level

**`ExecutiveReplayReport`** (`app/replay/replay_report.py`)
The canonical Phase 2 output. Gradient's answer to: *"If we migrated this workload, what would actually happen?"* Combines all migration simulations into a structured report with `recommended_migrations`, `do_not_migrate`, `risk_notes`, and `next_actions`. Available as JSON or Markdown via the API. **This is the primary report endpoint for Phase 2** — not the per-model `ReplayReport` / `ModelReplaySummary` objects, which are internal to the simulation layer.

### API endpoints (Phase 2)

| Endpoint | Description |
|---|---|
| `POST /audits/{audit_run_id}/replay/run` | Run replay against candidate models |
| `POST /replay/{replay_run_id}/simulate` | Run evidence-based migration simulation |
| `GET /replay/{replay_run_id}/report` | Executive replay report — JSON. **Requires `/simulate` first.** |
| `GET /replay/{replay_run_id}/report/markdown` | Executive replay report — Markdown. **Requires `/simulate` first.** |

Calling `/report` or `/report/markdown` before `/simulate` returns 422.

**Request body for `replay/run`:**
```json
{
  "candidate_models": ["gpt-4o-mini", "claude-3-haiku-20240307"],
  "task_types": ["summarization", "classification"],
  "max_records": 100,
  "evaluator_mode": "heuristic"
}
```
All fields are optional. Omitting `candidate_models` runs all enabled candidates. Omitting `task_types` runs all records. `max_records` must be ≥ 1 if provided. `evaluator_mode` accepts `"heuristic"`, `"exact_match"`, `"task_router"`, and `"llm_judge"`. `"prometheus"` returns 422.

**Required call sequence (with optional human review loop):**
```
POST /audits/{id}/replay/run        →  replay_run_id
POST /replay/{id}/simulate          →  simulations_count, top_scenarios
GET  /replay/{id}/report            →  ExecutiveReplayReport (JSON)
GET  /replay/{id}/report/markdown   →  Markdown memo

# Optional: upgrade evidence via human review
GET  /replay/{id}/review/export     →  review CSV (download)
     [reviewer fills in reviewer_label column]
POST /replay/{id}/review/import     →  reviews_applied count
POST /replay/{id}/simulate          →  re-simulate with human-reviewed evidence
GET  /replay/{id}/report            →  updated report (human_reviewed evidence level)
```

Calling `/report` after importing reviews but before re-running `/simulate` returns 409 — the report would be stale. Re-run `/simulate` to incorporate the new evidence.

`/simulate` is idempotent: calling it again replaces prior simulation rows atomically (delete + insert in a single transaction). `/report` always reflects the most recent `/simulate` output.

### Run a local replay

```python
from datetime import datetime
from app.database import get_usage_records
from app.providers.fake import FakeProvider
from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.replay.replay_runner import ReplayRunner, build_replay_requests_from_rows
from app.replay.replay_models import REPLAY_CANDIDATES
from app.replay.migration_simulator import simulate_from_replay_data
from app.utils.date_range import calculate_date_range_days

# usage_rows comes from get_usage_records(audit_run_id) — dicts with DB-style "id" fields
usage_rows = get_usage_records(audit_run_id)

runner = ReplayRunner(provider=FakeProvider(), evaluator=HeuristicEvaluator())
requests = build_replay_requests_from_rows(usage_rows)
results = runner.run(requests, REPLAY_CANDIDATES)

timestamps = [datetime.fromisoformat(r["timestamp"]) for r in usage_rows if r.get("timestamp")]
date_range_days = calculate_date_range_days(timestamps)

simulations = simulate_from_replay_data(usage_rows, [r.model_dump() for r in results], date_range_days)

for s in simulations:
    print(f"{s.scenario_name}: savings=${s.estimated_annual_savings:.0f}/yr  confidence={s.confidence_score:.0%}  → {s.recommendation}")
```

`usage_rows` must be the dicts returned by `get_usage_records()` (they carry the DB `id` field). Passing `UsageRecord.model_dump()` output directly will silently produce no simulation results because the dicts have no `id` and the simulator cannot link replay results back to source records.

Swap `FakeProvider` for a real provider implementation when you have API credentials. The rest of the pipeline is unchanged.

---

## Start with metadata-only audit in 10 minutes

Most teams don't want to send raw prompts and responses to a third-party tool on day one. The Gradient SDK gives you a low-friction path: start with metadata only, then add content capture when you're ready.

### Install

The SDK is in the `gradient_sdk/` package. No extra dependencies — only `pydantic`, which is already required by the backend.

### Step 1 — Instrument one workflow

```python
from gradient_sdk import GradientClient, audit

client = GradientClient(mode="local", local_path="audit.jsonl")

@audit(
    client=client,
    workflow="support_ticket_reply",
    task_type="customer_support",
    team="support",
    risk_level="medium",
    # Defaults: capture_prompt=False, capture_response=False
)
def generate_reply(ticket: str) -> dict:
    # your existing LLM call here
    return {"text": "...", "model": "gpt-4o", "provider": "openai",
            "input_tokens": 120, "output_tokens": 45, "estimated_cost": 0.00127}
```

No prompts or responses leave the process. The decorator captures:
- workflow name, team, task type, risk level
- model, provider, latency, token counts, estimated cost
- success/error status

### Step 2 — Run for one week

Events accumulate in `audit.jsonl` on disk. No server required.

### Step 3 — Export and upload

```python
from gradient_sdk.exporters import load_jsonl, export_phase1_csv

events = load_jsonl("audit.jsonl")
export_phase1_csv(events, "upload.csv")
# Upload upload.csv to Gradient → runs the Phase 1 audit engine
```

> **Limitation:** metadata-only exports produce Phase 1 spend and task-type analysis, but **cannot be replayed**. The replay engine needs actual prompt and response content. When you're ready for replay, add `capture_prompt=True, capture_response=True` to the `@audit` decorator and upload a new batch.

Or send events directly to the Gradient API (cloud mode):

```python
client = GradientClient(api_key="...", mode="cloud", api_url="https://your-gradient-instance")
```

Events are batched and sent to `POST /sdk/events`.

### Customer adoption path

| Step | Action | Time |
|------|--------|------|
| 1 | Install SDK, add `@audit` to 1–2 AI workflows | 10 min |
| 2 | Run metadata-only for one week | 1 week |
| 3 | Export and run procurement audit | 30 min |
| 4 | Identify low-risk workloads for replay | 1 hour |
| 5 | Validate migration with replay engine | 1–2 days |
| 6 | Run controlled pilot on 10–20% of traffic | 2 weeks |
| 7 | Full migration with production evidence | ongoing |

### SDK modes

| Mode | Behavior |
|------|----------|
| `local` | Appends events to a local JSONL file |
| `cloud` | Batches and sends events to the Gradient API |
| `dry_run` | Logs what would be captured (no writes) |
| `disabled` | No-op — zero overhead |

### Adding content capture (opt-in)

Once your team is comfortable with metadata-only, opt in to content capture per workflow:

```python
@audit(
    client=client,
    workflow="summarizer",
    task_type="summarization",
    capture_prompt=True,    # opt-in
    capture_response=True,  # opt-in
    redact=True,            # redact PII before storage (default)
)
def summarize(text: str) -> str:
    ...
```

Redaction removes emails, phone numbers, API keys, and long numeric strings automatically. See `gradient_sdk/redaction.py` for the full list and how to add custom patterns.

### Feedback signals

```python
client.log_feedback(
    event_id="...",
    feedback_score=1.0,
    feedback_label="accepted",
    notes="User accepted the AI response",
)
```

Feedback is stored alongside audit events and feeds future quality and ROI analysis.

### Examples

See `examples/` for runnable demos:

- `examples/sdk_metadata_only.py` — recommended starting point
- `examples/sdk_basic_usage.py` — explicit provider/model, dry_run mode
- `examples/sdk_capture_with_redaction.py` — prompt/response capture with PII redaction
- `examples/sdk_local_export.py` — load JSONL, export to Phase 1 CSV
