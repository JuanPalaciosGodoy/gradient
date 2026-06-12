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
- **Savings estimates are conservative and model-tier-based**, not derived from actual API pricing. They're a procurement signal, not a billing forecast.
- **No provider API connections yet.** CSV upload only. OpenAI, Anthropic, and Gemini connectors are stubbed.
- **SQLite only.** Fine for Phase 1. Not for multi-tenant production.

---

## Optional: Streamlit UI

The core product path is the API (upload → generate → retrieve). The Streamlit app is an optional local demo, not part of the core backend.

```bash
.venv/bin/streamlit run streamlit_app.py
```

It uses the audit engine directly (no separate API server needed). Upload a CSV, generate a report, view findings, and download the Markdown report. Useful for demos; not required for production use.

---

## Phase 2: Replay Engine

Phase 1 savings estimates are heuristic: conservative rates derived from model tiers, not actual output quality data.

Phase 2 replaces those estimates with evidence. It will run a sample of your historical prompts against cheaper candidate models and compare output quality. Instead of saying "summarization typically saves 45%," Gradient will *show you* whether `claude-3-5-sonnet` produces acceptable summaries for your specific workload — and quantify the quality tradeoff.

This is what makes it a procurement tool, not a token dashboard.

---

## Phase 2: Local Replay Scaffold

Phase 2 is fully functional locally — no real provider API calls required. The components below work together as a deterministic simulation layer.

### Key components

**`FakeProvider`** (`app/providers/fake.py`)
Implements the `GenerationProvider` interface without any external API calls. Uses MD5 hashing of `(prompt, model)` to return deterministic responses across runs. Pricing and latency are scaled by model tier from the catalog.

**`HeuristicEvaluator`** (`app/evaluation/heuristic_evaluator.py`)
The default evaluator — no LLM calls, no external APIs. Uses task-specific heuristics for all seven task types (summarization, classification, extraction, research, coding, customer support, other). Returns a `QualityEvaluation` with four fields: `score` (0.0–1.0), `explanation` (human-readable rationale), `confidence` (how reliable the score is for this task type), and `flags` (machine-readable signals such as `empty_response`, `research_conservative`, `missing_code_structure`). Research and coding scores are conservatively capped.

Select via `EVALUATOR_MODE=heuristic` (default). Use `get_evaluator(mode)` from `app/evaluation/factory.py` for programmatic selection.

**Prometheus / LLM judge evaluators** (`app/evaluation/prometheus_evaluator.py`, `app/evaluation/llm_judge_evaluator.py`)
Clean stubs — raise `NotImplementedError` until real credentials and endpoints are wired. Each file documents the expected input format and output parsing contract for Phase 3 integration.

**`ReplayRunner`** (`app/replay/replay_runner.py`)
Accepts any `GenerationProvider` and `BaseEvaluator`. Runs every `ReplayRequest` against every enabled `ReplayCandidate`. Errors are captured per result and never abort the full run.

Two construction helpers:
- `build_replay_requests(records)` — builds from in-memory `UsageRecord` objects; assigns synthetic UUIDs (no DB traceability)
- `build_replay_requests_from_rows(rows)` — builds from SQLite `usage_records` row dicts; uses the DB `id` as `original_record_id` for full traceability

Both builders carry the original `feedback` signal (positive/negative/etc.) into `ReplayRequest.feedback`. The runner passes it to the evaluator, where it is available as context for future quality scoring improvements.

**`ReplayReport` / `ModelReplaySummary`** (`app/replay/replay_report.py`)
Pydantic models summarizing outcomes per candidate model: quality score, latency, cost, `success_count`, `coverage_rate`, and `estimated_annual_savings`. Savings are computed only against the subset of requests that had successful results for each candidate — failed calls never inflate savings figures.

**`MigrationSimulator`** (`app/replay/migration_simulator.py`)
Estimates annual savings from a proposed model migration using catalog pricing ratios. Filters by `source_model` and `task_types_included`. Outputs a `MigrationSimulationResult` with `confidence_score` and `recommendation` (`proceed` / `hold` / `investigate`). No LLM calls; confidence is heuristic (record volume + catalog coverage).

### Run a local replay

```python
from app.providers.fake import FakeProvider
from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.replay.replay_runner import ReplayRunner, build_replay_requests
from app.replay.replay_report import build_replay_report
from app.replay.replay_models import REPLAY_CANDIDATES

runner = ReplayRunner(provider=FakeProvider(), evaluator=HeuristicEvaluator())
requests = build_replay_requests(usage_records)
results = runner.run(requests, REPLAY_CANDIDATES)
report = build_replay_report("run-001", requests, results, date_range_days=30)

for s in report.model_summaries:
    print(f"{s.model}: savings=${s.estimated_annual_savings:.0f}/yr  quality={s.avg_quality_score:.2f}  coverage={s.coverage_rate:.0%}")
```

Swap `FakeProvider` for a real provider implementation when you have API credentials. The rest of the pipeline is unchanged.
