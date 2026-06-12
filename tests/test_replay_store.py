"""Tests for replace_migration_simulations atomicity."""
import sqlite3

import pytest

from app.database import get_migration_simulations_for_replay_run
from app.replay.replay_models import MigrationSimulationResult
from app.replay.replay_store import replace_migration_simulations, save_migration_simulation


def _sim(scenario_name: str = "gpt-4o→gpt-4o-mini:summarization") -> MigrationSimulationResult:
    return MigrationSimulationResult(
        scenario_name=scenario_name,
        source_model="gpt-4o",
        target_model="gpt-4o-mini",
        current_annualized_cost=100.0,
        simulated_annualized_cost=50.0,
        estimated_annual_savings=50.0,
        estimated_savings_pct=50.0,
        average_quality_delta=-0.01,
        confidence_score=0.75,
        recommendation="migrate",
        rationale="test",
    )


def test_replace_simulations_replaces_prior_rows(db):
    save_migration_simulation(_sim("old"), replay_run_id="run-A")
    assert len(get_migration_simulations_for_replay_run("run-A")) == 1

    replace_migration_simulations("run-A", "audit-A", [_sim("new")])

    rows = get_migration_simulations_for_replay_run("run-A")
    assert len(rows) == 1
    assert rows[0]["scenario_name"] == "new"


def test_replace_simulations_idempotent_no_duplicates(db):
    replace_migration_simulations("run-B", "audit-B", [_sim()])
    count_1 = len(get_migration_simulations_for_replay_run("run-B"))

    replace_migration_simulations("run-B", "audit-B", [_sim()])
    count_2 = len(get_migration_simulations_for_replay_run("run-B"))

    assert count_1 == count_2 == 1


def test_replace_simulations_empty_clears_prior_rows(db):
    save_migration_simulation(_sim(), replay_run_id="run-C")
    assert len(get_migration_simulations_for_replay_run("run-C")) == 1

    replace_migration_simulations("run-C", "audit-C", [])
    assert len(get_migration_simulations_for_replay_run("run-C")) == 0


def test_replace_simulations_rollback_preserves_old_rows_on_failure(db, monkeypatch):
    """If the INSERT phase raises, the DELETE must be rolled back atomically."""
    from contextlib import contextmanager
    from app.config import settings
    import app.replay.replay_store as store_module

    save_migration_simulation(_sim("original"), replay_run_id="run-D")
    assert len(get_migration_simulations_for_replay_run("run-D")) == 1

    # Patch get_connection in replay_store's namespace with a wrapper whose
    # executemany always raises, while execute (used for DELETE) works normally.
    @contextmanager
    def failing_get_connection():
        real = sqlite3.connect(settings.database_path)
        real.row_factory = sqlite3.Row

        class BoomProxy:
            def execute(self, sql, params=None):
                return real.execute(sql, params) if params is not None else real.execute(sql)

            def executemany(self, sql, params):
                raise sqlite3.OperationalError("forced insert failure")

        try:
            yield BoomProxy()
            real.commit()
        except Exception:
            real.rollback()
            raise
        finally:
            real.close()

    monkeypatch.setattr(store_module, "get_connection", failing_get_connection)

    with pytest.raises(Exception):
        replace_migration_simulations("run-D", "audit-D", [_sim("replacement")])

    rows = get_migration_simulations_for_replay_run("run-D")
    assert len(rows) == 1, "Rollback should have restored the original row"
    assert rows[0]["scenario_name"] == "original"


def test_replace_simulations_multiple_rows_inserted(db):
    sims = [_sim(f"gpt-4o→gpt-4o-mini:{t}") for t in ("summarization", "classification", "coding")]
    replace_migration_simulations("run-E", "audit-E", sims)
    rows = get_migration_simulations_for_replay_run("run-E")
    assert len(rows) == 3
    names = {row["scenario_name"] for row in rows}
    assert names == {s.scenario_name for s in sims}
