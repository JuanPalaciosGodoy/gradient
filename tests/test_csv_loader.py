import pytest

from app.ingestion.csv_loader import load_csv

VALID_CSV = b"""prompt,response,timestamp,model,cost
"Summarize this report","Summary here",2024-01-15 09:00:00,gpt-4o,0.0342
"Classify this ticket","Category: billing",2024-01-16 10:00:00,gpt-4o-mini,0.0012
"""


def test_valid_csv_loads_records():
    records = load_csv(VALID_CSV)
    assert len(records) == 2
    assert records[0].model == "gpt-4o"
    assert records[0].cost == 0.0342
    assert records[1].model == "gpt-4o-mini"


def test_missing_required_columns_raises():
    csv = b"prompt,response,timestamp\ncontent,reply,2024-01-15\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_csv(csv)


def test_missing_single_column_names_it():
    csv = b"prompt,response,timestamp,cost\nhello,world,2024-01-15 09:00:00,0.01\n"
    with pytest.raises(ValueError, match="model"):
        load_csv(csv)


def test_invalid_cost_raises():
    csv = b"prompt,response,timestamp,model,cost\nhello,world,2024-01-15 09:00:00,gpt-4o,not_a_number\n"
    with pytest.raises(ValueError):
        load_csv(csv)


def test_negative_cost_raises():
    csv = b"prompt,response,timestamp,model,cost\nhello,world,2024-01-15 09:00:00,gpt-4o,-0.01\n"
    with pytest.raises(ValueError):
        load_csv(csv)


def test_invalid_timestamp_raises():
    csv = b"prompt,response,timestamp,model,cost\nhello,world,not-a-date,gpt-4o,0.01\n"
    with pytest.raises(ValueError):
        load_csv(csv)


def test_optional_feedback_column():
    csv = b"prompt,response,timestamp,model,cost,feedback\nhello,world,2024-01-15 09:00:00,gpt-4o,0.01,positive\n"
    records = load_csv(csv)
    assert records[0].feedback == "positive"


def test_missing_feedback_is_none():
    records = load_csv(VALID_CSV)
    assert records[0].feedback is None


def test_column_names_are_case_insensitive():
    csv = b"Prompt,Response,Timestamp,Model,Cost\nhello,world,2024-01-15 09:00:00,gpt-4o,0.01\n"
    records = load_csv(csv)
    assert len(records) == 1


def test_empty_model_raises():
    csv = b"prompt,response,timestamp,model,cost\nhello,world,2024-01-15 09:00:00,  ,0.01\n"
    with pytest.raises(ValueError):
        load_csv(csv)


# ── Metadata-only sentinel ────────────────────────────────────────────────────

def test_empty_prompt_becomes_sentinel():
    """Blank prompt must not become the literal string 'nan'."""
    from app.ingestion.validators import METADATA_ONLY_SENTINEL
    csv = b"prompt,response,timestamp,model,cost\n,some response,2024-01-15 09:00:00,gpt-4o,0.01\n"
    records = load_csv(csv)
    assert records[0].prompt == METADATA_ONLY_SENTINEL


def test_empty_response_becomes_sentinel():
    from app.ingestion.validators import METADATA_ONLY_SENTINEL
    csv = b"prompt,response,timestamp,model,cost\nsome prompt,,2024-01-15 09:00:00,gpt-4o,0.01\n"
    records = load_csv(csv)
    assert records[0].response == METADATA_ONLY_SENTINEL


def test_both_empty_prompt_and_response_become_sentinel():
    from app.ingestion.validators import METADATA_ONLY_SENTINEL
    csv = b"prompt,response,timestamp,model,cost\n,,2024-01-15 09:00:00,gpt-4o,0.01\n"
    records = load_csv(csv)
    assert records[0].prompt == METADATA_ONLY_SENTINEL
    assert records[0].response == METADATA_ONLY_SENTINEL


def test_task_type_column_preserved_when_present():
    csv = b"prompt,response,timestamp,model,cost,task_type\n,, 2024-01-15 09:00:00,gpt-4o,0.01,summarization\n"
    records = load_csv(csv)
    from app.schemas import TaskType
    assert records[0].task_type == TaskType.SUMMARIZATION
