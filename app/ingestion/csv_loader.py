import io

import pandas as pd

from app.ingestion.validators import validate_columns, validate_row
from app.schemas import UsageRecord

REQUIRED_COLUMNS = {"prompt", "response", "timestamp", "model", "cost"}


def load_csv(content: bytes) -> list[UsageRecord]:
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}") from e

    df.columns = df.columns.str.strip().str.lower()
    validate_columns(set(df.columns), REQUIRED_COLUMNS)

    records = []
    errors = []

    for idx, row in df.iterrows():
        try:
            records.append(validate_row(row, idx))
        except ValueError as e:
            errors.append(f"Row {idx + 2}: {e}")

    if errors:
        raise ValueError("CSV validation errors:\n" + "\n".join(errors[:10]))

    return records
