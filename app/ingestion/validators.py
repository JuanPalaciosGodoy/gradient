import pandas as pd
from pydantic import ValidationError

from app.schemas import UsageRecord


def validate_columns(present: set[str], required: set[str]) -> None:
    missing = required - present
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def validate_row(row: pd.Series, idx: int) -> UsageRecord:
    try:
        cost = float(row["cost"])
    except (ValueError, TypeError):
        raise ValueError(f"invalid cost value: {row['cost']!r}")

    try:
        ts = pd.to_datetime(row["timestamp"])
        if pd.isna(ts):
            raise ValueError("null timestamp")
        dt = ts.to_pydatetime()
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        raise ValueError(f"invalid timestamp: {row['timestamp']!r}")

    feedback = None
    if "feedback" in row.index:
        raw = row["feedback"]
        if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
            feedback = str(raw)

    try:
        return UsageRecord(
            prompt=str(row["prompt"]),
            response=str(row["response"]),
            timestamp=dt,
            model=str(row["model"]).strip(),
            cost=cost,
            feedback=feedback,
        )
    except ValidationError as e:
        raise ValueError(str(e)) from e
