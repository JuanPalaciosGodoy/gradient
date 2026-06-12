import pandas as pd
from pydantic import ValidationError

from app.schemas import TaskType, UsageRecord


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

    task_type = None
    if "task_type" in row.index:
        raw_tt = row["task_type"]
        if raw_tt is not None and not (isinstance(raw_tt, float) and pd.isna(raw_tt)):
            try:
                task_type = TaskType(str(raw_tt).lower().strip())
            except ValueError:
                pass  # unknown value; classifier will fill in during upload

    try:
        return UsageRecord(
            prompt=str(row["prompt"]),
            response=str(row["response"]),
            timestamp=dt,
            model=str(row["model"]).strip(),
            cost=cost,
            feedback=feedback,
            task_type=task_type,
        )
    except ValidationError as e:
        raise ValueError(str(e)) from e
