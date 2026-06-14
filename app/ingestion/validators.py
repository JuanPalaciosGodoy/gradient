import pandas as pd
from pydantic import ValidationError

from app.schemas import TaskType, UsageRecord

METADATA_ONLY_SENTINEL = "[metadata_only]"


def validate_columns(present: set[str], required: set[str]) -> None:
    missing = required - present
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def _clean_text_field(val) -> str:
    """Return val as a string; replace NaN/empty/None with the metadata-only sentinel.

    This prevents pandas NaN values from becoming the literal string "nan" when
    prompt or response columns are blank (e.g. metadata-only SDK exports).
    """
    if val is None:
        return METADATA_ONLY_SENTINEL
    if isinstance(val, float) and pd.isna(val):
        return METADATA_ONLY_SENTINEL
    s = str(val).strip()
    return s if s else METADATA_ONLY_SENTINEL


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
            prompt=_clean_text_field(row["prompt"]),
            response=_clean_text_field(row["response"]),
            timestamp=dt,
            model=str(row["model"]).strip(),
            cost=cost,
            feedback=feedback,
            task_type=task_type,
        )
    except ValidationError as e:
        raise ValueError(str(e)) from e
