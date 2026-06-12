"""Shared date-range helper used by both Phase 1 and Phase 2."""
from datetime import datetime


def calculate_date_range_days(timestamps: list[datetime]) -> int:
    """Inclusive calendar-day count for an observation window.

    Jan 1 → Jan 1  (same day)  → 1
    Jan 1 → Jan 2  (adjacent)  → 2
    Jan 1 → Jan 31             → 31
    Empty or single timestamp  → 1
    """
    if len(timestamps) < 2:
        return 1
    return max((max(timestamps) - min(timestamps)).days + 1, 1)
