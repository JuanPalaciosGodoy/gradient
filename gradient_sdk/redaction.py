"""Simple regex-based redaction for PII and secrets in captured text.

These patterns catch obvious cases. For production PII detection, integrate a
dedicated library (presidio, pii-codex, etc.) — see TODO below.
"""
import re
from typing import Optional

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
_LONG_NUMBER_RE = re.compile(r"\b\d{13,19}\b")
_SECRET_RE = re.compile(
    r"\b(sk|pk|api[-_]?key|secret|token|bearer)[-_]?[A-Za-z0-9]{16,}\b",
    re.IGNORECASE,
)

_EMAIL_PLACEHOLDER = "[EMAIL]"
_PHONE_PLACEHOLDER = "[PHONE]"
_NUMBER_PLACEHOLDER = "[NUMBER]"
_SECRET_PLACEHOLDER = "[SECRET]"

# TODO: replace with presidio or equivalent for production PII detection


def redact(
    text: str,
    custom_patterns: Optional[list[tuple[re.Pattern, str]]] = None,
) -> str:
    """Apply all built-in redaction rules, then any custom patterns."""
    text = _EMAIL_RE.sub(_EMAIL_PLACEHOLDER, text)
    text = _PHONE_RE.sub(_PHONE_PLACEHOLDER, text)
    text = _LONG_NUMBER_RE.sub(_NUMBER_PLACEHOLDER, text)
    text = _SECRET_RE.sub(_SECRET_PLACEHOLDER, text)
    if custom_patterns:
        for pattern, replacement in custom_patterns:
            text = pattern.sub(replacement, text)
    return text
