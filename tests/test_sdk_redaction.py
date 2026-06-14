"""Tests for gradient_sdk.redaction."""
import re

import pytest

from gradient_sdk.redaction import redact


def test_redact_email():
    result = redact("Contact alice@example.com for details.")
    assert "alice@example.com" not in result
    assert "[EMAIL]" in result


def test_redact_multiple_emails():
    result = redact("From bob@foo.org to carol@bar.net")
    assert "bob@foo.org" not in result
    assert "carol@bar.net" not in result
    assert result.count("[EMAIL]") == 2


def test_redact_phone_us_format():
    result = redact("Call me at 555-867-5309 anytime.")
    assert "867-5309" not in result
    assert "[PHONE]" in result


def test_redact_phone_with_area_code_parens():
    result = redact("Reach us at (800) 555-1234.")
    assert "555-1234" not in result
    assert "[PHONE]" in result


def test_redact_long_number_credit_card_like():
    result = redact("Card ending in 4242424242424242 was charged.")
    assert "4242424242424242" not in result
    assert "[NUMBER]" in result


def test_redact_short_number_not_redacted():
    result = redact("Order #12345 is confirmed.")
    assert "12345" in result  # too short to be a card number


def test_redact_api_key_pattern():
    result = redact("Bearer sk-AbCdEfGhIjKlMnOpQrStUvWx123456")
    assert "sk-AbCdEfGhIjKlMnOpQrStUvWx123456" not in result
    assert "[SECRET]" in result


def test_redact_no_pii_unchanged():
    text = "The forecast for Q3 is strong growth of 12%."
    assert redact(text) == text


def test_redact_custom_pattern():
    pattern = (re.compile(r"\bACCOUNT-\d+\b"), "[ACCOUNT]")
    result = redact("Your reference is ACCOUNT-98765.", custom_patterns=[pattern])
    assert "ACCOUNT-98765" not in result
    assert "[ACCOUNT]" in result


def test_redact_empty_string():
    assert redact("") == ""


def test_redact_combined():
    text = "Email alice@corp.com or call 800-555-0100 using key sk-TestKey123456789012"
    result = redact(text)
    assert "alice@corp.com" not in result
    assert "800-555-0100" not in result
    assert "sk-TestKey123456789012" not in result
