"""Tests for the @audit decorator."""
import pytest

from gradient_sdk import audit
from gradient_sdk.client import GradientClient
from gradient_sdk.models import AuditEvent


# ── Helpers ───────────────────────────────────────────────────────────────────

class CapturingClient:
    """Test double that collects captured events in memory."""
    def __init__(self):
        self.events: list[AuditEvent] = []
        self.mode = "local"

    def capture(self, event: AuditEvent) -> None:
        self.events.append(event)


# ── Return value passthrough ──────────────────────────────────────────────────

def test_decorator_does_not_alter_return_value():
    client = CapturingClient()

    @audit(client=client, workflow="test", task_type="summarization", provider="openai", model="gpt-4o-mini")
    def fn(text: str) -> str:
        return f"summary of: {text}"

    result = fn("hello world")
    assert result == "summary of: hello world"


def test_decorator_passes_through_dict_return():
    client = CapturingClient()

    @audit(client=client, workflow="test", task_type="classification", provider="openai", model="gpt-4o-mini")
    def fn() -> dict:
        return {"label": "billing", "confidence": 0.95}

    result = fn()
    assert result == {"label": "billing", "confidence": 0.95}


def test_decorator_passes_through_none_return():
    client = CapturingClient()

    @audit(client=client, workflow="test", task_type="other", provider="openai", model="gpt-4o")
    def fn():
        return None

    assert fn() is None


# ── Successful capture ────────────────────────────────────────────────────────

def test_decorator_captures_successful_call():
    client = CapturingClient()

    @audit(client=client, workflow="ticket_reply", task_type="customer_support",
           provider="openai", model="gpt-4o-mini", team="support")
    def fn(ticket: str) -> str:
        return "resolved"

    fn("my order is late")
    assert len(client.events) == 1
    event = client.events[0]
    assert event.workflow == "ticket_reply"
    assert event.task_type == "customer_support"
    assert event.team == "support"
    assert event.status == "success"
    assert event.provider == "openai"
    assert event.model == "gpt-4o-mini"
    assert event.latency_ms is not None
    assert event.latency_ms >= 0.0


def test_decorator_records_latency():
    import time
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m")
    def fn():
        time.sleep(0.01)
        return "ok"

    fn()
    assert client.events[0].latency_ms >= 10.0


def test_decorator_extracts_tokens_and_cost_from_dict_return():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t")
    def fn():
        return {
            "text": "result",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "input_tokens": 50,
            "output_tokens": 20,
            "estimated_cost": 0.00002,
        }

    fn()
    e = client.events[0]
    assert e.input_tokens == 50
    assert e.output_tokens == 20
    assert e.estimated_cost == pytest.approx(0.00002)
    assert e.model == "gpt-4o-mini"
    assert e.provider == "openai"


def test_decorator_extracts_tokens_from_openai_usage_shape():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="openai", model="gpt-4o")
    def fn():
        return {"usage": {"prompt_tokens": 100, "completion_tokens": 40}}

    fn()
    e = client.events[0]
    assert e.input_tokens == 100
    assert e.output_tokens == 40


def test_decorator_all_business_context_fields_propagated():
    client = CapturingClient()

    @audit(
        client=client,
        workflow="w",
        task_type="t",
        provider="p",
        model="m",
        business_unit="eng",
        process_name="daily_batch",
        tool_name="my_tool",
        environment="staging",
        risk_level="high",
        value_metric_name="tickets_closed",
    )
    def fn():
        return "ok"

    fn()
    e = client.events[0]
    assert e.business_unit == "eng"
    assert e.process_name == "daily_batch"
    assert e.tool_name == "my_tool"
    assert e.environment == "staging"
    assert e.risk_level == "high"
    assert e.value_metric_name == "tickets_closed"


# ── Failed calls ──────────────────────────────────────────────────────────────

def test_decorator_reraises_app_exception_by_default():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m")
    def fn():
        raise ValueError("API timeout")

    with pytest.raises(ValueError, match="API timeout"):
        fn()

    assert len(client.events) == 1
    e = client.events[0]
    assert e.status == "error"
    assert "API timeout" in e.error_message


def test_decorator_records_error_event_before_reraise():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m")
    def fn():
        raise RuntimeError("critical error")

    with pytest.raises(RuntimeError, match="critical error"):
        fn()

    assert client.events[0].status == "error"
    assert "critical error" in client.events[0].error_message


def test_suppress_application_exceptions_swallows_error():
    """Only when explicitly opted in should exceptions be suppressed."""
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m",
           suppress_application_exceptions=True)
    def fn():
        raise ValueError("boom")

    result = fn()  # must not raise
    assert result is None
    assert client.events[0].status == "error"


# ── SDK failure isolation ─────────────────────────────────────────────────────

def test_sdk_failure_does_not_break_production_function():
    """If client.capture() raises, the wrapped function's return value is still returned."""
    class BrokenClient:
        mode = "local"
        def capture(self, event):
            raise RuntimeError("client exploded")

    @audit(client=BrokenClient(), workflow="w", task_type="t", provider="p", model="m")
    def fn(x: int) -> int:
        return x * 2

    result = fn(21)
    assert result == 42


def test_sdk_failure_does_not_mask_app_exception():
    """When the app function raises, the original exception propagates even if SDK also fails."""
    class BrokenClient:
        mode = "local"
        def capture(self, event):
            raise RuntimeError("SDK exploded")

    @audit(client=BrokenClient(), workflow="w", task_type="t", provider="p", model="m")
    def fn():
        raise ValueError("app error")

    with pytest.raises(ValueError, match="app error"):
        fn()  # SDK failure must not replace or mask the ValueError


# ── Prompt and response capture ───────────────────────────────────────────────

def test_capture_prompt_false_does_not_store_prompt():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m",
           capture_prompt=False)
    def fn(prompt: str) -> str:
        return "answer"

    fn("sensitive prompt text")
    assert client.events[0].prompt is None


def test_capture_prompt_true_stores_prompt_by_name():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m",
           capture_prompt=True, redact=False)
    def fn(prompt: str) -> str:
        return "answer"

    fn("my prompt")
    assert client.events[0].prompt == "my prompt"


def test_capture_prompt_true_stores_first_positional_arg_as_fallback():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m",
           capture_prompt=True, redact=False)
    def fn(ticket: str) -> str:
        return "answer"

    fn("ticket content")
    assert client.events[0].prompt == "ticket content"


def test_capture_prompt_uses_named_prompt_arg():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m",
           capture_prompt=True, prompt_arg="ticket", redact=False)
    def fn(ticket: str) -> str:
        return "answer"

    fn("explicit ticket text")
    assert client.events[0].prompt == "explicit ticket text"


def test_capture_response_false_does_not_store_response():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m",
           capture_response=False)
    def fn() -> str:
        return "sensitive response"

    fn()
    assert client.events[0].response is None


def test_capture_response_true_stores_str_return():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m",
           capture_response=True, redact=False)
    def fn() -> str:
        return "the answer is 42"

    fn()
    assert client.events[0].response == "the answer is 42"


def test_capture_response_true_extracts_text_from_dict():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m",
           capture_response=True, redact=False)
    def fn() -> dict:
        return {"text": "extracted response", "model": "gpt-4o"}

    fn()
    assert client.events[0].response == "extracted response"


# ── Redaction applied during capture ─────────────────────────────────────────

def test_prompt_redaction_removes_email():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m",
           capture_prompt=True, redact=True)
    def fn(prompt: str) -> str:
        return "ok"

    fn("Contact me at alice@example.com about order 12345.")
    assert "alice@example.com" not in client.events[0].prompt
    assert "[EMAIL]" in client.events[0].prompt


# ── functools.wraps preservation ─────────────────────────────────────────────

def test_decorator_preserves_function_name():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m")
    def my_special_function():
        return "ok"

    assert my_special_function.__name__ == "my_special_function"


def test_decorator_preserves_docstring():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t", provider="p", model="m")
    def fn():
        """My docstring."""
        return "ok"

    assert fn.__doc__ == "My docstring."


# ── provider/model fallback to "unknown" ─────────────────────────────────────

def test_omitted_provider_and_model_default_to_unknown():
    """When provider/model are not specified and not returned, 'unknown' is used."""
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t")
    def fn() -> str:
        return "plain string response"

    fn()
    e = client.events[0]
    assert e.provider == "unknown"
    assert e.model == "unknown"


def test_provider_model_resolved_from_return_value():
    client = CapturingClient()

    @audit(client=client, workflow="w", task_type="t")
    def fn() -> dict:
        return {"text": "ok", "provider": "anthropic", "model": "claude-3-haiku-20240307"}

    fn()
    e = client.events[0]
    assert e.provider == "anthropic"
    assert e.model == "claude-3-haiku-20240307"


# ── AuditEvent identity validation ───────────────────────────────────────────

def test_audit_event_rejects_empty_provider():
    from pydantic import ValidationError
    from gradient_sdk.models import AuditEvent

    with pytest.raises(ValidationError, match="must not be blank"):
        AuditEvent(provider="", model="gpt-4o", workflow="w", task_type="t")


def test_audit_event_rejects_blank_workflow():
    from pydantic import ValidationError
    from gradient_sdk.models import AuditEvent

    with pytest.raises(ValidationError, match="must not be blank"):
        AuditEvent(provider="openai", model="gpt-4o", workflow="  ", task_type="t")


def test_audit_event_rejects_empty_task_type():
    from pydantic import ValidationError
    from gradient_sdk.models import AuditEvent

    with pytest.raises(ValidationError, match="must not be blank"):
        AuditEvent(provider="openai", model="gpt-4o", workflow="w", task_type="")
