"""@audit decorator for instrumenting AI workflow functions."""
import functools
import inspect
import logging
import time
from typing import Any, Callable, Optional

from gradient_sdk.models import AuditEvent

logger = logging.getLogger(__name__)


def audit(
    client,
    workflow: str,
    task_type: str,
    provider: str = "",
    model: str = "",
    team: Optional[str] = None,
    business_unit: Optional[str] = None,
    process_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    environment: Optional[str] = None,
    risk_level: Optional[str] = None,
    value_metric_name: Optional[str] = None,
    capture_prompt: bool = False,
    capture_response: bool = False,
    prompt_arg: Optional[str] = None,
    redact: bool = True,
    custom_redact_patterns=None,
    suppress_application_exceptions: bool = False,
):
    """Decorator that wraps an AI workflow function with Gradient audit logging.

    Args:
        client:                          GradientClient instance.
        workflow:                        Name identifying this workflow (e.g. "ticket_reply").
        task_type:                       Task category (e.g. "summarization", "classification").
        provider:                        LLM provider (e.g. "openai"). Can be omitted if the
                                         function return value contains a "provider" key.
                                         Defaults to "unknown" when unresolvable.
        model:                           Model name (e.g. "gpt-4o-mini"). Can be omitted if
                                         returned by the function. Defaults to "unknown".
        capture_prompt:                  If True, capture the prompt argument (default False).
        capture_response:                If True, capture the return value as the response (default False).
        prompt_arg:                      Name of the function argument that holds the prompt.
                                         When None and capture_prompt=True, the decorator tries
                                         common names ("prompt", "message", "text") and falls back
                                         to the first positional argument.
        redact:                          Apply PII/secret redaction before storing (default True).
        custom_redact_patterns:          Additional (re.Pattern, replacement) pairs to apply.
        suppress_application_exceptions: If True, swallow exceptions raised by the wrapped function
                                         and return None instead. Default False — application
                                         exceptions always propagate. SDK logging failures are
                                         always silenced regardless of this setting.

    Usage::

        from gradient_sdk import audit, GradientClient

        client = GradientClient(api_key="...", mode="local")

        @audit(
            client=client,
            workflow="support_ticket_reply",
            task_type="customer_support",
            team="support",
            risk_level="medium",
        )
        def generate_reply(ticket: str) -> str:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            status = "success"
            error_msg = None
            result = None
            exc_to_reraise = None

            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                status = "error"
                error_msg = str(exc)
                if not suppress_application_exceptions:
                    exc_to_reraise = exc
            finally:
                latency_ms = (time.monotonic() - start) * 1000
                try:
                    _dispatch_event(
                        client=client,
                        fn=fn,
                        args=args,
                        kwargs=kwargs,
                        result=result,
                        status=status,
                        error_msg=error_msg,
                        latency_ms=latency_ms,
                        workflow=workflow,
                        task_type=task_type,
                        provider=provider,
                        model=model,
                        team=team,
                        business_unit=business_unit,
                        process_name=process_name,
                        tool_name=tool_name,
                        environment=environment,
                        risk_level=risk_level,
                        value_metric_name=value_metric_name,
                        capture_prompt=capture_prompt,
                        capture_response=capture_response,
                        prompt_arg=prompt_arg,
                        redact=redact,
                        custom_redact_patterns=custom_redact_patterns,
                    )
                except Exception:
                    logger.debug("[gradient_sdk] event dispatch failed (non-fatal)", exc_info=True)

            if exc_to_reraise is not None:
                raise exc_to_reraise
            return result

        return wrapper
    return decorator


def _dispatch_event(
    client,
    fn,
    args,
    kwargs,
    result,
    status,
    error_msg,
    latency_ms,
    workflow,
    task_type,
    provider,
    model,
    team,
    business_unit,
    process_name,
    tool_name,
    environment,
    risk_level,
    value_metric_name,
    capture_prompt,
    capture_response,
    prompt_arg,
    redact,
    custom_redact_patterns,
) -> None:
    meta = _extract_metadata(result) if result is not None else {}

    resolved_model = model or meta.get("model") or "unknown"
    resolved_provider = provider or meta.get("provider") or "unknown"

    prompt_text: Optional[str] = None
    if capture_prompt:
        prompt_text = _resolve_prompt(fn, args, kwargs, prompt_arg)
        if redact and prompt_text:
            from gradient_sdk.redaction import redact as do_redact
            prompt_text = do_redact(prompt_text, custom_redact_patterns)

    response_text: Optional[str] = None
    if capture_response and result is not None:
        response_text = _extract_response(result)
        if redact and response_text:
            from gradient_sdk.redaction import redact as do_redact
            response_text = do_redact(response_text, custom_redact_patterns)

    event = AuditEvent(
        provider=resolved_provider,
        model=resolved_model,
        workflow=workflow,
        task_type=task_type,
        status=status,
        latency_ms=round(latency_ms, 2),
        input_tokens=meta.get("input_tokens"),
        output_tokens=meta.get("output_tokens"),
        estimated_cost=meta.get("estimated_cost"),
        error_message=error_msg,
        team=team,
        business_unit=business_unit,
        process_name=process_name,
        tool_name=tool_name,
        environment=environment,
        risk_level=risk_level,
        value_metric_name=value_metric_name,
        prompt=prompt_text,
        response=response_text,
    )
    client.capture(event)


def _extract_metadata(result: Any) -> dict:
    """Best-effort extraction of token/cost/model data from a function return value."""
    if not isinstance(result, dict):
        return {}
    usage = result.get("usage") or {}
    return {
        "input_tokens": (
            result.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("input_tokens")
        ),
        "output_tokens": (
            result.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("output_tokens")
        ),
        "estimated_cost": result.get("estimated_cost") or result.get("cost"),
        "model": result.get("model"),
        "provider": result.get("provider"),
    }


def _extract_response(result: Any) -> Optional[str]:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result.get("response") or result.get("text") or result.get("content")
    try:
        return str(result)
    except Exception:
        return None


def _resolve_prompt(fn, args, kwargs, prompt_arg: Optional[str]) -> Optional[str]:
    """Extract the prompt string from the wrapped function's arguments."""
    params = list(inspect.signature(fn).parameters.keys())

    if prompt_arg:
        if prompt_arg in kwargs:
            return str(kwargs[prompt_arg])
        if prompt_arg in params:
            idx = params.index(prompt_arg)
            if idx < len(args):
                return str(args[idx])
        return None

    for name in ("prompt", "message", "text", "input", "query"):
        if name in kwargs:
            return str(kwargs[name])
        if name in params:
            idx = params.index(name)
            if idx < len(args):
                return str(args[idx])

    if args:
        return str(args[0])
    return None
