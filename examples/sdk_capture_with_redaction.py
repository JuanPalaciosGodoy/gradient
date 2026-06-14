"""
Gradient SDK — prompt/response capture with redaction enabled.

Useful after a team has validated that their data is safe to capture.
Redaction removes emails, phone numbers, secrets, and long numeric strings
before they leave the process.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_sdk import GradientClient, audit

client = GradientClient(mode="local", local_path="audit_with_content.jsonl")


@audit(
    client=client,
    workflow="support_ticket_reply",
    task_type="customer_support",
    team="support",
    risk_level="medium",
    capture_prompt=True,    # opt-in to capturing the prompt
    capture_response=True,  # opt-in to capturing the response
    prompt_arg="ticket",    # name of the argument that holds the prompt text
    redact=True,            # redact PII before storing (default)
)
def generate_reply(ticket: str) -> dict:
    return {
        "text": "We've located your order. It will arrive by Thursday.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input_tokens": 140,
        "output_tokens": 22,
        "estimated_cost": 0.000035,
    }


if __name__ == "__main__":
    # Email and phone in the ticket will be redacted before storage
    reply = generate_reply(
        ticket="Hi, I'm john.doe@example.com and my phone is 555-867-5309. "
               "Order #4829384 hasn't arrived."
    )
    print("Reply:", reply["text"])
    print("Prompt and response written to audit_with_content.jsonl (PII redacted).")
