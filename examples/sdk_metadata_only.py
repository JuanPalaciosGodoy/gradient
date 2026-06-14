"""
Gradient SDK — metadata-only audit (no prompts or responses captured).

This is the recommended starting point. Run for one week, then use
the export to identify which workflows consume the most AI spend and
which models are candidates for replacement.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_sdk import GradientClient, audit

client = GradientClient(mode="local", local_path="audit_metadata.jsonl")


@audit(
    client=client,
    workflow="support_ticket_reply",
    task_type="customer_support",
    team="support",
    business_unit="customer_success",
    risk_level="medium",
    # Defaults: capture_prompt=False, capture_response=False, redact=True
)
def generate_reply(ticket: str) -> dict:
    """Simulate an LLM call that returns a structured result."""
    return {
        "text": "Thank you for reaching out. We will resolve this shortly.",
        "model": "gpt-4o",
        "provider": "openai",
        "input_tokens": 120,
        "output_tokens": 45,
        "estimated_cost": 0.00127,
    }


if __name__ == "__main__":
    reply = generate_reply("My order hasn't arrived yet.")
    print("Reply:", reply["text"])
    print("Audit event written to audit_metadata.jsonl — no prompt or response captured.")
