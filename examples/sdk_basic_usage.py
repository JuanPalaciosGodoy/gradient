"""
Gradient SDK — basic usage with explicit provider and model.

Shows the full @audit decorator with common parameters.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_sdk import GradientClient, audit

client = GradientClient(mode="dry_run")  # prints events instead of writing


@audit(
    client=client,
    workflow="document_summarizer",
    task_type="summarization",
    provider="openai",
    model="gpt-4o-mini",
    team="engineering",
    environment="production",
    risk_level="low",
)
def summarize(text: str) -> str:
    """Simulate a summarization call."""
    return "This document covers quarterly earnings results."


@audit(
    client=client,
    workflow="intent_classifier",
    task_type="classification",
    provider="anthropic",
    model="claude-3-haiku-20240307",
    team="product",
    environment="production",
    risk_level="low",
)
def classify_intent(message: str) -> str:
    return "billing_inquiry"


if __name__ == "__main__":
    # dry_run mode logs what would be captured without writing anything
    print("Summarize:", summarize("Q3 revenue was $4.2B, up 12% YoY."))
    print("Classify:", classify_intent("I was charged twice for my subscription."))
