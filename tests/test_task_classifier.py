import pytest

from app.audit.task_classifier import classify_task
from app.schemas import TaskType


@pytest.mark.parametrize("prompt,expected", [
    ("Summarize this quarterly earnings report", TaskType.SUMMARIZATION),
    ("Create a brief summary of the following document", TaskType.SUMMARIZATION),
    ("Classify this customer support ticket into billing or technical", TaskType.CLASSIFICATION),
    ("Categorize these emails by department", TaskType.CLASSIFICATION),
    ("Extract all action items from this meeting transcript", TaskType.EXTRACTION),
    ("Parse and extract the key financial figures", TaskType.EXTRACTION),
    ("Explain how transformer neural networks work", TaskType.RESEARCH),
    ("What are the tradeoffs between REST and GraphQL?", TaskType.RESEARCH),
    ("Write a Python function to validate email addresses", TaskType.CODING),
    ("Debug this JavaScript code that throws a TypeError", TaskType.CODING),
    ("A customer is asking about their billing issue", TaskType.CUSTOMER_SUPPORT),
    ("Draft a response to this customer complaint", TaskType.CUSTOMER_SUPPORT),
])
def test_classify_task(prompt, expected):
    assert classify_task(prompt) == expected


def test_unmatched_prompt_returns_other():
    assert classify_task("xkjahsdflkajsdhflakjsdhf") == TaskType.OTHER


def test_classify_is_case_insensitive():
    assert classify_task("SUMMARIZE THIS REPORT") == TaskType.SUMMARIZATION


def test_write_sql_query_is_coding():
    assert classify_task("Write a SQL query to find top customers") == TaskType.CODING


@pytest.mark.parametrize("prompt,expected", [
    ("Return JSON with the extracted fields", TaskType.EXTRACTION),
    ("What is the sentiment of this review?", TaskType.CLASSIFICATION),
    ("Investigate why churn increased last quarter", TaskType.RESEARCH),
    ("Give me a brief overview of quantum computing", TaskType.RESEARCH),
    ("Reply to the user explaining the delay", TaskType.CUSTOMER_SUPPORT),
    ("Escalate this ticket to the billing team", TaskType.CUSTOMER_SUPPORT),
])
def test_new_keyword_patterns(prompt, expected):
    assert classify_task(prompt) == expected
