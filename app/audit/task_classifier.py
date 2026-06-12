import re

from app.schemas import TaskType

# Patterns are checked in order; first match wins.
_PATTERNS: list[tuple[TaskType, list[str]]] = [
    (TaskType.SUMMARIZATION, [
        r"\bsummariz", r"\bsummary\b", r"\btl;?dr\b", r"\bshorten\b", r"\bcondense\b",
        r"\bexecutive summary\b",
    ]),
    (TaskType.CLASSIFICATION, [
        r"\bclassif", r"\bcategori", r"\blabel\b", r"\bsentiment\b",
        r"\bwhich (category|type|class)\b", r"\bsort (into|by)\b",
        r"\btag (this|these|the)\b", r"\bis this (spam|positive|negative|urgent)\b",
    ]),
    (TaskType.EXTRACTION, [
        r"\bextract\b", r"\bpull out\b", r"\bparse\b", r"\bfind fields\b",
        r"\blist (all|the|key)\b", r"\bfind (all|the)\b",
        r"\bjson\b", r"\bstructured (output|data|format)\b",
    ]),
    (TaskType.RESEARCH, [
        r"\bresearch\b", r"\bexplain\b", r"\bwhat (is|are)\b", r"\bhow does\b",
        r"\bcompare\b", r"\banalyze\b", r"\binvestigate\b", r"\bpros and cons\b",
        r"\bdifference between\b", r"\btradeoff", r"\bwhen should\b",
        r"\boverview of\b", r"\btell me about\b",
    ]),
    (TaskType.CODING, [
        r"\bwrite (?:\w+ ){0,3}(function|class|script|code|program|test|query)\b",
        r"\bimplement\b", r"\bdebug\b", r"\bfix (this|the|my) (bug|code|error)\b",
        r"\brefactor\b", r"\bsql\b", r"\bpython\b", r"\bjavascript\b",
        r"\bunit test\b", r"\bapi endpoint\b",
    ]),
    (TaskType.CUSTOMER_SUPPORT, [
        r"\bcustomer\b", r"\bticket\b", r"\bsupport\b", r"\bcomplaint\b",
        r"\brefund\b", r"\bapolog", r"\bdraft a (response|reply|email)\b",
        r"\baccount manager\b", r"\breply to (the |a |this )?user\b",
        r"\bescalat", r"\bonboarding\b",
    ]),
]


def classify_task(prompt: str) -> TaskType:
    text = prompt.lower()
    for task_type, patterns in _PATTERNS:
        if any(re.search(p, text) for p in patterns):
            return task_type
    return TaskType.OTHER
