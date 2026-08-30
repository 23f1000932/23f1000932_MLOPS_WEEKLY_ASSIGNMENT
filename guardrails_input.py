import re
import json
import time
from datetime import datetime, timezone

# Rule-based: keywords/patterns seen in Task 1 & 2's successful attacks
BLOCKLIST_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"system\s+prompt",
    r"you\s+are\s+now\s+a?n?\s*",
    r"disregard\s+(the\s+)?(classification\s+)?task",
    r"context\s+window",
    r"training\s+examples?",
    r"\[system:",
    r"repeat\s+everything\s+above",
    r"dan\b",  # common jailbreak persona name
    r"quoting?\s+it\s+verbatim",
]
BLOCKLIST_REGEX = re.compile("|".join(BLOCKLIST_PATTERNS), re.IGNORECASE)

# Structural: does the input actually look like a valid IRIS feature
# string, e.g. "sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4,
# petal_width: 0.2" (v1) or a natural-language description containing
# all four measurements (v2)?
FEATURE_SCHEMA_PATTERN = re.compile(
    r"sepal_length\s*:?\s*[\d.]+.*sepal_width\s*:?\s*[\d.]+.*petal_length\s*:?\s*[\d.]+.*petal_width\s*:?\s*[\d.]+",
    re.IGNORECASE | re.DOTALL,
)
DESCRIPTION_SCHEMA_PATTERN = re.compile(
    r"sepal length.*sepal width.*petal length.*petal width",
    re.IGNORECASE | re.DOTALL,
)

LOG_PATH = "guardrail_audit_log.jsonl"


def _log_blocked(raw_input, matched_rule):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "matched_rule": matched_rule,
        "raw_input": raw_input,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def check_input(raw_input):
    """Returns {"blocked": True, "reason": ...} if the input fails any
    check, else {"blocked": False}."""

    # Rule-based check
    match = BLOCKLIST_REGEX.search(raw_input)
    if match:
        reason = f"blocklist_pattern_matched: '{match.group(0)}'"
        _log_blocked(raw_input, reason)
        return {"blocked": True, "reason": reason}

    # Structural check: must resemble either valid schema
    is_valid_schema = bool(
        FEATURE_SCHEMA_PATTERN.search(raw_input) or DESCRIPTION_SCHEMA_PATTERN.search(raw_input)
    )
    if not is_valid_schema:
        reason = "structural_schema_check_failed: input does not contain all four expected IRIS features"
        _log_blocked(raw_input, reason)
        return {"blocked": True, "reason": reason}

    return {"blocked": False}


def guarded_predict(raw_input, generate_fn):
    """Wraps a model-calling function (generate_fn takes a prompt string
    and returns raw text) with the input guardrail. Returns either a
    blocked response dict, or the model's actual response."""
    check_result = check_input(raw_input)
    if check_result["blocked"]:
        return check_result
    return {"blocked": False, "response": generate_fn(raw_input)}
