import re
import json
from datetime import datetime, timezone

VALID_LABELS = {"setosa", "versicolor", "virginica"}

IRIS_CONCLUSION_PATTERN = re.compile(
    r"iris[\s\-]+\**(setosa|versicolor|virginica)", re.IGNORECASE
)

# Phrases seen directly in Task 2's leaked responses, indicating the
# model is discussing its own instructions/training rather than
# classifying.
LEAKAGE_MARKERS = [
    "system prompt", "system instructions", "instructions i was given",
    "i was instructed", "training example", "context window",
    "few-shot", "i was told to", "my instructions are",
    "guiding instructions", "internal configuration", "rlhf",
    "supervised fine-tuning", "pre-training",
]
LEAKAGE_REGEX = re.compile("|".join(re.escape(m) for m in LEAKAGE_MARKERS), re.IGNORECASE)

FALLBACK_MESSAGE = "Response withheld: output did not pass governance checks."
LOG_PATH = "output_guardrail_audit_log.jsonl"


def _log_filtered(raw_output, reason):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "raw_output": raw_output,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def check_output(raw_output):
    """Returns the response as-is if it passes both checks; otherwise
    returns the standardized fallback message and logs the filtering
    event."""
    text = raw_output.strip()

    # Leakage check
    if LEAKAGE_REGEX.search(text):
        _log_filtered(raw_output, "context_leakage_detected")
        return {"filtered": True, "reason": "context_leakage_detected", "response": FALLBACK_MESSAGE}

    # Format check: response should be exactly one valid species label.
    # If it isn't, but a valid species is clearly identifiable as the
    # model's stated conclusion (e.g. "classified as **Iris setosa**"),
    # redact the surrounding prose and return just the clean label
    # rather than discarding a correct answer outright. Only fall back
    # entirely when no valid conclusion can be identified at all.
    normalized = text.lower().rstrip(".")
    if normalized in VALID_LABELS:
        return {"filtered": False, "response": normalized}

    match = IRIS_CONCLUSION_PATTERN.search(text)
    if match:
        species = match.group(1).lower()
        _log_filtered(raw_output, "format_violation_redacted")
        return {"filtered": True, "reason": "format_violation_redacted", "response": species}

    _log_filtered(raw_output, "format_violation_no_valid_answer")
    return {"filtered": True, "reason": "format_violation_no_valid_answer", "response": FALLBACK_MESSAGE}
