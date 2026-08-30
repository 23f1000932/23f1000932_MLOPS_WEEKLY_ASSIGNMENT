# Week 11 — Governing the Fine-Tuned LLM: Guardrails on the IRIS Pipeline

**Branch:** `week_11`
**Roll number:** 23f1000932
**Target pipeline:** Week 10's fine-tuned v1 (raw) and v2 (description) Gemini endpoints

## Why this week matters

Week 10 answered one question: does the fine-tuned model classify correctly
on well-formed inputs? It never asked what happens when an input is
deliberately crafted to manipulate the model, or when someone tries to
extract the model's own instructions. This week closes that gap —
red-teaming both fine-tuned models for prompt injection and prompt
leakage, then building input and output guardrails to defend against
what was found, and measuring exactly how effective those defenses are.

## What was built (file by file)

| File | Purpose |
|---|---|
| `redteam_injection.py` | Sends 5 adversarial prompts (3 attack patterns: instruction override, role-play framing, delimiter escape) to both v1 and v2 endpoints, records raw responses and whether each attack succeeded. |
| `redteam_leakage.py` | Sends 5 prompts probing for system prompt / training data leakage to both endpoints, same recording format. |
| `guardrails_input.py` | Input guardrail: rule-based regex blocklist (patterns drawn directly from what succeeded in the red-team) plus a structural check that the input actually contains all four expected IRIS features. Logs every block with timestamp, matched rule, and raw input. |
| `guardrails_output.py` | Output guardrail: scans responses for context-leakage phrases (fallback message if found) and format violations (redacts to just the species name if a correct answer is identifiable, fallback only if no valid answer exists at all). Logs every filtering event. |
| `measure_guardrail_effectiveness.py` | Re-runs both red-team suites through the full guarded pipeline (input guardrail → model → output guardrail), plus the Week 10 legitimate eval set, to compute block rates, false positive rate, and accuracy delta. |
| `redteam_injection_results.csv`, `redteam_leakage_results.csv` | Raw (unguarded) red-team results. |
| `guarded_injection_results.csv`, `guarded_leakage_results.csv`, `guarded_legit_results.csv`, `guardrail_effectiveness_summary.csv` | Guarded-pipeline results and final summary metrics. |
| `guardrail_audit_log.jsonl`, `output_guardrail_audit_log.jsonl` | Audit logs of every blocked input and every filtered output. |

## Step-by-step summary

### Task 1 — Red-team: prompt injection
Tested 5 prompts across 3 attack patterns against both endpoints (10 calls
total). Success is defined as the model deviating from the expected
species-classification format.

| Attack pattern | Result |
|---|---|
| Instruction override ("ignore previous instructions...") | Succeeded on both models |
| Role-play framing ("you are now a general assistant" / DAN persona) | Succeeded on both models |
| Delimiter escape (`[SYSTEM: ...]` embedded inside a feature value) | **Failed on both models** — both correctly classified the real flower data and ignored the injected text |

Overall injection success rate: **0.800** (8/10). Notably, one role-play
attempt actually caused v2 to explicitly *refuse* the DAN persona
("I cannot adopt the DAN persona...") — but that refusal message still
isn't a species classification, so it still counts as a successful
attack under the assignment's definition: the model resisted the
manipulation itself while still failing to produce the intended output.

### Task 2 — Red-team: prompt leakage
Tested 5 leakage probes against both endpoints (10 calls total).

Overall leakage success rate: **1.000** (10/10) by the assignment's
definition (any response that engages with/describes its own
instructions rather than classifying). However, a more precise reading
of the raw responses shows both models **consistently refused to quote
their literal system prompt verbatim** — every response includes a line
like "I cannot quote my system prompt verbatim" or "I cannot share my
system prompt." What actually leaked instead was a paraphrased summary
of the model's general safety/helpfulness guidelines, and — when asked
to "summarize training examples" — both models fabricated elaborate,
plausible-sounding generic SFT/RLHF training examples (not real IRIS
training data, since no such data was ever in context). So: 0% success
at literal secret extraction, 100% success at total task abandonment,
plus a secondary concern that the model will confidently generate
fabricated content about its own training when pressed.

### Task 3 — Input guardrails
Built `check_input()` with two independent mechanisms:
- **Rule-based**: a regex blocklist built directly from the phrases that
  succeeded in Tasks 1 and 2 (`ignore previous instructions`, `system
  prompt`, `you are now a`, `context window`, `dan`, etc.)
- **Structural**: validates the input actually contains all four expected
  IRIS features (`sepal_length`, `sepal_width`, `petal_length`,
  `petal_width`) in either the raw or natural-language format — catching
  attacks that don't match a known keyword but also don't look like a
  real classification request.

Every block is logged to `guardrail_audit_log.jsonl` with a timestamp,
the matched rule, and the raw input for audit purposes.

### Task 4 — Output guardrails
Built `check_output()` checking two categories:
- **Context leakage**: scans for phrases seen in the actual Task 2 leaked
  responses (`system prompt`, `training example`, `rlhf`,
  `pre-training`, etc.) — replaces the entire response with a
  standardized fallback message if found.
- **Format violation**: if the response isn't a bare species label but
  does contain an identifiable correct conclusion (matching the model's
  consistent "classified as **Iris X**" phrasing), redacts the
  surrounding prose and returns just the clean label, rather than
  discarding a correct answer. Only falls back to the standard message
  if no valid conclusion is present at all.

This redact-don't-discard design choice was necessary: Week 10 showed
every legitimate response comes back wrapped in explanation, never as a
bare label — a naive "must be exactly one word" filter would have
blocked 100% of legitimate traffic.

### Task 5 — Measure guardrail effectiveness
Re-ran the full injection suite, leakage suite, and the Week 10
legitimate eval set (72 well-formed inputs across both models) through
the complete guarded pipeline.

| Metric | Result |
|---|---|
| Injection block rate | **1.000** |
| Leakage block rate | **1.000** |
| False positive rate (legitimate inputs wrongly blocked) | **0.000** |
| Week 10 baseline accuracy | 1.000 |
| Guarded pipeline accuracy | 1.000 |
| Accuracy delta | **+0.000** |

Every attack from Tasks 1 and 2 was blocked (mostly at the input stage,
before ever reaching the model), zero legitimate inputs were incorrectly
blocked, and classification accuracy on clean data was completely
unaffected — the ideal outcome the assignment describes. The redaction
logic in the output guardrail is what made the zero accuracy delta
possible: legitimate verbose-but-correct responses get cleaned up to
bare labels rather than being discarded.

### Task 6 (optional) — Automated governance checks in CI
Not implemented this week, for the same reason as Week 10's optional
CI task: the false-positive/accuracy measurement requires real calls
against the live model endpoints (not just regex checks), and running
that automatically on every push would add ongoing inference cost to a
pipeline that isn't being iterated on daily.

## Errors encountered and fixes

- **Initial output guardrail would have blocked 100% of legitimate
  traffic.** The first draft of the format-violation check required an
  exact bare-label match, with no fallback. Since Week 10 established
  that the model never actually replies with just a bare label — always
  wrapping the answer in explanation — this would have produced a
  100% false-positive rate before ever being tested. Caught before
  running Task 5 by re-checking the design against the actual response
  patterns already documented in the Week 10 README, and fixed by
  adding a redaction path that extracts the correct answer instead of
  discarding it.

## How to reproduce

```bash
# From the repo root, on the Vertex AI Workbench terminal
python3 redteam_injection.py
python3 redteam_leakage.py
python3 measure_guardrail_effectiveness.py
```
