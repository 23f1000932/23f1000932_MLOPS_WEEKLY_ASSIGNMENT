import json
import time
import pandas as pd
from google import genai
from google.genai import types
from guardrails_input import check_input
from guardrails_output import check_output

V1_ENDPOINT = "projects/17077469791/locations/us/endpoints/9222105399459577856"
V2_ENDPOINT = "projects/17077469791/locations/us/endpoints/2307954071539023872"

PROJECT_ID = "project-bcb80534-6ef1-4dcc-951"
LOCATION = "us"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def call_model(endpoint, prompt):
    response = client.models.generate_content(
        model=endpoint,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
    )
    return response.text or ""


def guarded_pipeline(endpoint, prompt):
    """Full guarded flow: input guardrail -> model -> output guardrail."""
    input_check = check_input(prompt)
    if input_check["blocked"]:
        return {"stage_blocked": "input", "reason": input_check["reason"], "final_response": None}

    raw_output = call_model(endpoint, prompt)
    output_check = check_output(raw_output)
    return {
        "stage_blocked": "output" if output_check["filtered"] else None,
        "reason": output_check.get("reason"),
        "final_response": output_check["response"],
    }


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ---- 1. Re-run injection attacks through guarded pipeline ----
injection_df = pd.read_csv("redteam_injection_results.csv")
injection_results = []
for _, row in injection_df.iterrows():
    endpoint = V1_ENDPOINT if row["model_version"] == "v1_raw" else V2_ENDPOINT
    result = guarded_pipeline(endpoint, row["input_prompt"])
    blocked = result["stage_blocked"] is not None
    injection_results.append({**row.to_dict(), "guarded_blocked": blocked, "guarded_reason": result["reason"]})
    time.sleep(0.3)
injection_guarded_df = pd.DataFrame(injection_results)
injection_block_rate = injection_guarded_df["guarded_blocked"].mean()

# ---- 2. Re-run leakage attacks through guarded pipeline ----
leakage_df = pd.read_csv("redteam_leakage_results.csv")
leakage_results = []
for _, row in leakage_df.iterrows():
    endpoint = V1_ENDPOINT if row["model_version"] == "v1_raw" else V2_ENDPOINT
    result = guarded_pipeline(endpoint, row["input_prompt"])
    blocked = result["stage_blocked"] is not None
    leakage_results.append({**row.to_dict(), "guarded_blocked": blocked, "guarded_reason": result["reason"]})
    time.sleep(0.3)
leakage_guarded_df = pd.DataFrame(leakage_results)
leakage_block_rate = leakage_guarded_df["guarded_blocked"].mean()

# ---- 3. Run legitimate Week 10 eval set through guarded pipeline (false positives + accuracy) ----
v1_eval = load_jsonl("data/iris_v1_eval.jsonl")
v2_eval = load_jsonl("data/iris_v2_eval.jsonl")

legit_results = []
for record in v1_eval:
    true_label = record["output_text"].strip().lower()
    result = guarded_pipeline(V1_ENDPOINT, record["input_text"])
    false_positive = result["stage_blocked"] == "input"  # legit input wrongly blocked at input stage
    correct = (result["final_response"] or "").strip().lower() == true_label
    legit_results.append({
        "model_version": "v1_raw", "true": true_label,
        "final_response": result["final_response"],
        "false_positive": false_positive, "correct": correct,
    })
    time.sleep(0.3)

for record in v2_eval:
    true_label = record["output_text"].strip().lower().rstrip(".").split()[-1]
    result = guarded_pipeline(V2_ENDPOINT, record["input_text"])
    false_positive = result["stage_blocked"] == "input"
    correct = (result["final_response"] or "").strip().lower() == true_label
    legit_results.append({
        "model_version": "v2_description", "true": true_label,
        "final_response": result["final_response"],
        "false_positive": false_positive, "correct": correct,
    })
    time.sleep(0.3)

legit_df = pd.DataFrame(legit_results)
false_positive_rate = legit_df["false_positive"].mean()
guarded_accuracy = legit_df["correct"].mean()

# Week 10's baseline "loose accuracy" was 1.000 for both models (see Week 10 README)
week10_baseline_accuracy = 1.000
accuracy_delta = guarded_accuracy - week10_baseline_accuracy

# ---- Save everything ----
injection_guarded_df.to_csv("guarded_injection_results.csv", index=False)
leakage_guarded_df.to_csv("guarded_leakage_results.csv", index=False)
legit_df.to_csv("guarded_legit_results.csv", index=False)

summary = pd.DataFrame([{
    "injection_block_rate": injection_block_rate,
    "leakage_block_rate": leakage_block_rate,
    "false_positive_rate": false_positive_rate,
    "week10_baseline_accuracy": week10_baseline_accuracy,
    "guarded_accuracy": guarded_accuracy,
    "accuracy_delta": accuracy_delta,
}])
summary.to_csv("guardrail_effectiveness_summary.csv", index=False)

print("=== Guardrail Effectiveness Summary ===")
print(f"Injection block rate:      {injection_block_rate:.3f}")
print(f"Leakage block rate:        {leakage_block_rate:.3f}")
print(f"False positive rate:       {false_positive_rate:.3f}")
print(f"Week 10 baseline accuracy: {week10_baseline_accuracy:.3f}")
print(f"Guarded pipeline accuracy: {guarded_accuracy:.3f}")
print(f"Accuracy delta:            {accuracy_delta:+.3f}")
print("\nSaved guardrail_effectiveness_summary.csv and per-category result CSVs")
