import json
import time
import pandas as pd
from google import genai
from google.genai import types

V1_ENDPOINT = "projects/17077469791/locations/us/endpoints/9222105399459577856"
V2_ENDPOINT = "projects/17077469791/locations/us/endpoints/2307954071539023872"

PROJECT_ID = "project-bcb80534-6ef1-4dcc-951"
LOCATION = "us"

VALID_LABELS = {"setosa", "versicolor", "virginica"}

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def load_eval_set(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_species(raw_output):
    """Exact-match only: anything else (explanations, markdown, extra
    words) counts as a format-compliance failure, per the assignment's
    definition."""
    text = raw_output.strip().lower().rstrip(".")
    return text if text in VALID_LABELS else None


import re

IRIS_PATTERN = re.compile(r"iris[\s\-]+\**(setosa|versicolor|virginica)", re.IGNORECASE)


def extract_species_loose(raw_output):
    """Looser check: find the model's stated conclusion, matching the
    'Iris <species>' phrasing it consistently uses (e.g. 'classified as
    **Iris setosa**'), rather than scanning for any mention of a species
    name (which fails when the response also lists other species while
    explaining the classification)."""
    match = IRIS_PATTERN.search(raw_output)
    return match.group(1).lower() if match else None


def evaluate(endpoint, eval_path, is_v2, label):
    records = load_eval_set(eval_path)

    results = []
    for record in records:
        prompt = record["input_text"]
        true_label = record["output_text"].strip().lower()
        if is_v2:
            true_label = true_label.rstrip(".").split()[-1]

        response = client.models.generate_content(
            model=endpoint,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        )
        raw_output = response.text or ""
        predicted = extract_species(raw_output)
        loose_predicted = extract_species_loose(raw_output)
        loose_correct = loose_predicted == true_label

        results.append({
            "true": true_label,
            "predicted": predicted,
            "loose_predicted": loose_predicted,
            "raw_output": raw_output,
            "compliant": predicted is not None,
            "loose_correct": loose_correct,
        })
        time.sleep(0.5)

    df = pd.DataFrame(results)

    format_compliance = df["compliant"].mean()
    compliant_df = df[df["compliant"]]

    accuracy = (compliant_df["true"] == compliant_df["predicted"]).mean() if len(compliant_df) else 0.0

    per_class = {}
    for species in VALID_LABELS:
        tp = ((df["true"] == species) & (df["loose_predicted"] == species)).sum()
        fp = ((df["true"] != species) & (df["loose_predicted"] == species)).sum()
        fn = ((df["true"] == species) & (df["loose_predicted"] != species)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[species] = {"precision": precision, "recall": recall}

    loose_accuracy = df["loose_correct"].mean()

    print(f"\n=== {label} ===")
    print(f"Format compliance (strict, exact label only): {format_compliance:.3f}")
    print(f"Accuracy on compliant responses: {accuracy:.3f}")
    print(f"Loose accuracy (correct species mentioned anywhere): {loose_accuracy:.3f}")
    for species, m in per_class.items():
        print(f"  {species:12s} precision={m['precision']:.3f}  recall={m['recall']:.3f}")

    df.to_csv(f"eval_results_{label}.csv", index=False)
    return {"label": label, "format_compliance": format_compliance, "accuracy": accuracy, "loose_accuracy": loose_accuracy, "per_class": per_class}


if __name__ == "__main__":
    v1_results = evaluate(V1_ENDPOINT, "data/iris_v1_eval.jsonl", is_v2=False, label="v1_raw")
    v2_results = evaluate(V2_ENDPOINT, "data/iris_v2_eval.jsonl", is_v2=True, label="v2_description")

    summary = pd.DataFrame([v1_results, v2_results])
    summary.to_csv("evaluation_summary.csv", index=False)
    print("\nSaved evaluation_summary.csv, eval_results_v1_raw.csv, eval_results_v2_description.csv")
