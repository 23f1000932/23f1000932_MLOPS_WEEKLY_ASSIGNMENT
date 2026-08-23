import json
import time
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel

# TODO: fill these in once tuning jobs finish (from the Vertex AI console,
# tuning job's "Deploy & test" tab -> endpoint resource name, format:
# projects/<PROJECT_NUMBER>/locations/us-central1/endpoints/<ENDPOINT_ID>)
V1_ENDPOINT = "REPLACE_ME_V1_ENDPOINT"
V2_ENDPOINT = "REPLACE_ME_V2_ENDPOINT"

PROJECT_ID = "project-bcb80534-6ef1-4dcc-951"
LOCATION = "us-central1"

VALID_LABELS = {"setosa", "versicolor", "virginica"}

vertexai.init(project=PROJECT_ID, location=LOCATION)


def load_eval_set(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_species(raw_output, is_v2):
    """Normalize a model's raw text output into a species label, or None
    if it doesn't match any valid species (a format-compliance failure)."""
    text = raw_output.strip().lower()
    if is_v2:
        # v2 expects "This is Iris setosa." -> pull out the last word, strip punctuation
        text = text.rstrip(".").split()[-1] if text else ""
    for label in VALID_LABELS:
        if label == text:
            return label
    return None


def evaluate(endpoint_resource_name, eval_path, is_v2, label):
    model = GenerativeModel(endpoint_resource_name)
    records = load_eval_set(eval_path)

    results = []
    for record in records:
        prompt = record["input_text"]
        true_label = record["output_text"].strip().lower()
        if is_v2:
            true_label = true_label.rstrip(".").split()[-1]

        response = model.generate_content(prompt)
        raw_output = response.text
        predicted = extract_species(raw_output, is_v2)

        results.append({
            "true": true_label,
            "predicted": predicted,
            "raw_output": raw_output,
            "compliant": predicted is not None,
        })
        time.sleep(0.5)  # gentle rate limiting

    df = pd.DataFrame(results)

    format_compliance = df["compliant"].mean()
    compliant_df = df[df["compliant"]]

    accuracy = (compliant_df["true"] == compliant_df["predicted"]).mean() if len(compliant_df) else 0.0

    per_class = {}
    for species in VALID_LABELS:
        tp = ((compliant_df["true"] == species) & (compliant_df["predicted"] == species)).sum()
        fp = ((compliant_df["true"] != species) & (compliant_df["predicted"] == species)).sum()
        fn = ((compliant_df["true"] == species) & (compliant_df["predicted"] != species)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[species] = {"precision": precision, "recall": recall}

    print(f"\n=== {label} ===")
    print(f"Format compliance: {format_compliance:.3f}")
    print(f"Accuracy (on compliant responses): {accuracy:.3f}")
    for species, m in per_class.items():
        print(f"  {species:12s} precision={m['precision']:.3f}  recall={m['recall']:.3f}")

    df.to_csv(f"eval_results_{label}.csv", index=False)
    return {"label": label, "format_compliance": format_compliance, "accuracy": accuracy, "per_class": per_class}


if __name__ == "__main__":
    v1_results = evaluate(V1_ENDPOINT, "data/iris_v1_eval.jsonl", is_v2=False, label="v1_raw")
    v2_results = evaluate(V2_ENDPOINT, "data/iris_v2_eval.jsonl", is_v2=True, label="v2_description")

    summary = pd.DataFrame([v1_results, v2_results])
    summary.to_csv("evaluation_summary.csv", index=False)
    print("\nSaved evaluation_summary.csv, eval_results_v1_raw.csv, eval_results_v2_description.csv")
