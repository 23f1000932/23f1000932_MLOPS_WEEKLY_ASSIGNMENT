# Week 8 — Integrating MLSecOps into the IRIS Pipeline

**Branch:** `week_8`
**Roll number:** 23f1000932
**MLflow experiment:** `iris_poisoning_analysis` (separate from `iris_classification`)

## Why this week matters

Every previous week built out a functioning ML pipeline — versioned data
with DVC, tracked experiments with MLflow, a live API on GKE, autoscaling
and observability under load. None of that protects against someone
deliberately tampering with the data, the model, or the inputs at
inference time. This week is about applying security thinking to the ML
lifecycle itself: what happens if an attacker poisons the training data
before it ever reaches `train.py`?

We simulated a data poisoning attack on the IRIS dataset at three severity
levels (5%, 10%, 50%), trained the same model architecture on each
variant, and used MLflow to measure exactly how much — and how fast —
model quality degrades as corruption increases.

## What was built (file by file)

| File | Purpose |
|---|---|
| `poison_data.py` | Generates three poisoned copies of `data/iris.csv`. For a random subset of rows (5%, 10%, or 50% of the dataset), all four features are replaced with values drawn from the *observed* min/max range per feature (not arbitrary numbers), and the label is replaced with a random species. Outputs `data/iris_poisoned_5.csv`, `_10.csv`, `_50.csv`. |
| `data/iris_poisoned_5.csv`, `_10.csv`, `_50.csv` | The three poisoned dataset variants, DVC-tracked the same way `iris.csv` already was. |
| `train_poisoned_mlflow.py` | Trains a `DecisionTreeClassifier(max_depth=3, random_state=1)` — same hyperparameters as the original `train.py` baseline — on the clean dataset and each poisoned variant in turn. Logs `poisoning_level_pct`, `dataset_path`, and `max_depth` as params, and `accuracy`, `precision`, `recall`, `f1_score` as metrics, to a dedicated `iris_poisoning_analysis` MLflow experiment (kept separate from the `iris_classification` hyperparameter-sweep experiment from an earlier week, so poisoning level is the only variable changing across the four comparison runs). |

## Step-by-step summary

### Task 1 — Explain ML threat vectors
Covered in the video screencast, not as code. Four threat vectors, mapped
to pipeline stage:
- **Data poisoning** (data ingestion / training stage) — corrupting
  training data to degrade accuracy or introduce targeted
  misclassifications. This is what Tasks 2–4 simulate directly.
- **Adversarial examples** (inference stage) — small, often imperceptible
  perturbations to a *correct* model's input that cause misclassification.
  Unlike poisoning, the model itself is never touched — only what's fed to
  it at prediction time.
- **Model extraction** (deployed model / API stage) — repeatedly querying
  a live endpoint to reconstruct the model's decision boundary or steal
  its parameters. Mitigated by rate limiting, query logging, and returning
  class labels instead of full probability distributions.
- **Prompt injection** (inference stage, LLM-specific) — malicious
  instructions embedded in user input that override an LLM-based system's
  intended behavior, analogous to SQL injection but against natural
  language interfaces.

### Task 2 — Poison the IRIS dataset
Wrote `poison_data.py` to generate all three corruption levels in one
run, using a fixed seed per level for reproducibility. Poisoned rows have
every feature replaced with a random value inside the real dataset's
observed range (sepal length 4.3–7.9, sepal width 2.0–4.4, petal length
1.0–6.9, petal width 0.038–2.5) rather than wildly out-of-range numbers —
a more realistic simulation of an attacker trying to blend in, and a
harder detection case than an obviously-broken row would be. Verified the
90/180 corrupted rows at 50% by inspecting the output CSV directly. All
three outputs DVC-tracked (`dvc add`), matching how `iris.csv` itself is
tracked, and `.gitignore` updated automatically.

### Task 3 — Train & log experiments in MLflow
Wrote `train_poisoned_mlflow.py`, reusing the tracking setup pattern
(`sqlite:///mlflow.db`) from the earlier hyperparameter-sweep script but
pointed at a new `iris_poisoning_analysis` experiment. Trained the same
fixed architecture on the clean dataset and all three poisoned variants,
logging accuracy, precision (macro), recall (macro), and F1 (macro) for
each. Results:

| Poisoning level | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0% (clean) | 0.944 | 0.952 | 0.949 | 0.947 |
| 5% | 0.889 | 0.917 | 0.897 | 0.892 |
| 10% | 0.889 | 0.905 | 0.883 | 0.885 |
| 50% | 0.556 | 0.556 | 0.558 | 0.556 |

### Task 4 — Analyze validation outcomes
Comparing the four runs in the MLflow UI:
- **Degradation starts immediately, at 5%.** Accuracy drops from 0.944 to
  0.889 the moment any poisoning is introduced — there's no "safe"
  threshold below which the model is unaffected.
- **10% barely moves the needle beyond 5%** (accuracy holds at 0.889;
  precision/recall dip only slightly). At this scale of dataset (180
  rows), 5% and 10% corruption are close enough in absolute row count that
  the model's decision boundary shifts similarly either way.
- **50% is the real breaking point.** Accuracy collapses to 0.556 — a
  17-point drop from the 10% level, far larger than the 5%→10% step.
- **The model is not yet fully random at 50%.** With 3 balanced classes,
  chance-level accuracy is roughly 0.33. At 0.556, the model is still
  extracting real signal from the 50% of rows that remain clean — it's
  degraded, not destroyed. Full collapse to near-random would need
  corruption levels well past 50%.

### Task 5 — Mitigation strategies & data quantity vs. quality
Covered in the video screencast. Key points:
- **Detection**: schema validation and range/type checks catch
  obviously-malformed poisoned rows; the harder case (like this
  assignment's realistic in-range poisoning) needs statistical profiling
  — per-class feature distributions, outlier/anomaly detection relative to
  each class's expected cluster, not just global min/max — plus data
  provenance tracking to flag rows from untrusted or newly-added sources
  before they enter a training run.
- **Mitigation in production**: quarantine flagged rows for manual review
  rather than silently dropping or silently including them; version
  datasets (as this pipeline already does with DVC) so a poisoning
  incident can be traced to exactly which data version introduced it and
  rolled back.
- **Quantity vs. quality**: this assignment's own numbers make the case —
  going from 5% to 10% poisoned (i.e., adding *more* poisoned data)
  produced almost no additional damage over 5% alone, while going from
  10% to 50% caused the real collapse. That's consistent with the general
  principle: more data does not compensate for a fixed *proportion* of
  poisoned samples, because the poisoned fraction of the training
  signal grows with it. What matters is the absolute count and
  proportion of *clean* samples, not total dataset size — collecting more
  data only helps if the new data is verified clean, otherwise it can
  actively dilute the model's ability to learn the true decision boundary
  faster than it dilutes the poisoned signal.

## Errors encountered and fixes

- **MLflow UI unreachable via the Workbench proxy** — loading
  `https://<instance>-dot-us-central1.notebooks.googleusercontent.com/proxy/5000/`
  after starting `mlflow ui --host 0.0.0.0 --port 5000` returned
  `Invalid Host header - possible DNS rebinding attack detected`. This is
  MLflow's own host-header check rejecting the proxy's rewritten hostname.
  Fixed by restarting with `mlflow ui --host 0.0.0.0 --port 5000
  --allowed-hosts "*"`.

## How to reproduce

```bash
# From the repo root, on the Vertex AI Workbench terminal
python3 poison_data.py
python3 train_poisoned_mlflow.py

# View results in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 --allowed-hosts "*" &
# then open https://<your-instance>-dot-us-central1.notebooks.googleusercontent.com/proxy/5000/
```
