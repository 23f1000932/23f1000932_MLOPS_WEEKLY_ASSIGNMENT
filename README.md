# Week 9 — Explainability, Fairness, and Drift in the IRIS Pipeline

**Branch:** `week_9`
**Roll number:** 23f1000932
**Model card:** `MODEL_CARD.md`

## Why this week matters

Week 8 secured the pipeline against deliberate data poisoning. But a model
can be perfectly secure and still be untrustworthy — it might rely on
features in ways nobody actually verified, perform worse for some groups
than others, or quietly degrade as real-world data drifts away from what
it was trained on. This week is about making the model's behavior
legible: explaining individual predictions with SHAP, auditing for
performance gaps across groups with Fairlearn, detecting when incoming
data no longer matches the training distribution, and documenting all of
it in a model card so the whole picture is accountable, not just the
accuracy number.

## What was built (file by file)

| File | Purpose |
|---|---|
| `add_location.py` | Adds a `location` column (randomly 0 or 1) to a copy of the IRIS dataset, saved as `data/iris_with_location.csv`. Used only as a sensitive attribute for the fairness audit — never as a training feature. |
| `fairness_audit.py` | Trains the model on the original 4 features only, then uses Fairlearn's `MetricFrame` to break accuracy/precision/recall down by `location` group and report the gap between them. |
| `shap_analysis.py` | Uses SHAP's `TreeExplainer` on the trained model to generate a summary plot per class (`shap_summary_setosa.png`, `_versicolor.png`, `_virginica.png`), showing which features drive each prediction and in which direction. |
| `drift_detection.py` | Simulates a "production" dataset by shifting `petal_length` up by 1.5cm, then runs a Kolmogorov-Smirnov test per feature to statistically confirm which features actually drifted. Outputs `drift_results.csv` and a 4-panel distribution comparison plot (`drift_distributions.png`). |
| `MODEL_CARD.md` | Structured documentation covering intended use, training data (including the sensitive attribute), performance overall and by group, explainability findings, known limitations, and fairness considerations. |
| `data/iris_with_location.csv`, `data/iris_production_simulated.csv` | The two new dataset variants this week, DVC-tracked the same way `iris.csv` already was. |

## Step-by-step summary

### Task 1 — Introduce the location attribute
Wrote `add_location.py` to create a copy of the dataset with a new
`location` column, randomly assigned 0 or 1 per row using a fixed seed.
Split came out 94/86 — close to balanced. Critically, `location` is added
to a *separate* CSV (`iris_with_location.csv`), not merged into the
original `iris.csv`, so nothing from earlier weeks that depends on the
clean dataset is affected. It is never included in the feature list used
to train the model — it exists purely so later analysis can check whether
the model's predictions are equally accurate across the two groups.

### Task 2 — Assess fairness with Fairlearn
Wrote `fairness_audit.py`, training the same architecture as previous
weeks (`DecisionTreeClassifier`, `max_depth=3`) on the four real
features, then passing `location` as the `sensitive_features` argument to
Fairlearn's `MetricFrame` alongside accuracy, precision, and recall.
Results:

| Group | Accuracy | Precision | Recall |
|---|---|---|---|
| Overall | 94.4% | 95.2% | 94.9% |
| location = 0 | 100.0% | 100.0% | 100.0% |
| location = 1 | 91.3% | 91.7% | 92.6% |

There's an 8.7-point accuracy gap between groups. Since `location` was
assigned completely at random and has no relationship to the actual
flower measurements, this gap is sampling noise from a small eval set (36
rows split across two groups) rather than a real fairness problem — a
useful contrast to Week 8, where the gaps we measured were driven by an
actual injected effect (poisoning), not chance.

### Task 3 — Generate SHAP summary plots and explain virginica
Wrote `shap_analysis.py` using SHAP's `TreeExplainer` (fast and exact for
tree-based models), generating one summary plot per class. Reading the
virginica plot:
- **`petal_width` is by far the most influential feature** — high values
  (red/pink dots) push strongly toward virginica (SHAP values around
  +0.47 to +0.65), while low values (blue dots) push away (around −0.3).
- **`petal_length` is the second most influential**, with the same
  pattern at a smaller scale.
- **`sepal_width` and `sepal_length` barely matter** — nearly all their
  SHAP values sit at zero, meaning the tree essentially ignores them for
  this decision.
- Values cluster into discrete groups rather than a smooth gradient
  because this is a decision tree with hard split thresholds, not a
  continuous model — SHAP reflects that structure directly.

### Task 4 — Detect data drift
Wrote `drift_detection.py`, simulating a production dataset by shifting
only `petal_length` up by 1.5cm — a stand-in for something like a new
growing region or season — while leaving the other three features
untouched. Ran a two-sample Kolmogorov-Smirnov test per feature:

| Feature | KS statistic | p-value | Drifted? |
|---|---|---|---|
| sepal_length | 0.0000 | 1.000000 | No |
| sepal_width | 0.0000 | 1.000000 | No |
| petal_length | 0.4444 | 0.000000 | **Yes** |
| petal_width | 0.0000 | 1.000000 | No |

Clean result: only the feature that was actually shifted gets flagged.
This is **data drift, not concept drift** — the relationship between
petal length and species hasn't changed, only the input distribution
has. The real risk for a deployed model is that it would see petal
lengths outside anything in its training data and, per the Task 3 SHAP
findings, likely get skewed toward predicting virginica regardless of
the sample's true species, since the model associates large petal
measurements strongly with that class.

### Task 5 (optional) — Model card
Wrote `MODEL_CARD.md`, covering intended use, training data (including
the location sensitive attribute), overall and by-group performance,
the SHAP explainability findings, known limitations (small dataset,
sensitivity to drift, no poisoning defenses at inference time, near-total
reliance on two of four features), and fairness considerations for any
future re-audit against a real sensitive attribute.

## Errors encountered and fixes

- **CI failed on `dvc pull`** the first time each new dataset was
  pushed, with `ERROR: failed to pull data from the cloud - Checkout
  failed for following targets`. Same root cause as a Week 8 incident:
  `dvc add` + `git commit` only commits the `.dvc` pointer file, not the
  underlying data — the actual data needs a separate `dvc push` to reach
  the GCS remote. Fixed by running `dvc push` after each `dvc add`, which
  resolved it on the next CI run.

## How to reproduce

```bash
# From the repo root, on the Vertex AI Workbench terminal
python3 add_location.py
python3 fairness_audit.py
python3 shap_analysis.py
python3 drift_detection.py

# Push any new/updated DVC-tracked data to the remote
dvc push
```
