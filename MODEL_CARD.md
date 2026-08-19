# Model Card — IRIS Species Classifier

## Intended Use
Classifies iris flowers into one of three species (setosa, versicolor,
virginica) from four physical measurements. Built as a teaching pipeline
for this course, deployed as a REST API (`/predict`) on GKE. Not intended
for use outside this educational context, and not validated on any
dataset beyond the classic IRIS dataset.

## Training Data
- Source: the classic IRIS dataset, 180 rows (`data/iris.csv`), 60 rows
  per species — perfectly balanced.
- Features used for training: `sepal_length`, `sepal_width`,
  `petal_length`, `petal_width` (all continuous, measured in cm).
- Target: `species` (3 classes: setosa, versicolor, virginica).
- A `location` column (randomly assigned 0/1) was added this week
  specifically as a **sensitive attribute for fairness auditing** — it is
  never used as a training feature. It represents a stand-in for a real
  demographic or group identifier (e.g. region of collection) that a
  production system might need to audit fairness against, even without
  using it as a predictor.

## Model
`DecisionTreeClassifier` (scikit-learn), `max_depth=3`, `random_state=1`.
Trained on an 80/20 stratified train/eval split (`random_state=42`).

## Performance
- Overall accuracy on clean, undrifted eval data: **94.4%**
- Precision (macro): **95.2%**, Recall (macro): **94.9%**

### By location group (fairness audit, Week 9 Task 2)
| Group | Accuracy | Precision | Recall |
|---|---|---|---|
| location = 0 | 100.0% | 100.0% | 100.0% |
| location = 1 | 91.3% | 91.7% | 92.6% |

The ~8.7-point accuracy gap between groups reflects the small eval set
size (36 rows split across two groups) rather than real bias — `location`
was assigned at random and carries no relationship to the actual flower
measurements or true species label.

## Explainability (Week 9 Task 3)
SHAP analysis shows `petal_width` and `petal_length` dominate every
class's predictions; `sepal_width` and `sepal_length` have almost no
influence on the model's decisions. For virginica specifically, high
petal width and petal length values are the strongest evidence in favor
of the class; low values are the strongest evidence against it.

## Known Limitations
- **Trained on a small, synthetic-scale dataset (180 rows).** Real-world
  deployments would need substantially more data and more diverse
  conditions before this level of confidence is warranted.
- **Sensitive to distribution shift.** Week 9 Task 4 showed that a
  1.5cm upward shift in `petal_length` alone is easily detectable via a
  Kolmogorov-Smirnov test (p < 0.001) and would likely skew predictions
  toward virginica for any species, since the model has never seen
  petal lengths in that range.
- **No poisoning defenses at inference time.** Week 8 showed the model's
  accuracy degrades sharply once training data corruption exceeds 10%
  — this model was trained on clean data only, but nothing in the
  current pipeline automatically detects poisoned data before training.
- **Ignores two of its four input features almost entirely**
  (`sepal_width`, `sepal_length`), per the SHAP analysis. This is a
  property of this specific small dataset and shallow tree, not a
  general guarantee that these features are unimportant for iris
  classification.

## Fairness Considerations
This model was audited against one sensitive attribute (`location`),
which was randomly assigned in this exercise and is not a real
demographic. In a production setting, this audit process (Fairlearn's
`MetricFrame`, comparing accuracy/precision/recall across groups) should
be re-run against real sensitive attributes relevant to the deployment
context, and should be repeated on a much larger eval set than the 36
rows used here, since small eval sets can show a fairness gap that is
purely sampling noise rather than a real disparity.
