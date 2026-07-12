# 23f1000932_MLOPS_WEEKLY_ASSIGNMENT — Week 3

## Feast Feature Store Integration for the IRIS Pipeline

This branch (`week_3`) extends the IRIS pipeline with a **feature store layer**,
using [Feast](https://feast.dev/), so that training and inference draw features
from a single, consistent source — instead of the model reading raw CSV data
directly.

## Why This Was Needed

In Week 2, DVC gave version control for data and model files — it answers
*"which version of the data produced this model?"* But DVC does not address a
different problem: *"how do I make sure training and real-time inference always
see the exact same engineered features?"* If training reads a CSV one way and a
production service computes features another way, small inconsistencies
("training/serving skew") can silently degrade model quality. Feast solves this
by sitting between raw data and the model, offering one place to define, store,
and retrieve features — both for historical (offline) training and low-latency
(online) inference.

## Objective

1. Initialize a Feast feature repository within the existing IRIS project.
2. Define an entity, a data source, and a feature view for the IRIS dataset.
3. Register these definitions and materialize feature values into the online
   store.
4. Train the IRIS classifier using features fetched from Feast's **offline**
   store — not the raw CSV.
5. Simulate real-time inference by fetching features for specific samples from
   Feast's **online** store, and verify the predictions are consistent with the
   ground truth.

## Repository Structure (week_3 branch)

```
iris_feature_repo/
└── feature_repo/
    ├── feature_store.yaml       # Feast config: local provider, SQLite backend
    ├── iris_features.py         # Entity, data source, and feature view definitions
    ├── data/
    │   ├── iris.parquet         # IRIS data with iris_id + event_timestamp columns
    │   ├── iris.parquet.csv     # Same data in CSV form (for easy inspection)
    │   └── registry.db          # Feast registry (tracks applied definitions)
    └── model_feast.joblib       # Model trained on features fetched from Feast
23f1000932_Assignment_3_May2026_MLOps.ipynb   # Main notebook: all steps below
README.md
```

## File-by-File: What Each One Does and Why

| File | Purpose |
|---|---|
| `feature_store.yaml` | Feast's core config file. Configured to use the **local** provider with a **SQLite** online store — no cloud backend needed for this assignment, keeping the setup simple and self-contained. |
| `iris_features.py` | Defines the three building blocks Feast needs: an **Entity** (`iris_id`, uniquely identifying each flower sample), a **FileSource** (pointing at `data/iris.parquet`), and a **FeatureView** (`iris_features`, mapping the four numeric IRIS columns — sepal/petal length and width — to that entity and source). This file must exist on disk (not just as notebook variables) because `feast apply` scans `.py` files in the repo to discover definitions. |
| `data/iris.parquet` / `iris.parquet.csv` | The IRIS dataset (180 rows — the augmented version carried over from Week 2) with two columns added: `iris_id` (a unique integer per row, used as the entity key) and `event_timestamp` (set to the notebook's run time for every row). Feast requires every row to have a timestamp since it's designed for data that changes over time; IRIS is static, so this timestamp is a placeholder rather than a meaningful "event time." |
| `data/registry.db` | Feast's registry — created by `feast apply`, it records the entity and feature view definitions so Feast knows what's available to serve. |
| `model_feast.joblib` | The trained `DecisionTreeClassifier`, fit on features retrieved via `store.get_historical_features(...)` rather than a direct `pd.read_csv()` call — this is the key behavioral change Task 4 asks for. |
| `23f1000932_Assignment_3_May2026_MLOps.ipynb` | The full workflow: installing Feast, initializing the repo, defining features, applying/materializing them, training via the offline store, and running inference via the online store. |

## What Is NOT Included (by design)

- `online_store.db` — Feast's own `.gitignore` excludes this, since it's a
  regenerable local cache, similar in spirit to excluding raw data/model
  binaries directly in Git during Week 1/2.
- Video screencast — submitted separately per assignment instructions.

## Step-by-Step Summary of What Was Done

1. **Branch setup** — Created `week_3` from `week_2`.
2. **Task 1 — Initialize Feast**: Ran `feast init iris_feature_repo`, removed the
   auto-generated example/demo files, and confirmed `feature_store.yaml` uses the
   local/SQLite provider by default.
3. **Task 2 — Define entity, source, feature view**: Copied the IRIS dataset in,
   added `iris_id` and `event_timestamp` columns, saved it as a Parquet file (the
   format Feast's `FileSource` expects), and wrote `iris_features.py` defining the
   `iris_id` entity, the `iris_source` FileSource, and the `iris_features`
   FeatureView over the four numeric columns.
4. **Task 3 — Apply & materialize**: Ran `feast apply` to register the
   definitions in the registry, then `feast materialize-incremental` to push
   feature values into the SQLite online store. Verified both `registry.db` and
   `online_store.db` were created.
5. **Task 4 — Offline training**: Built an entity dataframe of all `iris_id`s and
   timestamps, called `store.get_historical_features(...)` to pull the four
   feature columns from Feast's offline store, and trained a
   `DecisionTreeClassifier` on the result — achieving 100% evaluation accuracy on
   the held-out split.
6. **Task 5 — Online inference**: Picked two sample IDs (5 and 100), called
   `store.get_online_features(...)` to fetch their features from the online
   store, ran the trained model on them, and cross-checked the online-store
   values against the raw dataset — confirming they matched exactly, and that
   the model's predictions (`setosa` and `virginica`) matched the true species
   labels for both samples.
7. **Task 6 (BigQuery backend)** — Skipped, as it is explicitly marked optional
   in the assignment.

## Errors Encountered and How They Were Resolved

During Task 4, importing `scikit-learn` began failing with binary-incompatibility
and missing-symbol errors (`numpy.dtype size changed`, then
`ImportError: cannot import name 'get_namespace_and_device'`). This was caused by
installing `feast` pulling in a newer `numpy` version than the one `scikit-learn`
had been compiled against, leaving a partially mismatched install. This was
resolved by:
1. Uninstalling `numpy`, `scipy`, and `scikit-learn`.
2. Manually removing leftover package folders to clear any corrupted files pip
   didn't fully clean up.
3. Reinstalling `scikit-learn` fresh with `--no-cache-dir`, allowing pip to
   select a `numpy` version compatible with both Feast (`numpy>=2.0`) and
   scikit-learn.
4. **Restarting the Jupyter kernel** — this was the critical missing step
   initially, since the kernel had cached the broken module in memory even after
   the packages were reinstalled on disk.

## Cloud / Local Resources Used

- **Vertex AI Workbench** — same instance from Weeks 1–2, used to run all Feast
  commands and notebook cells.
- **Local SQLite** — used as both Feast's registry backend and online store
  backend (per Task 1's instruction that a local backend is sufficient).
- **Git** — tracks the feature repo configuration, definitions, notebook, and
  trained model for this branch.

## Author

Ayan Hussain — 23f1000932, B.S. Data Science & Applications, IIT Madras.