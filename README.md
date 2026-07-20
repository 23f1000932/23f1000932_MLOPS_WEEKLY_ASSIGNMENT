# 23f1000932_MLOPS_WEEKLY_ASSIGNMENT — Week 5

## MLflow Experiment Tracking & Model Registry for the IRIS Pipeline

This branch (`week_5`) adds **MLflow** to the IRIS pipeline — tracking every
training experiment's hyperparameters, metrics, and model artifacts, and
introducing a central Model Registry that the evaluation pipeline now pulls
from instead of DVC.

## Why This Was Needed

By Week 4, the pipeline had reproducible training, versioned data and models
(DVC), consistent feature serving (Feast), and automatic testing on every push
(CI). What was still missing: a structured way to compare different training
configurations against each other, and a single source of truth for which
model version is the "current" one. MLflow solves both — it logs every run's
parameters and results so they can be compared side-by-side, and its Model
Registry lets downstream code fetch a model by name and version instead of a
manual file path.

## Objective

1. Add hyperparameter tuning to the training loop, varying at least two
   hyperparameters across multiple runs.
2. Log every run's parameters, metrics, and trained model with MLflow.
3. Compare the logged experiments visually in the MLflow Tracking UI.
4. Remove model artifact tracking from DVC — models now live only in MLflow.
5. Update the evaluation pipeline to fetch its model from the MLflow Model
   Registry, resolved by name and version, instead of DVC or a local path.

## Repository Structure (relevant additions in week_5)

```
train_mlflow.py             # Tasks 1 & 2: hyperparameter tuning + MLflow logging
evaluate_from_registry.py   # Task 5: evaluation using a model from the registry
mlflow.db                   # MLflow's SQLite tracking store (params, metrics, run info)
mlruns/                     # MLflow's artifact store (logged models per run)
.gitignore                  # Updated to exclude model.joblib (Task 4)
README.md
```

## File-by-File: What Each One Does and Why

| File | Purpose |
|---|---|
| `train_mlflow.py` | Trains four `DecisionTreeClassifier` models, each with a different combination of `max_depth` and `min_samples_split` (the two tuned hyperparameters). Each run is wrapped in `mlflow.start_run()`, logging its hyperparameters with `mlflow.log_param`, its accuracy/precision/recall with `mlflow.log_metric`, and the trained model itself with `mlflow.sklearn.log_model`. |
| `evaluate_from_registry.py` | Loads a model directly from the MLflow Model Registry using `models:/iris_classifier/1` — a name-and-version URI — rather than reading any local file or DVC-tracked artifact. Runs it against the evaluation split and prints accuracy, precision, and recall to confirm the fetched model behaves identically to its original training run. |
| `mlflow.db` | SQLite database MLflow uses as its tracking backend — stores every run's parameters, metrics, and metadata so they can be queried and compared in the UI. |
| `mlruns/` | MLflow's artifact store — contains the actual serialized model files for each logged run. |
| `.gitignore` | Updated to exclude `model.joblib` entirely, since models are no longer tracked as loose files or through DVC — MLflow is now the exclusive model store. |

## What Is NOT Included (by design)

- `model.joblib` — no longer committed to Git or tracked by DVC. The trained
  model lives only inside MLflow's artifact store (`mlruns/`) and the
  registry.
- `model.joblib.dvc` — removed in Task 4, since DVC now tracks data only.
- Video screencast — submitted separately per assignment instructions.

## Step-by-Step Summary of What Was Done

1. **Branch setup** — Created `week_5` from `main`, installed MLflow.
2. **Task 1 — Hyperparameter tuning**: Wrote a loop in `train_mlflow.py`
   trying four combinations of `max_depth` (2–5) and `min_samples_split`
   (2–6). Confirmed the configurations genuinely produced different results —
   accuracy ranged from 0.889 (shallowest tree) to 0.944 (deeper trees).
3. **Task 2 — MLflow logging**: Wrapped each training iteration in
   `mlflow.start_run()`, logging hyperparameters, evaluation metrics, and the
   trained model as an artifact. Used a SQLite-backed tracking store
   (`sqlite:///mlflow.db`), since the newer MLflow version deprecated the
   plain file-based backend.
4. **Task 3 — Compare in the UI**: Launched `mlflow ui`, viewed all four
   logged runs in the Runs table, and used the built-in Parallel Coordinates
   Plot to compare `max_depth` and `min_samples_split` against precision,
   accuracy, and recall across multiple runs side-by-side.
5. **Task 4 — Remove model from DVC**: Ran `dvc remove model.joblib.dvc` to
   stop DVC from tracking the model, then ensured the raw `model.joblib`
   binary was excluded from Git entirely (added to `.gitignore`) rather than
   committed directly — keeping MLflow as the single source of truth for
   models, with DVC continuing to track only `data/iris.csv.dvc`.
6. **Task 5 — Fetch from registry for evaluation**: Registered the
   best-performing run's model (`sedate-panda-579`, accuracy 0.944) into the
   MLflow Model Registry under the name `iris_classifier`, version 1. Wrote
   `evaluate_from_registry.py`, which loads the model via
   `mlflow.sklearn.load_model("models:/iris_classifier/1")` and confirmed its
   evaluation output (accuracy 0.944, precision 0.952, recall 0.949) exactly
   matches the metrics originally logged for that run.
7. **Task 6 (MLflow in CI)** — Skipped, as it is explicitly marked optional in
   the assignment.

## Errors Encountered and How They Were Resolved

1. **Filesystem tracking backend deprecated**: The first attempt to run
   `mlflow.set_tracking_uri("file:./mlruns")` raised an
   `MlflowException`, since this MLflow version has moved the plain
   file-based store into maintenance mode. Resolved by switching to a
   SQLite-backed store instead: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`.
2. **MLflow UI unreachable through the Workbench proxy ("Invalid Host header
   — possible DNS rebinding attack")**: MLflow's built-in security middleware
   only trusts requests arriving under `localhost` by default, and rejected
   requests coming through the Workbench proxy hostname. Resolved by starting
   the server with explicit `--allowed-hosts` and `--cors-allowed-origins`
   flags matching the exact Workbench proxy URL, which allowed both the page
   load and its background API calls to succeed.
3. **Raw model binary briefly committed to Git after removing it from DVC**:
   After running `dvc remove model.joblib.dvc`, a plain `git add -A` picked
   up the now-untracked `model.joblib` file and committed it directly to
   Git — the opposite of the intended outcome. Resolved with
   `git rm --cached model.joblib` followed by adding it to `.gitignore`, so
   the model exists only inside MLflow going forward.

## Cloud / Tooling Resources Used

- **Vertex AI Workbench** — same instance from all previous weeks, used to run
  training, the MLflow UI, and the registry evaluation script.
- **MLflow (local, SQLite-backed)** — experiment tracking and model registry,
  running entirely within the Workbench instance, accessed via its proxy URL.
- **DVC** — continues to version `data/iris.csv.dvc` only, no longer tracks
  model artifacts.
- **Git** — tracks code, MLflow's tracking database and artifact store, and
  this README.

## Author

Ayan Hussain — 23f1000932, B.S. Data Science & Applications, IIT Madras.