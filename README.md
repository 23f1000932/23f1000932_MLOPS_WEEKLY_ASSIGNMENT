# 23f1000932_MLOPS_WEEKLY_ASSIGNMENT

## Week 1 — IRIS ML Pipeline on Vertex AI

This repository contains the Week 1 MLOps assignment: an end-to-end machine learning
pipeline for IRIS classification, built and executed on Google Cloud's Vertex AI
platform, using Google Cloud Storage (GCS) for data and artifact management.

## Objective

Build a bare-bone ML pipeline that:
1. Fetches training data from Google Cloud Storage.
2. Trains an IRIS classifier on Vertex AI Workbench.
3. Stores model artifacts and logs back to GCS, organized by execution timestamp.
4. Runs a separate inference step that retrieves the trained model from GCS and
   evaluates it on a held-out evaluation set.

## Repository Contents

| File | Purpose |
|---|---|
| `23f1000932_Assignment_1_May2026_MLOps.ipynb` | Main notebook containing all pipeline code: GCS bucket setup, data upload, model training, and inference, executed twice to produce two independent timestamped runs. |
| `iris.csv` | Raw, unmodified IRIS dataset used as the source input for the pipeline (150 records, 4 features, 3 species classes). |
| `train_log.txt` | Sample output log from a training run, recording model type, hyperparameters, and number of training samples used. |
| `inference_log.txt` | Sample output log from an inference run, recording the evaluation accuracy achieved by the corresponding trained model. |
| `README.md` | This file — explains the purpose of each item in the repository. |

## What is NOT included (by design)

- **Trained model files** (`model.joblib`) — these are binary execution artifacts and
  are stored in Google Cloud Storage instead, under
  `gs://23f1000932-mlops-week1/artifacts/<timestamp>/`.
- **Dataset splits used for training** (`train.csv`, `eval.csv`) — these are also
  stored in GCS under `gs://23f1000932-mlops-week1/data/`, not committed to this repo.
- **Video screencast** — submitted separately per assignment instructions, not part
  of this repository.

## Pipeline Summary

1. **GCS Bucket Setup** — Created bucket `23f1000932-mlops-week1` in `us-central1`.
2. **Data Preparation** — Split `iris.csv` into an 80% training set and 20%
   evaluation set (stratified by species), uploaded both to
   `gs://23f1000932-mlops-week1/data/`.
3. **Training** — Trained a `DecisionTreeClassifier` (max_depth=3) using data fetched
   from GCS; saved model and training log to a timestamped folder under
   `artifacts/` in GCS.
4. **Inference** — Fetched the trained model back from GCS, ran predictions on the
   evaluation set, and uploaded an inference log with accuracy results.
5. **Repeated Execution** — Steps 3 and 4 were executed a second time, producing a
   second, independent timestamped artifacts folder in GCS — demonstrating
   traceability across multiple pipeline runs.

## Cloud Resources Used

- **Vertex AI Workbench** — managed Jupyter environment for running the pipeline.
- **Google Cloud Storage** — storage for input data and output artifacts.
- **IAM** — Storage Admin role granted to the Workbench's default Compute Engine
  service account to enable GCS read/write access.

## Author

Ayan Hussain — 23f1000932, B.S. Data Science & Applications, IIT Madras.
