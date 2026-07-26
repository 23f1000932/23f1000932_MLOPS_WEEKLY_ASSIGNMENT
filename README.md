# Week 6 — Docker, Artifact Registry & GKE (IRIS API)

## Why this week's setup

The IRIS classifier is now served as a real API instead of a notebook prediction cell. A Flask app wraps the model,
Docker packages that app into a portable image, Artifact Registry stores versioned images, and GKE runs the
container as a live, restart-safe Pod behind a LoadBalancer. GitHub Actions ties it together: every push to
`week_6` builds a new image, pushes it, and redeploys it — no manual `docker push` or `kubectl apply` required.

## Files in this branch

| File | Purpose |
|---|---|
| `app.py` | Flask API with a `/predict` endpoint; loads the exported model at startup |
| `export_model.py` | Pulls the best model from the MLflow registry and writes it to `model_export/model.joblib` as a plain file |
| `model_export/model.joblib` | The exported model the container actually serves — no MLflow runtime dependency |
| `Dockerfile` | Builds the API image on `python:3.12-slim` |
| `requirements.txt` | `flask`, `scikit-learn`, `joblib` |
| `k8s/deployment.yaml` | Kubernetes Deployment + LoadBalancer Service for the API |
| `.github/workflows/cd.yml` | CI/CD: builds the image, pushes to Artifact Registry, deploys to GKE |

## Step-by-step summary

- **Task 1** — Created the `week_6` branch.
- **Task 2** — Built `app.py` and `Dockerfile`; tested locally with `docker build` + `docker run`, confirmed
  `/predict` returns a real prediction before touching CI at all.
- **Task 3** — Created a dedicated service account for CI/CD with only the roles it needs (Artifact Registry writer,
  GKE deployer), stored its key as the `GKE_SA_KEY` GitHub secret.
- **Task 4** — Wrote `cd.yml`: authenticates with `GKE_SA_KEY`, builds the image, tags it with the commit SHA, and
  pushes it to the `iris-api-repo` Artifact Registry repo.
- **Task 5** — Created an Autopilot GKE cluster (`iris-cluster`, `us-central1`), wrote `k8s/deployment.yaml`, and
  extended `cd.yml` to substitute the commit SHA into the manifest and `kubectl apply` it after every successful
  push. Verified with `kubectl get pods` / `kubectl get svc` and a live `curl` call against the LoadBalancer IP.
- **Task 6 (optional)** — Skipped.

## Errors encountered and how they were fixed

1. **IAM permission errors (`container.clusters.create`, Artifact Registry admin).** My personal account had broad
   project-level roles but not these specific ones. Fixed by explicitly granting `roles/container.admin` and the
   Artifact Registry admin role to my account.
2. **Dockerfile edit didn't persist.** An earlier edit to switch the `COPY` target from `mlflow.db`/`mlruns/` to
   `model_export/` was made in the editor but never actually saved, so a later build silently used the old
   Dockerfile and failed with a missing-file error inside the container. Caught by `cat`-ing the file before
   assuming the code itself was wrong.
3. **`.gitignore` silently excluded a needed file.** An unanchored `model.joblib` rule added in Week 5 (to stop DVC
   from tracking the model) also matched the nested `model_export/model.joblib` path, so it was never committed —
   the Docker build in CI failed with `"model_export": not found` even though it worked locally. Fixed by scoping
   the rule to the repo root (`/model.joblib`) and force-adding the file.

## Verifying the deployment

```bash
kubectl get pods
kubectl get svc iris-api-service
curl http://<EXTERNAL-IP>/predict -X POST -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Cleanup (after recording the screencast)

```bash
gcloud container clusters delete iris-cluster --region=us-central1
```
GKE clusters bill continuously while running, so this is deleted once the screencast evidence is captured.
