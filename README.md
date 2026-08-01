# Week 7 — Stress Testing, Observability & Scaling the IRIS Pipeline

**Branch:** `week_7`
**Roll number:** 23f1000932
**Live API:** `http://35.254.178.209/predict` (GKE LoadBalancer, cluster `iris-cluster`)

## Why this week matters

Week 6 got the IRIS API live on GKE, but a working deployment isn't the same
as a *production-ready* one. This week answers the questions that matter
once real traffic shows up: how many concurrent users can the API handle
before it slows down or errors out? Does Kubernetes actually scale the way
it's supposed to under load? And when scaling is *not* available, what
breaks first?

We used `wrk` to generate high-concurrency HTTP load, configured a
Kubernetes Horizontal Pod Autoscaler (HPA) to scale Pods automatically,
watched it happen live through `kubectl` and GCP's Cloud Monitoring /
Cloud Logging dashboards, and then deliberately took autoscaling away to
see where the single-Pod bottleneck shows up.

## What was built (file by file)

| File | Purpose |
|---|---|
| `post_predict.lua` | wrk script — sends a POST request to `/predict` with a sample IRIS JSON payload and the correct `Content-Type` header, since wrk defaults to GET. |
| `hpa.yaml` | Exported HPA config (`kubectl get hpa iris-api -o yaml`). Final committed version has `minReplicas: 1`, `maxReplicas: 3`, target CPU 50%. |
| `task2_wrk_results.txt` | Task 2 evidence — 1000 connections, no autoscaling limit yet applied (before HPA existed). |
| `task3_wrk_results.txt` | Task 3 evidence — 1000 connections with HPA active (`max=3`), captured while watching Pods scale 1 → 3. |
| `task4_wrk_results.txt` | Task 4 evidence — same load test re-run while GCP Cloud Monitoring and Cloud Logging were open, to correlate wrk's numbers with GCP's own dashboards. |
| `task5_wrk_results.txt` | Task 5 evidence — HPA constrained to `maxReplicas: 1`, load increased to 2000 connections, to force and observe the single-Pod bottleneck. |
| `.github/workflows/cd.yml` | Extended with three new steps after `Deploy to GKE`: install `wrk` on the runner, fetch the LoadBalancer's external IP, then run an automated stress test against the live API on every push to `week_7`/`main`. |

## Step-by-step summary

### Task 1 — Automated stress test in CI/CD
Added three steps to `cd.yml` right after the existing `kubectl rollout
status` step: install `wrk` from source on the GitHub Actions runner, read
the Service's external IP via `kubectl get svc -o jsonpath`, then run
`wrk -t4 -c200 -d15s` against the live `/predict` endpoint. Used a lighter
load (200 connections, 15s) than the manual tests since this runs on every
push and needs to stay fast — the heavy 1000/2000-connection stress tests
are already covered manually in Tasks 2–5. Confirmed working end-to-end
in GitHub Actions (run #7, Success, 4m 40s).

### Task 2 — Simulate high-concurrency traffic
Installed `wrk` on the Vertex AI Workbench instance, wrote
`post_predict.lua`, ran a small sanity check (10 connections) to confirm
the API responds correctly to the script's POST body, then ran the real
test: `wrk -t4 -c1000 -d30s --latency`. Result: ~192 req/sec, 50th
percentile latency 649ms, some socket errors/timeouts at this concurrency
— expected, since no autoscaling was configured yet at this point.

### Task 3 — Configure the Horizontal Pod Autoscaler
Confirmed the Deployment already had `resources.requests.cpu: 500m` set
(required for HPA to calculate utilization). Created the HPA with
`kubectl autoscale deployment iris-api --cpu=50% --min=1 --max=3`. Opened
two extra terminal tabs running `kubectl get pods -w` and
`kubectl get hpa -w`, then re-ran the Task 2 load test. Watched CPU climb
to 164% (well over the 50% target), triggering the HPA to scale from 1 Pod
to 3 within about 10 seconds. Saved the config as `hpa.yaml`.

### Task 4 — Monitor with GCP Cloud Monitoring & Cloud Logging
With Kubernetes Engine → Workloads → `iris-api` → Observability open in
one browser tab and Cloud Logging (Logs Explorer, filtered to
`resource.type="k8s_container"` and `container_name="iris-api"`) open in
another, re-ran the load test. The CPU Request % Used graph showed a clear
spike across all 3 Pods matching the load test window, and Logs Explorer
streamed a burst of `POST /predict HTTP/1.1" 200` entries in real time —
confirming the dashboards correlate directly with the load test, not just
`kubectl` output.

### Task 5 — Observe bottlenecks under constrained scaling
Patched the HPA down to `maxReplicas: 1`
(`kubectl patch hpa iris-api -p '{"spec":{"maxReplicas":1}}'`), confirmed
it scaled back down to exactly 1 Pod, then ran
`wrk -t4 -c2000 -d30s --latency` — double the concurrency of Task 3, but
now with zero scale-out capacity.

**Comparison — Task 3 (max=3, 1000 conns) vs Task 5 (max=1, 2000 conns):**

| Metric | Task 3 | Task 5 |
|---|---|---|
| Requests/sec | 191.83 | 163.18 |
| 50th percentile latency | 649ms | 824ms |
| 90th percentile latency | 1.30s | 1.41s |
| Total requests completed (30s) | 5,774 | 4,903 |
| Timeouts | 286 | 238 |

Despite offering **double** the concurrent load, Task 5 completed **fewer**
total requests with **worse** latency at every percentile. This is the
textbook throughput-plateau signature: once the single Pod's CPU request
(500m) is saturated, adding more concurrent connections doesn't increase
throughput — it just queues more requests behind the same fixed compute
capacity, so latency climbs instead. With `maxReplicas: 3` in Task 3, the
extra load could be spread across additional Pods; with `maxReplicas: 1`
in Task 5, CPU is the hard ceiling and there's no escape valve. Cloud
Monitoring's CPU graph confirmed this — Task 5's single-Pod CPU spike was
sharper and more sustained than Task 3's spread-across-3-Pods graph.

## Errors encountered and fixes

- **`kubectl get svc` initially failed** with
  `executable gke-gcloud-auth-plugin not found`. Fixed by installing
  `google-cloud-cli-gke-gcloud-auth-plugin` via `apt-get` and setting
  `USE_GKE_GCLOUD_AUTH_PLUGIN=True`, then re-running
  `gcloud container clusters get-credentials`.
- **HPA briefly showed `REPLICAS: 0`** right after creation via
  `kubectl autoscale`. Not a real issue — the Deployment itself still
  showed `1/1 Running`; the HPA's reported replica count just lagged a few
  seconds behind actual cluster state and corrected itself.
- **Logs Explorer's default 5-minute time window** initially caught almost
  no relevant logs since the load test had already finished. Fixed by
  widening the range to "Last 1 hour."
- **A recurring `sklearn` `UserWarning`** ("X does not have valid feature
  names...") appears on every prediction request because `app.py` passes a
  plain Python list to `model.predict()` rather than a DataFrame with
  named columns. Harmless — predictions still return `200` — but it
  roughly doubles the log line count per request and gets tagged
  `severity: ERROR` in Cloud Logging simply because Python's
  `warnings.warn()` writes to stderr, which GKE's default logging pipeline
  labels as ERROR regardless of actual content. Not fixed this week since
  it doesn't affect functionality, but worth noting so the "ERROR" count
  in Logs Explorer isn't mistaken for real failures.

## How to reproduce

```bash
# From the repo root, on the Vertex AI Workbench terminal
wrk -t4 -c1000 -d30s --latency -s post_predict.lua http://35.254.178.209/predict

# Watch scaling live (separate terminal tabs)
kubectl get pods -w
kubectl get hpa -w

# Constrain and re-test (Task 5 scenario)
kubectl patch hpa iris-api -p '{"spec":{"maxReplicas":1}}'
wrk -t4 -c2000 -d30s --latency -s post_predict.lua http://35.254.178.209/predict

# Restore full autoscaling
kubectl patch hpa iris-api -p '{"spec":{"maxReplicas":3}}'
```