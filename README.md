# Week 7 Screencast Script (5–7 minutes)

Plain-English narration script. Read naturally, don't rush — pause on each
dashboard/terminal long enough for it to actually be visible on screen.
Checklist points A–J are marked inline so you can confirm coverage while
recording.

---

## Intro (30 seconds)

"Hi, this is Ayan, roll number 23f1000932, and this is my Week 7 submission
for the MLOps Weekly Assignment — Stress Testing, Observability, and
Scaling the IRIS pipeline.

Last week I got the IRIS prediction API running live on Google Kubernetes
Engine. This week is about answering a harder question: what actually
happens when real traffic hits it? I'll walk through five tasks — running
high-concurrency load tests with wrk, configuring Kubernetes autoscaling,
watching it scale live, monitoring it through GCP's dashboards, and finally
taking autoscaling away to find the bottleneck."

**[Checklist A — introduction and context, done]**

---

## Part 1: CI/CD integration (45 seconds)

*(Screen: GitHub Actions, workflow run page)*

"First, Task 1 — I extended my existing CD pipeline so stress testing runs
automatically. After the image builds and deploys to GKE, the workflow now
installs wrk on the runner, grabs the live service's external IP, and
fires a quick load test against it — right here in the Actions tab. You
can see this run completed successfully in about four and a half minutes,
with the stress test step passing at the end."

*(Show the green checkmark and expand the 'Run automated stress test' step
briefly)*

**[Checklist B — CI/CD stress test integration, done]**

---

## Part 2: Manual load test with wrk (1 minute)

*(Screen: Terminal, wrk running)*

"Task 2 is a manual, heavier load test — over a thousand concurrent
connections against the live `/predict` endpoint, using this wrk script
that sends a proper JSON POST request instead of wrk's default GET."

*(Show `cat post_predict.lua` briefly, then run or show the saved output)*

"Here's the result — about 192 requests per second, with a median latency
of around 650 milliseconds, and some timeouts starting to show up. This
was before autoscaling was configured, so one pod is absorbing all of
this alone."

**[Checklist C — wrk high-concurrency test (>1000 connections), done]**
**[Checklist D — recorded req/sec, latency, error count, done]**

---

## Part 3: Configuring and observing the HPA (1.5 minutes)

*(Screen: Terminal — three tabs: wrk, `kubectl get pods -w`, `kubectl get hpa -w`)*

"For Task 3, I configured a Horizontal Pod Autoscaler — minimum 1 pod,
maximum 3, targeting 50% CPU utilization. I've got three terminals open:
one to fire the load test, and two watching pods and the HPA live."

*(Trigger the load test, then switch to the watch tabs)*

"Watch this — CPU jumps well past the 50% target, up to about 164%, and
within about ten seconds Kubernetes spins up two more pods, going from 1
replica to 3. That's the autoscaler doing exactly what it's supposed to."

**[Checklist E — HPA configured with min/max replicas, done]**
**[Checklist F — autoscaling observed live via kubectl, done]**

---

## Part 4: GCP Monitoring and Logging (1.5 minutes)

*(Screen: Browser — GCP Console, Workloads → iris-api → Observability tab)*

"Task 4 is about correlating that scaling event with GCP's own dashboards,
not just kubectl. Here's the CPU Request percent graph, broken down per
pod — you can clearly see the spike lining up with when I ran the load
test, and it's spread across all three pod names once the HPA scaled out."

*(Switch to Logs Explorer tab)*

"And here's Cloud Logging, filtered to just the iris-api container. During
the load test this streams in a flood of POST /predict 200 responses in
real time — this confirms requests really were reaching and being handled
by the pods, not just that CPU went up for some other reason."

**[Checklist G — Cloud Monitoring dashboard shown, done]**
**[Checklist H — Cloud Logging filtered by pod/container shown, done]**

---

## Part 5: Constrained scaling and bottleneck analysis (1.5 minutes)

*(Screen: Terminal, then back to browser for comparison)*

"Finally, Task 5 — I wanted to see what happens without autoscaling to
rely on. I patched the HPA down to a maximum of 1 pod, confirmed it scaled
back down, and then doubled the load — 2000 connections instead of 1000."

*(Show the task5 wrk output)*

"And here's the interesting part — even though I sent twice the load,
throughput actually went down, from about 192 requests per second to 163.
Latency got worse across the board — median jumped from 649 milliseconds
to 824. Fewer total requests completed in the same 30 seconds, despite
more being sent.

That's the bottleneck: with only one pod and no room to scale out, CPU is
a hard ceiling. Extra concurrent connections don't get processed faster —
they just queue up behind the same fixed compute capacity, so latency
climbs instead of throughput increasing. The Cloud Monitoring CPU graph
for this run shows a much sharper, sustained spike on the single pod
compared to Task 3's load spread across three."

**[Checklist I — maxReplicas=1 constrained test with 2000 connections, done]**
**[Checklist J — bottleneck identified and explained with comparison data, done]**

---

## Closing (15 seconds)

"That covers all five tasks — automated stress testing in CI/CD, manual
high-concurrency load testing, live autoscaling, GCP dashboard monitoring,
and a constrained-scaling bottleneck analysis. All the code, results, and
the HPA config are committed on the week_7 branch. Thanks for watching."

---

**Total estimated runtime: ~6.5 minutes**