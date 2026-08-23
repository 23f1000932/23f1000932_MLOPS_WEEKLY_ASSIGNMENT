# Week 10 — From MLOps to LLMOps: Fine-Tuning Gemini on the IRIS Pipeline

**Branch:** `week_10`
**Roll number:** 23f1000932
**Base model:** `gemini-3.5-flash`

## Why this week matters

Every prior week assumed a traditional ML model — you write the training
code, you own every parameter, you control the exact output format. LLMs
change that: you start from a foundation model someone else pretrained,
adapt it with a comparatively tiny dataset, and the interface is
text-in/text-out rather than a fixed schema. This week applies the same
operational discipline (versioning, evaluation, comparison) to that
different lifecycle — fine-tuning Gemini on two representations of the
same IRIS classification task and measuring what changes.

## What was built (file by file)

| File | Purpose |
|---|---|
| `prepare_v1_raw.py` | Converts `data/iris.csv` into JSONL with raw feature values as the input text (`"sepal_length: 5.1, ..."`) and the bare species name as output. |
| `prepare_v2_description.py` | Converts the same data into natural-language JSONL (`"A flower specimen has a sepal length of..."`) with a full-sentence output (`"This is Iris setosa."`). |
| `split_train_eval.py` | Splits both JSONL versions 80/20 using the same stratified seed as every prior week, so v1 and v2 share identical rows in train vs. eval — required for a fair comparison. |
| `convert_to_gemini_format.py` | Converts the assignment's `input_text`/`output_text` schema into the `contents`/`role`/`parts` structure Vertex AI's current Gemini tuning actually requires. |
| `evaluate_models.py` | Calls both tuned model endpoints against their held-out eval sets and computes format compliance (strict), accuracy on compliant responses, loose accuracy (correct species stated anywhere), and per-class precision/recall. |
| `data/iris_v1_*.jsonl`, `data/iris_v2_*.jsonl` | Raw and Gemini-format train/eval files for both representations, DVC-tracked and uploaded to `gs://23f1000932-mlops-week1/week10-llmops/`. |
| `eval_results_v1_raw.csv`, `eval_results_v2_description.csv`, `evaluation_summary.csv` | Per-row and summary evaluation output. |

## Step-by-step summary

### Task 1 & 2 — Prepare v1 (raw) and v2 (natural language) formats
Wrote both conversion scripts exactly matching the assignment's example
formats, split each into 144 train / 36 eval rows (same row indices
across both versions), and uploaded all four files to GCS.

### Task 3 — Fine-tune two Gemini model versions on Vertex AI
Submitted two supervised fine-tuning jobs via Vertex AI Studio, both on
`gemini-3.5-flash`, both with identical hyperparameters (3 epochs,
learning rate multiplier 1.0, default adapter size) — the only variable
between them is the training data representation. Both succeeded:

| Model | Validation accuracy (during tuning) |
|---|---|
| v1 (raw features) | ~0.851 |
| v2 (natural language) | ~0.686 |

### Task 4 — Evaluate & compare model versions
Ran both tuned endpoints against their 36-row eval sets. Two metrics
were tracked: **strict format compliance** (does the response consist of
exactly the species name, per the assignment's definition) and **loose
accuracy** (is the correct species stated anywhere in the response, via
the pattern the model consistently used: "classified as **Iris
<species>**"). Results, identical for both models:

| Metric | v1 (raw) | v2 (description) |
|---|---|---|
| Format compliance (strict) | 0.000 | 0.000 |
| Loose accuracy | 1.000 | 1.000 |
| Per-class precision/recall (all 3 classes) | 1.000 / 1.000 | 1.000 / 1.000 |

**Which version performed better, and why:** neither — both tied
exactly. Both models correctly identify the species in every single
eval example when read for content, and both fail every single
strict-format check. The bottleneck this week wasn't the training data
representation at all; it was Gemini's default explanatory response
style overriding the terse single-word completions the training data
demonstrated. During tuning itself, v1 showed meaningfully higher
validation accuracy (0.851 vs 0.686) than v2 — suggesting the raw
feature format was easier for the model to fit during training — but
that gap disappeared entirely once both were evaluated on actual
correctness rather than the tuning job's own accuracy metric, since both
ultimately reason their way to the right answer regardless of which
input format they were trained on.

### Task 5 (optional) — Automated evaluation in CI
Not implemented this week. Given the debugging required just to get a
single evaluation run working correctly (see below), and that each
evaluation run makes 72 real billed inference calls against live
endpoints, adding this as an automatic on-every-push CI step was judged
not worth the ongoing cost for a course assignment pipeline that isn't
actually being iterated on daily.

## Errors encountered and fixes

- **First tuning job failed instantly**: `Converting from
  'VertexTextBison' to 'GenerateContent' dataset format is currently not
  supported for this model.` The assignment's example JSONL format
  (`input_text`/`output_text`) matches the older PaLM/Bison tuning
  schema, not what current Gemini models on Vertex AI expect. Fixed by
  writing `convert_to_gemini_format.py` to transform the data into the
  `contents: [{role, parts: [{text}]}]` structure Gemini tuning actually
  requires, while keeping the original files as-is since they match the
  assignment's literal spec.
- **Extensive endpoint-calling failures during Task 4**: the
  `google-cloud-aiplatform` SDK's `GenerativeModel` and `Endpoint`
  classes both rejected every location value tried (`us-central1`,
  `us`, `global`) with a mix of `ValueError` (unsupported region),
  `400 BadRequest` (wrong location for this endpoint), and `404 NotFound`
  (right location, but wrong API surface) — despite the tuning job's own
  API record confirming the endpoint's true location as `us` and the ID
  as correct. Root cause: tuned Gemini endpoints deployed via Vertex AI
  Studio are called through the newer `google-genai` SDK
  (`from google import genai`, `vertexai=True`), not the classic
  `aiplatform.Endpoint`/`GenerativeModel` path — confirmed by pulling the
  exact working code sample from the Studio "Test" panel's "Code" button
  rather than continuing to guess host/region combinations. Fixed by
  rewriting `evaluate_models.py` to use `genai.Client`.
- **0% format compliance investigated, not "fixed"**: initial evaluation
  runs returned exactly 0.000 accuracy for both models across the board,
  which looked like a bug. Inspecting the raw responses showed the model
  was answering correctly every time, just wrapped in a full markdown
  explanation ("Based on the measurements provided, this flower is
  classified as **Iris setosa**...") instead of the bare label the
  training data taught. This is a real, intended LLM-specific failure
  mode per the assignment's own definition of format compliance, not a
  bug — added a secondary "loose accuracy" metric (regex-matching the
  model's stated conclusion) to separately measure underlying
  correctness alongside the strict compliance number the assignment
  requires.

## How to reproduce

```bash
# From the repo root, on the Vertex AI Workbench terminal
python3 prepare_v1_raw.py
python3 prepare_v2_description.py
python3 split_train_eval.py
python3 convert_to_gemini_format.py

# Upload to GCS, then submit both tuning jobs via the Vertex AI Studio
# console (Model details: gemini-3.5-flash, us-central1, 3 epochs,
# learning rate multiplier 1.0; Tuning dataset: existing GCS files)

# Once both jobs succeed, fill in their endpoint resource names in
# evaluate_models.py, then:
python3 evaluate_models.py
```
