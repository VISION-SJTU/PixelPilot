# Reasoning data: generation, conditioning, selection

The teacher prompt is retained from `create_vision_cot_data.py`. It supplies the
image, ego speed, ground-truth front-view boxes and action to a teacher and asks
for a concise explanation grounded in the relevant objects. The original code
used Qwen-VL-Max through the DashScope-compatible endpoint despite naming a
function `gpt4o_query`. The model id is explicit in the reorganized tool because
an alias such as `qwen-vl-max-latest` can change over time.

This release adds resumable generation and explicit conditioning/selection glue.
It does not claim to have recovered the exact historical failure-case mining
cohort or conditioning implementation. Generation alone does not establish that
a reasoning sample improves the student policy.

## Candidate generation

Prepare a unique integrated annotation subset of **training failure cases**.
Do not mine or add validation/test examples to SFT training. Preview offline:

```bash
python -m pixelpilot.reasoning --annotations data/failure_cases.json \
  --image-root data --model YOUR_TEACHER_MODEL \
  --output outputs/reasoning_candidates.jsonl --dry-run
```

The preview makes zero API calls and writes no candidate file. To deliberately
request teacher generation, configure `DASHSCOPE_API_KEY` and run the same command
without `--dry-run`. It sends the selected images/prompts to the configured
teacher endpoint. Add `--resume` after an interrupted generation. Model ids must
match and already completed samples are not requested again. Secrets are read
only from the environment and are never stored in outputs.

Candidates are plain reasoning text keyed by frame id. Output-control tags are
rejected to avoid accidentally injecting perception/action/answer structures.
A successful request is still a candidate, not an accepted training record.
No API request was made during this repository preparation.

## Paired student inference

Run the same checkpoint and decoding configuration on the same training cases,
then run conditioning with the candidate file:

```bash
python -m pixelpilot.infer --model checkpoints/pixelpilot-7b-sft \
  --annotations data/failure_cases.json --image-root data --greedy \
  --output outputs/baseline.jsonl
python -m pixelpilot.infer --model checkpoints/pixelpilot-7b-sft \
  --annotations data/failure_cases.json --image-root data --greedy \
  --reasoning-candidates outputs/reasoning_candidates.jsonl \
  --output outputs/conditioned.jsonl
```

Conditioning appends an explicit `Additional visual reasoning` field to the
user input and evaluates only the candidate subset. This is the release's new
conditioning interface; it is not asserted to equal an absent historical
prefill implementation. Use an exactly matching baseline subset and decoding
settings. `--greedy` removes sampling variation for this diagnostic; it does not
replace the paper's benchmark decoding defaults.

Lift both outputs independently with the same calibration and `--frame PLANNING`,
and prepare independent metric GT for the candidate subset using the generation
outputs. The selection tool requires identical ids across all paired files.

## Strict selection and SFT attachment

```bash
python -m pixelpilot.select_reasoning \
  --baseline-actions outputs/baseline.jsonl --conditioned-actions outputs/conditioned.jsonl \
  --baseline-trajectories outputs/baseline_metric.json \
  --conditioned-trajectories outputs/conditioned_metric.json \
  --ground-truth data/failure_cases_metric_gt.json \
  --candidates outputs/reasoning_candidates.jsonl \
  --annotations data/failure_cases.json --output data/selected_reasoning_sft.json
```

A candidate is accepted only when the baseline action is wrong, conditioning
predicts the complete correct path/speed action set, and full-horizon mean metric
L2 decreases. Samples with incomplete six-waypoint GT are counted and excluded.
Frame, GT-label and sample-set mismatches fail. Coordinates are not compared in
pixel units as a substitute for metric trajectory quality.

Accepted reasoning is inserted between perception and action in the original
assistant target; original perception/action/trajectory labels are preserved.
The prompt is updated to request the think block. The selector leaves original
records untouched. An empty accepted set is saved as an empty array; omit it
from SFT preparation rather than inventing reasoning data.

For a non-empty selection, add it to the recovered multi-task corpus:

```bash
python -m pixelpilot.prepare --format sft \
  --train-json data/annotations/train_sft.json \
  --extra-train-json data/selected_reasoning_sft.json \
  --val-json data/annotations/val_integrated.json \
  --image-root data --output-dir data/nuscenes_sft_with_reasoning
bash scripts/train_sft.sh --model checkpoints/pixelpilot-7b-sft \
  --dataset-dir data/nuscenes_sft_with_reasoning \
  --output-dir checkpoints/pixelpilot-7b-sft-with-reasoning
```

The number/weight of selected reasoning examples is explicit in the resulting
corpus. This release does not infer the final paper's reasoning-data mixture
from an experiment filename.
