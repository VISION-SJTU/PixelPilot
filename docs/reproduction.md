# Reproduction status

The source release now uses the additionally supplied `LLaMA-Factory-0.9.4`
research folder. Its internal version is **0.9.4.dev0**. The initial release's
missing-SFT and missing-PlanningMetric statements are superseded by this version.
No paper score is inferred from filenames or historical logs.

## Recovered components

| Component | Current state |
| --- | --- |
| SFT framework | Actual supplied source under `LLaMAFactory/`, including `train/sft/workflow.py` and Trainer |
| 7B SFT configurations | Three supplied variants retained under `configs/historical/`; the two-epoch `p_a_t_pat` variant is the portable default |
| Annotation generation | Selected multi-task projection/labeling script reorganized as `pixelpilot.generate`; fixed seed added |
| Original info files | Not included; validation info schema checked across 6019 records during preparation, before the original research snapshots were removed from the repository |
| Teacher reasoning | Original prompt and Qwen-VL-Max endpoint pattern recovered; resumable wrapper added |
| Conditioning and selection | New explicit glue for candidate conditioning, action correction and metric-L2 improvement; original complete selection implementation was not recovered |
| GRPO | Supplied Seg-Zero-based implementation, with historical reward retained |
| Collision/map evaluator | Recovered `PlanningMetric` and map API included; strict id-aligned wrapper added |
| Model weights | Not bundled; no verified release download was supplied |
| Raw nuScenes-to-info converter | Not recovered; generation starts from existing OmniDrive-style infos and metric JSON |
| Waymo / Bench2Drive / CARLA | Outside the provided nuScenes release |

The default SFT recipe is **a recovered experiment**, not a certified match to
the final paper checkpoint. It uses 720000 image pixels, cutoff 8192, full SFT,
ZeRO-3, learning rate 5e-6, two epochs, batch size 2/device and accumulation 8.
The source dataset name encodes `0.5p_0.5a_1t_1pat`. The other 7B configurations
use one epoch and different task mixtures. They are kept as references, rather
than silently merged into the default.

## Data choices and release changes

The selected source script computes perception, action, trajectory and integrated
labels. It samples perception/action tasks with probability 0.5, filters small
boxes and caps boxes at 30. The release retains those equations and thresholds,
adds a deterministic seed, explicit input/output paths and field validation,
and emits unique integrated records plus independent metric GT for evaluation.
A historical unseeded sample list cannot be reconstructed from its ratio alone.

The recovered `fb_all_cot.py` requests thinking in the prompt but does not itself
insert generated thinking into the assistant target. Its train/val speed
thresholds also differ. It is not substituted for the selected multi-task
baseline. The release's reasoning workflow is separately documented and tested.

Native collages and explicit model pixel limits are supported. Newly encoded
JPEGs and changed resize ordering can differ from old image bytes. The recovered
annotation mapping and generic lifting's discrete pixel-center unrotation are
not silently treated as identical; see [data notes](data.md).

The default SFT recipe preserves `val_size=0.01` and disabled in-training
validation from the source. This is not a newly asserted scene-disjoint
internal split. The supplied external validation split is kept frame-disjoint
from training by the release data tools.

## Historical GRPO reward

The actual supplied launchers select `compute_score=nuscenes`. The active total
in `verl/utils/reward_score/nuscenes_reward.py` is:

```text
reward = format_reward + accuracy_reward + non_repeat_reward
```

Its behavior remains distinct from the full reward described in the paper:

1. The structural regex expects adjacent perception/action/answer blocks; an
   intervening think block does not satisfy that regex.
2. Perception IoU is not added to the total. An action F1 helper exists but its
   invocation and contribution to the total are inactive.
3. Trajectory reward uses lifted **3D L2**, representation consistency and a pixel
   threshold term, rather than entirely image-space training objectives.
4. Lifting in this reward uses CAM_FRONT, native 1600 × 900 dimensions, model
   784 × 448 dimensions and ground z=-1. It has no rear-unrotation branch.
5. The pixel-threshold reward divides hits by hits, so any hit gives 1, rather
   than a six-waypoint hit fraction.
6. Its historical parsing/geometry fallbacks remain numerical behavior of this
   variant. GRPO preparation rejects unsupported label representations.

No replacement scientific reward was invented during cleanup. Missing dataset
initialization now fails explicitly, dataset location is configurable, imports
are lazy and calibration is cached. The reward formula matched the original on
22 synthetic cases in the initial preparation; this is not real-data validation.

## Evaluation and training changes

Original action-evaluation variants that randomly changed predictions remain
excluded. Deterministic text diagnostics report the predictions as written.
Fixed-IoU box F1 is not mAP. The recovered planning helper and a new wrapper
provide L2, collision and boundary metrics. The wrapper matches sample ids,
uses info-file metric GT, counts exclusions, and aborts on scoring errors rather
than changing denominators silently. An unused red-light method was omitted
because its `expand_lane` dependency was missing; selected metrics do not use it.

The GRPO trainer's off-by-one stop condition is corrected: a requested one-step
run now reaches its batch. This changes the erroneous historical schedule, not
the reward. Exact checkpoint continuation of global-step/data position remains
uncertified. All modified upstream files carry modification notices.

## Validation limits

Local CPU tests, source/wheel builds and format checks are recorded in
[validation](validation.md). No teacher API was called, no dataset imagery was
uploaded, and no local checkpoint inference or distributed CUDA training was
run during preparation. The real nuScenes camera tables/maps were unavailable
in the provided code snapshot, so full projection generation and collision/map
evaluation have not been exercised on real sensor data.

Before claiming the paper's full result, establish the exact final checkpoint,
SFT/GRPO recipe, annotation cohort and complete reward, then run the target-server
training and benchmark protocol. This source release provides the recovered
components and explicit connections without claiming those results.
