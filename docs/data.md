# nuScenes data preparation

Run commands from the repository root in the [data environment](installation.md).

## Generate annotations and collages

Arrange the local inputs as follows:

```text
data/
├── nuscenes/                  # Original camera data, metadata tables and maps
├── infos/train.pkl
├── infos/val.pkl
├── raw/nuscenes_train.json    # Original metric-space conversations
└── raw/nuscenes_val.json
```

```bash
python -m pixelpilot.generate \
  --train-json data/raw/nuscenes_train.json --train-infos data/infos/train.pkl \
  --val-json data/raw/nuscenes_val.json --val-infos data/infos/val.pkl \
  --nusc-root data/nuscenes --output-dir data/annotations --seed 42

python -m pixelpilot.collage --annotations data/annotations/train_integrated.json \
  --nusc-root data/nuscenes --output-dir data/nusc_fb_train --native-resolution
python -m pixelpilot.collage --annotations data/annotations/val_integrated.json \
  --nusc-root data/nuscenes --output-dir data/nusc_fb_val --native-resolution
```

Use the generated mixed tasks for [SFT](training.md#multi-task-sft), and the unique
integrated tasks for [GRPO](training.md#grpo) and [inference](inference.md).

## Local input schema

`pixelpilot.generate` adapts the supplied
`prepare_dataset/generare_2d/fb_mix_percep_action_traj.py`. Its inputs are:

- Original single-turn metric-space nuScenes train/validation JSON. Each row has
  a nuScenes sample token `id`, one image path, and user/assistant messages. The
  user prompt contains metric history pairs; it is not a pre-existing 2D prompt.
- A trusted local pickle with an `infos` list. Each selected frame needs `token`,
  `gt_planning`, `gt_planning_mask`, `gt_planning_command`, `can_bus`, `next`,
  `bboxes2d` and `labels2d`. Successor velocity is used for speed-action labels.
- nuScenes calibration and pose tables plus original camera images.

These artifacts are excluded from the source distribution and must be supplied
separately. The upstream converter that originally produced the info pickles
was not recovered. The original research snapshots used during preparation
have been removed from the repository.

## Generation rules preserved from the selected script

The pipeline maps metric planning `(forward, left)` to the supplied local
projection convention `(-left, forward, z=-1)` and uses CAM_FRONT for future
waypoints and CAM_BACK for history. It retains the original equations and the
historical rear-history continuous-coordinate mapping `(784-u, 896-v)`.
That annotation convention differs by one pixel from discrete image-center
unrotation in generic `pixelpilot.lift`; both are explicit rather than silently
conflated.

Perception uses front-camera boxes scaled from 1600 × 900 to 784 × 448, drops
boxes with width or height below 10 pixels, and retains at most 30 boxes.
Class names are the ten supplied nuScenes classes. Meta-actions follow the
source command mapping, a 0.1 m stationary threshold, and +/-0.1 m/s speed-change
thresholds. Missing successor ids retain the source's zero-velocity fallback.

Future points within 1 m longitudinal distance and near-ego history points use
`<trajY>` tokens. Other future points retain the original projection even when
the source visibility mask marks them invalid; the release does not invent a
new filtering policy. Projection dimensions and the ground plane are fixed to
the recovered nuScenes setup.

For each frame, perception and action tasks are independently sampled at 0.5,
while trajectory and integrated tasks are always included. Empty perception-only
tasks are omitted. The original script used unseeded randomness; this release
uses an explicit seed (42 for train and 43 for validation by default), then
shuffles the mixed SFT records. This makes the recipe reproducible but does not
recover a historical unseeded sample list.

## Outputs

```text
data/annotations/
  train_sft.json            multi-task records with repeated frame ids
  train_integrated.json     one complete perception/action/trajectory task per frame
  train_metric_gt.json      independent metric GT and validity masks
  val_sft.json
  val_integrated.json
  val_metric_gt.json
  generation_summary.json   frame and task counts plus sampling seed
```

`PLANNING` coordinates mean `(x forward, y left)` under the recovered
`gt_planning` convention. They are related to the projection's LIDAR_TOP plane
by `(lidar_y, -lidar_x)` and are intentionally distinguished from generic `EGO`.
Metric GT is read directly from `gt_planning`, never re-lifted from a 2D target.
Train/validation overlap, unknown ids and missing fields fail explicitly.

## Images and model preprocessing

The recovered image script stacks a native 1600 × 900 front image above a
180-degree-rotated rear image. Use `pixelpilot.collage --native-resolution` to
store that 1600 × 1800 composition. SFT and inference explicitly use a 720000
pixel limit, leading to the 784 × 896 label canvas in this camera setup.

The default collage helper can also emit 784 × 896 images directly. Resizing
before versus after composition may change interpolation near the seam; do not
claim byte-identical historical inputs. JPEG files are newly encoded.

## SFT and GRPO preparation

`pixelpilot.prepare --format sft` supports perception-only, action-only,
trajectory-only and integrated targets, including accepted reasoning targets.
A frame id may appear across distinct tasks; exact duplicate records are rejected.
The converter writes absolute local image paths and `dataset_info.json` with
`pixelpilot_train` and optional `pixelpilot_val`. Re-run conversion after moving
to another machine. Frame-level train/val overlap is rejected even for multi-task
records.

`--extra-train-json` appends a non-empty set of selected reasoning records before
SFT conversion. It may be repeated. All samples retain their original frame ids.

GRPO conversion requires unique frame ids and complete integrated targets. Use
`train_integrated.json`, not the multi-task SFT file. It exports `id`, `problem`,
`solution`, `meta_action`, `perception`, `image`, `img_height`, `img_width` as a
local Hugging Face DatasetDict. The supplied historical reward consumes only the
trajectory text in `solution`; the other labels are preserved. Its parser requires
nonnegative integer `[u, v]` pairs with comma-space or decimal `<trajY>` tokens.
Unsupported targets fail rather than silently receiving zero accuracy reward.

Original 3D `<PLANNING>` files are input to generation, not direct input to the
2D training-data converter. Existing output paths are never overwritten by the
preparation commands.
