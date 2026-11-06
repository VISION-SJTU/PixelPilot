# Evaluation contracts

## Text prediction diagnostics

`pixelpilot.infer` writes `id`, `predict`, `label`, model name, decoding settings,
image preprocessing limits and conditioning status. The default decoding uses
temperature 1, top-p 0.5 and top-k 20 with a recorded seed. `--greedy` is available
for diagnostics. Max image pixels default to 720000 to match the recovered
training configuration, including native-sized collage input.

```bash
python -m pixelpilot.evaluate --predictions outputs/predictions.jsonl --task action
python -m pixelpilot.evaluate --predictions outputs/predictions.jsonl --task boxes
```

Action evaluation reports per-class precision/recall/F1, micro F1, macro F1 over
supported classes, and path/speed/joint accuracy. An invalid prediction counts
as an empty action set and is reported. Invalid ground truth fails.
Detection uses class-aware Hungarian matching at a fixed IoU threshold and
reports precision/recall/F1; it is not AP/mAP. Invalid perception predictions
count as no boxes. Predictions are never randomly modified.

Legacy prediction files without ids can be used for text diagnostics only.
They cannot certify split completeness. Retain id-preserving complete outputs
and the frozen split file for benchmark reporting.

## Image-to-metric lifting

```bash
python -m pixelpilot.lift --predictions outputs/predictions.jsonl \
  --nusc-root data/nuscenes --version v1.0-trainval --ground-z -1.0 \
  --frame PLANNING --output outputs/trajectories.json
```

The explicit plane height is in current LIDAR_TOP coordinates. The helper uses
per-sensor capture poses, removes the front/rear collage mapping, rescales to
native pixels and intersects rays with the plane. Parallel/backward rays,
malformed trajectories and out-of-bounds pixels fail rather than becoming zero
trajectories. Historical `<trajY>` tokens retain an explicit `[0,Y,z]` LIDAR_TOP
representation. Their meaning must remain consistent with the generated labels.

Output frame `PLANNING` maps LIDAR_TOP `(x,y)` to `(y,-x)`, matching the recovered
`gt_planning` projection convention. The default `LIDAR_TOP` output remains
available for existing consumers. Do not label one as the other, and do not
assume a generic ego/IMU transform without validating the original schema.

Generic rear unrotation uses discrete pixel centers; recovered annotation
history uses a continuous-coordinate convention offset by one pixel. New JPEG
encoding, plane approximation and that convention are not claimed byte-exact
historical reproduction.

## L2 diagnostics

```bash
python -m pixelpilot.evaluate_planning --predictions outputs/trajectories.json \
  --ground-truth data/annotations/val_metric_gt.json
```

GT records come independently from `gt_planning` and include six boolean valid
flags. The tool requires identical unique ids, matching explicit frames and six
finite waypoints. It reports endpoint and prefix-mean L2 at 1/2/3 seconds, with
complete validity through each horizon and the denominator for that horizon.
It never obtains metric GT by lifting the 2D answer label.

## Recovered nuScenes collision and boundary evaluator

```bash
python -m pixelpilot.benchmark.evaluate --predictions outputs/trajectories.json \
  --split-json data/annotations/val_integrated.json --infos data/infos/val.pkl \
  --nusc-root data/nuscenes --output outputs/nuscenes_metrics.json
```

This path includes the supplied `PlanningMetric` and custom nuScenes map API.
Install `requirements/evaluation.txt` and provide camera data plus
`maps/expansion`. Predictions must use `PLANNING` coordinates and exactly match
the frozen split ids. Metric GT, actor boxes/futures and map location are read
from the trusted info file, not the prediction's label field.

The wrapper preserves the selected source metric behavior:

- Only samples whose complete six-step GT planning mask is valid are scored.
- L2 is the mean displacement through 1, 2 and 3 seconds, then averaged per sample.
- Collision is whether any predicted ego-box collision occurs through the horizon,
  using the helper's existing GT-collision exclusion behavior and vehicle categories.
- Boundary violation is whether the ego footprint leaves the drivable map,
  excluding a frame that starts outside the drivable area.
- Collision/boundary outputs are fractions; multiply by 100 to express percent.

The wrapper reports input/scored/incomplete-GT counts and aborts on other errors.
The original wrapper's row-order association, broad exception continuation and
partial-denominator behavior are not used. Missing `expand_lane` affected an
unused red-light helper; that helper is omitted and is not part of these metrics.

The evaluator source is recovered, but no real-map execution or exact match to
the paper's final benchmark scores has been verified. Do not interpret the
included implementation or local CPU unit tests as evidence of those results.
