# Validation of source release 0.2.0

Prepared 2026-11-05 from the supplied Seg-Zero and additional LLaMA-Factory
research snapshots. Results below distinguish source/CPU validation from full
training or nuScenes benchmark reproduction.

## Repository cleanup: 2026-11-06

The three original research snapshots, including their historical data, logs
and credential-bearing scripts, were moved to the system Trash. Redundant root
source archives and system metadata were also removed. The bundled training
frameworks, project source, configurations, documentation, examples, tests and
license/provenance records remain in the repository. Original-file hash and
real-info checks recorded below were performed before this cleanup.

## Repository layout check: 2026-11-06

The release now lives directly at the repository root. The SFT framework moved
from `third_party/llamafactory/` to `LLaMAFactory/`; launchers, dependency paths,
source packaging, CI configuration and provenance destinations were updated.
At the time of these checks, the original research snapshots were retained
without edits; they were subsequently removed as described above.

- All **29 existing CPU regression tests passed** after relocation.
- Ruff, shell syntax checks and SFT/GRPO launch previews passed.
- Root source/wheel builds and the separate LLaMA-Factory wheel build succeeded.
- All **74 relative links** in the root/project documentation resolved.
- **258 provenance entries** matched the original source hashes, and all recorded
  release destinations existed at their updated locations.
- The source audit checked **315 selected files** with no configured-pattern
  findings. Git ignore checks excluded the three local research snapshots,
  generated archives and build products.

These are local source/CPU checks. No GPU training, checkpoint inference,
teacher API requests or real-map evaluation were run in this layout pass.

## Observed checks

- **29 CPU regression tests passed.** Coverage includes recovered SFT configuration,
  multi-task repeated-frame handling, exact-duplicate rejection, deterministic
  task sampling, action/box/waypoint parsing, native rear rotation, recovered
  projection versus an explicit calibration transform, metric frame/id/mask
  checks, reasoning correction/improvement selection and attachment, benchmark
  split alignment, historical reward import/formula behavior, the actual trainer
  stop boundary, and reproducible archive/hash checks.
- End-to-end SFT JSON conversion, generated registration and launch preflight
  accepted two different tasks from the same frame. Integrated GRPO DatasetDict
  conversion and save/readback also passed on synthetic imagery.
- The supplied validation info pickle was read successfully: **6019 records**,
  all containing the fields required by generation and the recovered planning
  wrapper. This checks schema, not correctness of camera calibration or maps.
- Reorganized multi-task prompts, targets and sampling matched the original
  selected generator on **20 real-info cases with synthetic camera projection**.
  All three extracted geometric method bodies matched the original abstract
  syntax trees. No actual nuScenes camera projection was run.
- Twelve CLI help entry points passed. SFT launch and teacher generation previews
  ran offline, with no model loading or external API calls. Shell syntax and
  local documentation link checks passed.
- Ruff and Python syntax compilation passed for release code and all 170 bundled
  LLaMA-Factory Python source files.
- Standard source/wheel builds succeeded, including a separate
  `llamafactory-0.9.4.dev0` wheel build. The root source export carries the full
  bundled SFT source, required configuration files and original license notices.
- **258 provenance entries across 256 source files** matched original hashes;
  the original research code was not edited. Data, weights and logs are excluded
  from the source export. The configured credential/private-path pattern scan
  found no matches in selected source files; this is not exhaustive detection.
- The retained GRPO reward had matched its original implementation on 22
  synthetic cases during initial preparation. That evidence concerns the
  retained numerical formula, not the paper's complete reward.

## Environment and limits

Local checks used macOS / Python 3.12.7, numpy 1.26.4, scipy 1.13.1, Pillow 11.3.0,
PyYAML 6.0.2, OmegaConf 2.3.0 and pyquaternion 0.9.9. Dataset conversion used
datasets 4.0.0, pyarrow 21.0.0 and huggingface-hub 0.34.4. The old info pickle
contains Shapely geometries; local readback used Shapely 2.2.0 compatibility
loading and did not resave or modify the original pickle.

The scipy 1.15.3 macOS wheel failed to load a native component, so the local test
environment used 1.13.1. The Linux GRPO requirements retain the historical 1.15.3
pin. Shapely 1.8.5.post1 did not build in the local Python 3.12 environment; its
compatible 2.x reader was used for schema inspection. These substitutions are
not evidence that the complete historical CUDA environment installs unchanged.

No teacher API request, data upload, 7B checkpoint inference, distributed SFT or
GRPO training, or real-map collision/boundary evaluation was performed.
No paper benchmark scores or remote CI success are claimed. The included GitHub
Actions workflow is a future check, not a completed remote run.

See [reproduction status](reproduction.md) for the remaining scientific differences.
