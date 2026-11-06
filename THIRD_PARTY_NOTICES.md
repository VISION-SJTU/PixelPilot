# Third-party provenance

| Component | Source | Included | Provenance / license |
| --- | --- | --- | --- |
| SFT framework | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | `LLaMAFactory/`, complete supplied Python source and installation metadata | Local `LLaMA-Factory-0.9.4` research snapshot; internal version `0.9.4.dev0`; Apache-2.0; original notices retained |
| GRPO framework | [Seg-Zero](https://github.com/dvlab-research/Seg-Zero), based on [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl) | `verl/`, local checkpoint merger | Supplied Seg-Zero-main snapshot; Apache-2.0; original headers/license retained |
| Data projection and labeling | Supplied LLaMA-Factory research preparation scripts | `pixelpilot/data/projection.py`, `pixelpilot.generate`, retained teacher prompt and new wrappers | Derived from `fb_mix_percep_action_traj.py` and `create_vision_cot_data.py`; source hashes and adaptations recorded |
| Planning/map metrics | Supplied `evaluation_scripts/planning_utils.py`, `map_api.py`, `eval_2d.py` | `pixelpilot/benchmark/` | Original NVIDIA/nuScenes-devkit notices retained; Apache-2.0; wrapper adapted for explicit ids and errors |
| Base model | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | No weights | Independent pretrained-model terms |
| Dataset / upstream preparation | [nuScenes](https://www.nuscenes.org/nuscenes), OmniDrive-style local infos | No data or pickles | Obtain dataset/artifacts separately and follow their terms |

No upstream commit ids were included in the local research directories. The
bundled framework is not asserted to be an unmodified official 0.9.4 release.
`docs/source_manifest.json` contains original hashes before release adaptations;
entries with no destination record reference-only sources. New connecting code
is explicitly distinguished from recovered experiment code in the documentation.

The source exporter includes licenses and excludes datasets, weights, logs,
notebook caches and package build products, including those created while
building the bundled framework. Third-party README files remain upstream
reference documentation; the root README defines this release's actual scope.
