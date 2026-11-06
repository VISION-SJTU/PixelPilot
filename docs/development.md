# Development and source release

Run commands from the repository root. The repository installs `pixelpilot`
and the bundled `verl` module; the SFT framework is installed separately from
`LLaMAFactory/`. See [installation](installation.md) for training environments.

## CPU checks

The following dependency versions were used for the local macOS CPU checks:

```bash
python -m pip install numpy==1.26.4 scipy==1.13.1 Pillow==11.3.0 \
  PyYAML==6.0.2 omegaconf==2.3.0 pyquaternion==0.9.9
python -m pip install -r requirements/dev.txt
python -m pip install --no-deps -e .

python -m unittest discover -s tests -v
ruff check --config pyproject.toml pixelpilot verl scripts tests LLaMAFactory/src
python -m compileall -q pixelpilot verl scripts LLaMAFactory/src
bash -n scripts/train_sft.sh scripts/train_grpo.sh
bash scripts/train_sft.sh --dry-run
MODEL_PATH=checkpoints/pixelpilot-7b-sft bash scripts/train_grpo.sh --dry-run
python scripts/audit_release.py
```

The [GitHub Actions workflow](../.github/workflows/checks.yml) runs CPU checks
and package builds on Python 3.10 / Linux. An included workflow does not imply a
successful remote run. Scientific and GPU validation limits are documented in
[validation](validation.md) and [reproduction status](reproduction.md).

## Package builds

```bash
python -m pip install 'setuptools>=77' wheel
python -m build --no-isolation
python -m build --no-isolation --wheel LLaMAFactory
```

The root source distribution carries the SFT framework and configuration files.
The root wheel installs the `pixelpilot` and `verl` Python packages; it does not
replace the complete repository needed by the training launchers.

## Export a source archive

```bash
python scripts/audit_release.py
python scripts/export_release.py --output dist/PixelPilot-nuscenes-source.tar.gz
```

The exporter uses an explicit source allowlist, refuses an existing output, and
includes a `RELEASE_MANIFEST.json` with per-file SHA-256 values. Archive entries
have deterministic names, order and timestamps. The provenance of supplied
research files is retained separately in [source_manifest.json](source_manifest.json).

Datasets, checkpoints, predictions, logs, caches, local credentials and build
products are excluded. The original `LLaMA-Factory093/`,
`LLaMA-Factory-0.9.4/` and `Seg-Zero-main/` research snapshots have been removed
from the repository. Their historical origin paths and hashes remain in the
source manifest for attribution. The bundled release code is self-contained;
training inputs and model weights must be obtained separately.
