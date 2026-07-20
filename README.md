# fVLM Merlin

Train the fVLM visual-language encoder on abdominal CT data and export reusable per-organ features. The project uses the original fVLM objective with an 11-organ abdominal model.

## Helios setup

The tested environment is a GH200 compute node. Dataset presets point to the team storage under `/net/storage/pr3/plgrid/plggjmiag` and never write there.

```bash
git clone --recurse-submodules <repository-url>
cd fvlm_merlin
./scripts/setup_helios.sh
./scripts/check_data.sh pooled
./scripts/submit_smoke.sh merlin
./scripts/submit_train.sh pooled 4
```

Pass `2` as the second smoke argument to verify DDP: `./scripts/submit_smoke.sh merlin 2`.

`setup_helios.sh` creates `.venv` on a compute node and downloads initial weights to `assets/`. Jobs activate that environment themselves; do not use it on the login node.

Use `merlin` for Merlin train/validation or `pooled` for Merlin, Swiss, and Turkish training with Merlin validation. A 2-GPU run uses the same configuration:

```bash
./scripts/submit_train.sh pooled 2
```

To resume, submit the generic job with `RESUME_CHECKPOINT` set:

```bash
RESUME_CHECKPOINT=/path/to/checkpoint.pth ./scripts/submit_train.sh pooled 4
```

Run outputs are placed in `outputs/runs/<preset>-<job-id>`. Each run records its resolved configuration, manifest checksum, revisions, and SLURM metadata.

## Commands

Inside a compute job or interactive GH200 allocation:

```bash
fvlm-merlin build-manifest --config configs/data/pooled.yaml --output-dir outputs/manifests/pooled
fvlm-merlin validate outputs/manifests/pooled/train.json outputs/manifests/pooled/val.json
fvlm-merlin train --config configs/train/gh200.yaml
fvlm-merlin export-features --manifest MANIFEST --config configs/train/gh200.yaml --checkpoint CHECKPOINT --output FEATURES.pt
```

Feature extraction can be sharded with `--num-shards` and `--shard-index`. Merge completed shards with:

```bash
fvlm-merlin export-features --output features.pt --merge features.shard*.pt
```

The feature file contains the fixed organ order, one embedding and presence flag per organ, a pooled study embedding, and preprocessing provenance. It is independent of downstream decoder code.

## Data and configuration

Data presets contain shared roots and source split definitions. Copy a preset to use another dataset layout. Generated manifests retain root aliases and relative paths, so they can move between users who share the same data roots. Missing files are errors by default. The shared presets explicitly exclude stale metadata rows without converted volumes and record every exclusion in `summary.json`.

Images and masks are resampled together to 1.5 mm isotropic spacing. Images use continuous interpolation and masks use nearest-neighbour interpolation before fVLM cropping.

Dataset captions are passed through unchanged. The fVLM objective identifies canonical normal captions by their normal-text prefix.

For another Linux/CUDA system, create a Python environment with a compatible PyTorch and MONAI installation, install this package with `pip install -e '.[test]'`, set dataset roots in a data preset, and use the same CLI commands. Set `FVLM_ROOT` to use an fVLM checkout outside this repository.
