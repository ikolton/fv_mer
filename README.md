# fVLM Merlin

Train the fVLM visual-language encoder on abdominal CT data and export reusable per-organ features. The project adapts the original fVLM objective to an 11-organ abdominal model.

## Quick start on Helios

The repository is configured for GH200 nodes and the shared converted datasets under `/net/storage/pr3/plgrid/plggjmiag`. Generated manifests, logs, checkpoints, and features stay inside the repository.

```bash
git clone --recurse-submodules https://github.com/ikolton/fv_mer.git
cd fv_mer
./scripts/setup_helios.sh
```

Setup submits a short GH200 job, creates `.venv`, installs the package, and downloads the MAE and CXR-BERT initial weights into `assets/`. The script waits for completion and reports any setup failure directly. The environment is built for the compute-node architecture, so use it from batch jobs or an interactive GH200 allocation.

### 1. Check the data

```bash
./scripts/check_data.sh pooled
```

This queues an exhaustive data-validation job. It builds portable train and validation manifests, checks split overlap and file availability, and opens every image-mask pair to verify NIfTI geometry and supported organ labels. Checks run in parallel and may take a while on the pooled dataset. The job ID and log path are printed after submission. Use `merlin` instead of `pooled` to select only Merlin data.

The presets are:

- `merlin`: Merlin training data and its validation manifest.
- `pooled`: Merlin, Swiss, and Turkish training data with the Merlin validation manifest.

Training optimizes the original fVLM objective on the train split. After every epoch it evaluates the same objective on the complete Merlin validation split using deterministic 3D center crops. Validation logs include total and organ-wise losses, and the lowest total validation loss selects `checkpoint_best.pth`.

### 2. Run a smoke test

```bash
./scripts/submit_smoke.sh merlin
./scripts/submit_smoke.sh merlin 2
```

The first command tests one GPU; the second tests the distributed two-GPU path. Each smoke job selects eight studies per split, runs one short training epoch, and writes a checkpoint under `outputs/runs/`. Run at least the two-GPU smoke before a multi-GPU training job.

### 3. Train

```bash
./scripts/submit_train.sh pooled 4
```

The first argument selects the data preset and the second selects two or four GPUs. Training rebuilds the selected manifest, checks a small image-mask sample as a fast preflight, and launches fVLM through PyTorch distributed training. Run `check_data.sh` once for exhaustive validation before the first full training job. Logs are written to `logs/train_<job-id>.out` and `logs/train_<job-id>.err`.

To resume a compatible run:

```bash
RESUME_CHECKPOINT=/path/to/checkpoint.pth ./scripts/submit_train.sh pooled 4
```

Run files are grouped under `outputs/runs/<preset>-<job-id>`. The run root contains the resolved configuration and metadata with the source revisions, training-manifest checksum, SLURM job ID, and world size. Checkpoints are stored in the timestamped fVLM subdirectory.

## Export features

Feature export loads a trained checkpoint and produces one normalized embedding and presence flag for every organ, plus a pooled study embedding. The output records the organ order and preprocessing provenance for downstream use.

Run this inside a GH200 job or interactive allocation:

```bash
fvlm-merlin export-features \
  --manifest outputs/manifests/pooled/val.json \
  --config configs/train/gh200.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --output outputs/features/pooled-val.pt
```

For a large manifest, split extraction across jobs by giving each job the same shard count and a different zero-based shard index:

```bash
MANIFEST=outputs/manifests/pooled/val.json
CHECKPOINT=/path/to/checkpoint.pth
for SHARD in 0 1 2 3; do
  fvlm-merlin export-features \
    --manifest "$MANIFEST" \
    --config configs/train/gh200.yaml \
    --checkpoint "$CHECKPOINT" \
    --output "outputs/features/pooled-val.shard${SHARD}.pt" \
    --num-shards 4 \
    --shard-index "$SHARD"
done
fvlm-merlin export-features \
  --output outputs/features/pooled-val.pt \
  --merge outputs/features/pooled-val.shard{0,1,2,3}.pt
```

Merging verifies the schema, checkpoint, manifest checksum, shard count, and record uniqueness before writing the final file.

## Data presets

Data presets in `configs/data/` define shared roots, annotations, and split membership. Generated manifests retain root aliases and relative image paths, making them portable between users with access to the same roots. Rows without converted image or mask files are skipped by the shared presets and listed in `summary.json`.

Images and masks are resampled together to 1.5 mm isotropic spacing. Images use continuous interpolation and masks use nearest-neighbour interpolation before fVLM cropping. Source segmentation IDs are remapped to the fixed organ order in `src/fvlm_merlin/organs.py`.

Dataset captions are passed through unchanged. The fVLM objective identifies canonical normal captions by their normal-text prefix.

To support another dataset layout, copy a preset and update its roots and annotation paths. For another Linux/CUDA system, create an environment with compatible PyTorch and MONAI versions, install the package with `pip install -e '.[test]'`, and use the same CLI commands. Set `FVLM_ROOT` to use an fVLM checkout outside the repository.

## Direct commands

The scripts above are the normal Helios entry points. Their underlying commands are useful for debugging inside a compute allocation:

```bash
fvlm-merlin build-manifest --config configs/data/pooled.yaml --output-dir outputs/manifests/pooled
fvlm-merlin validate outputs/manifests/pooled/train.json outputs/manifests/pooled/val.json
fvlm-merlin train --config configs/train/gh200.yaml
pytest -q
```

`validate` checks every volume by default. Use `--max-per-dataset N` for a quick preflight and `--workers N` to control parallel reads.
