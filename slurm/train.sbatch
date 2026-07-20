#!/bin/bash
#SBATCH --job-name=fvlm-train
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00

set -euo pipefail
ROOT="${FVLM_MERLIN_ROOT:-${SLURM_SUBMIT_DIR:?Submit training through scripts/submit_train.sh}}"
cd "$ROOT"
source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
export MODULEPATH=/net/software/aarch64/el9modules/all/Core
module load GCCcore/13.2.0
set +e
module load ML-bundle/24.06a
MODULE_STATUS=$?
set -e
if [[ "$MODULE_STATUS" -ne 0 ]]; then
  echo "WARNING: ML-bundle returned $MODULE_STATUS; continuing to the CUDA preflight." >&2
fi
source .venv/bin/activate
export FVLM_MERLIN_ROOT="$ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
python -c 'import torch; assert torch.cuda.is_available(); print("gpus", torch.cuda.device_count())'

PRESET="${DATA_PRESET:-pooled}"
MANIFEST_DIR="$ROOT/outputs/manifests/$PRESET"
RUN_DIR="$ROOT/outputs/runs/${RUN_NAME:-${PRESET}-${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR" logs

fvlm-merlin build-manifest --config "$ROOT/configs/data/$PRESET.yaml" --output-dir "$MANIFEST_DIR"
fvlm-merlin validate "$MANIFEST_DIR/train.json" "$MANIFEST_DIR/val.json"

OPTIONS=(
  "datasets.abdominal_fvlm_caption.build_info.annotations.train.storage=$MANIFEST_DIR/train.json"
  "datasets.abdominal_fvlm_caption.build_info.annotations.val.storage=$MANIFEST_DIR/val.json"
  "run.output_dir=$RUN_DIR"
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  OPTIONS+=("run.resume_ckpt_path=$RESUME_CHECKPOINT")
fi

python -m torch.distributed.run --nproc_per_node="${GPUS}" -m fvlm_merlin.cli train \
  --config "$ROOT/configs/train/gh200.yaml" --options "${OPTIONS[@]}"
