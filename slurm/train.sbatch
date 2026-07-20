#!/bin/bash
#SBATCH --job-name=fvlm-train
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1 || true
module load ML-bundle/24.06a
source .venv/bin/activate
export FVLM_MERLIN_ROOT="$ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

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
