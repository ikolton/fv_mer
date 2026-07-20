#!/bin/bash
#SBATCH --job-name=fvlm-smoke
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail
ROOT="${FVLM_MERLIN_ROOT:-${SLURM_SUBMIT_DIR:?Submit smoke through scripts/submit_smoke.sh}}"
cd "$ROOT"
source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module use /net/software/aarch64/el9modules/all/Core
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
python -c 'import torch; assert torch.cuda.is_available(); print("gpus", torch.cuda.device_count())'
mkdir -p outputs/manifests/smoke outputs/runs/smoke logs
fvlm-merlin build-manifest --config "configs/data/${DATA_PRESET:-merlin}.yaml" \
  --output-dir outputs/manifests/smoke --limit-per-split 8 --prefer-abnormal
fvlm-merlin validate outputs/manifests/smoke/train.json outputs/manifests/smoke/val.json --samples-per-dataset 1
python -m torch.distributed.run --nproc_per_node="${GPUS:-1}" -m fvlm_merlin.cli train --config configs/train/smoke.yaml \
  --options "run.output_dir=$ROOT/outputs/runs/smoke-${SLURM_JOB_ID}"
