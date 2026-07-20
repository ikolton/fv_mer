#!/bin/bash
#SBATCH --job-name=fvlm-data-check
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/data_check_%j.out
#SBATCH --error=logs/data_check_%j.err

set -euo pipefail
ROOT="${FVLM_MERLIN_ROOT:-${SLURM_SUBMIT_DIR:?Submit through scripts/check_data.sh}}"
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
  echo "WARNING: ML-bundle returned $MODULE_STATUS; continuing to the data preflight." >&2
fi
source .venv/bin/activate
export FVLM_MERLIN_ROOT="$ROOT"

PRESET="${DATA_PRESET:-pooled}"
MANIFEST_DIR="$ROOT/outputs/manifests/$PRESET"
mkdir -p "$MANIFEST_DIR" logs
fvlm-merlin build-manifest \
  --config "$ROOT/configs/data/$PRESET.yaml" \
  --output-dir "$MANIFEST_DIR"
fvlm-merlin validate \
  "$MANIFEST_DIR/train.json" \
  "$MANIFEST_DIR/val.json" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
