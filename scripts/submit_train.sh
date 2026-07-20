#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-pooled}"
GPUS="${2:-4}"
if [[ "$GPUS" != "2" && "$GPUS" != "4" ]]; then
  echo "GPU count must be 2 or 4" >&2
  exit 2
fi
mkdir -p "$ROOT/logs"
MEMORY=$((90 * GPUS))G
JOB_ID="$(sbatch --parsable --ntasks-per-node="$GPUS" --gres="gpu:$GPUS" --mem="$MEMORY" \
  --output="$ROOT/logs/train_%j.out" --error="$ROOT/logs/train_%j.err" \
  --export=ALL,DATA_PRESET="$PRESET",GPUS="$GPUS" "$ROOT/slurm/train.sbatch")"
echo "Training queued: $JOB_ID"
echo "Log: $ROOT/logs/train_${JOB_ID}.out"

