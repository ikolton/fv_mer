#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-merlin}"
GPUS="${2:-1}"
if [[ "$GPUS" != "1" && "$GPUS" != "2" ]]; then
  echo "Smoke GPU count must be 1 or 2" >&2
  exit 2
fi
mkdir -p "$ROOT/logs"
JOB_ID="$(sbatch --parsable --gres="gpu:$GPUS" \
  --export=ALL,FVLM_MERLIN_ROOT="$ROOT",DATA_PRESET="$PRESET",GPUS="$GPUS" "$ROOT/slurm/smoke.sbatch")"
echo "Smoke queued: $JOB_ID"
echo "Log: $ROOT/logs/smoke_${JOB_ID}.out"
