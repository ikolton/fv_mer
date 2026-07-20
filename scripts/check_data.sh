#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-pooled}"
cd "$ROOT"
mkdir -p "$ROOT/outputs/manifests/$PRESET" "$ROOT/logs"
JOB_ID="$(sbatch --parsable \
  --export=ALL,FVLM_MERLIN_ROOT="$ROOT",DATA_PRESET="$PRESET" "$ROOT/slurm/data_check.sbatch")"
echo "Data check queued: $JOB_ID"
echo "Log: $ROOT/logs/data_check_${JOB_ID}.out"
