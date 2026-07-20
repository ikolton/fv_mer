#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
git submodule update --init --recursive
JOB_ID="$(sbatch --parsable --export=ALL,FVLM_MERLIN_ROOT="$ROOT" slurm/setup.sbatch)"
echo "Setup queued: $JOB_ID"
echo "Log: $ROOT/logs/setup_${JOB_ID}.out"
while squeue -h -j "$JOB_ID" | grep -q .; do
  sleep 10
done
IFS='|' read -r STATE EXIT_CODE < <(
  sacct -X -j "$JOB_ID" --noheader --parsable2 --format=State,ExitCode | head -n 1
)
if [[ "$STATE" != "COMPLETED" || "$EXIT_CODE" != "0:0" ]]; then
  echo "Setup failed: job $JOB_ID ended with state=$STATE exit_code=$EXIT_CODE" >&2
  echo "See $ROOT/logs/setup_${JOB_ID}.err" >&2
  exit 1
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Setup failed: $ROOT/.venv/bin/python was not created" >&2
  exit 1
fi
echo "Environment ready: $ROOT/.venv"
