#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-pooled}"
mkdir -p "$ROOT/outputs/manifests/$PRESET" "$ROOT/logs"
JOB_ID="$(sbatch --parsable --job-name=fvlm-data-check --partition=plgrid-gpu-gh200 \
  --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --output="$ROOT/logs/data_check_%j.out" --error="$ROOT/logs/data_check_%j.err" \
  --wrap="source /usr/share/lmod/lmod/init/bash; module load ML-bundle/24.06a; source '$ROOT/.venv/bin/activate'; fvlm-merlin build-manifest --config '$ROOT/configs/data/$PRESET.yaml' --output-dir '$ROOT/outputs/manifests/$PRESET'; fvlm-merlin validate '$ROOT/outputs/manifests/$PRESET/train.json' '$ROOT/outputs/manifests/$PRESET/val.json'")"
echo "Data check queued: $JOB_ID"
echo "Log: $ROOT/logs/data_check_${JOB_ID}.out"

