#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
git submodule update --init --recursive
sbatch --wait slurm/setup.sbatch
echo "Environment ready: $ROOT/.venv"

