#!/bin/bash
#SBATCH --job-name=fvlm-setup
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --exclude=x1002c4s0b1n0
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

set -euo pipefail
ROOT="${FVLM_MERLIN_ROOT:-${SLURM_SUBMIT_DIR:?Run setup from the project root}}"
cd "$ROOT"
source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1 || true
set +e
module load ML-bundle/24.06a
MODULE_STATUS=$?
set -e
if [[ "$MODULE_STATUS" -ne 0 ]]; then
  echo "WARNING: ML-bundle returned $MODULE_STATUS; continuing to the Python preflight." >&2
fi
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[test]'
python -m fvlm_merlin.cli download-assets
python - <<'PY'
import torch
import transformers
import fvlm_merlin
assert transformers.__version__ == "4.25.0"
assert torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("fvlm_merlin", fvlm_merlin.__version__)
PY
