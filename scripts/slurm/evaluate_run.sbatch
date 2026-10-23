#!/usr/bin/env bash
#SBATCH --job-name=sm-eval-run
#SBATCH --partition=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/slurm/runtime.sh"
exec "$SUPERMIMIC_PYTHON" scripts/eval_run.py "$@"
