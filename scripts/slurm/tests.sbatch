#!/usr/bin/env bash
#SBATCH --job-name=sm-tests
#SBATCH --partition=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/slurm/runtime.sh"
"$SUPERMIMIC_PYTHON" -m pytest -q --basetemp="$SUPERMIMIC_OUTPUT_ROOT/tests/$SLURM_JOB_ID" "$@"
