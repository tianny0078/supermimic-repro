#!/usr/bin/env bash
#SBATCH --job-name=sm-lafan
#SBATCH --partition=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/slurm/runtime.sh"
# Default is a full-scale, 240-iteration pilot. For a measured training budget,
# override sbatch --time and pass --mode paper --max-hours HOURS explicitly.
exec "$SUPERMIMIC_PYTHON" -m supermimic.experiment --mode pilot \
  --name "pilot-$SLURM_JOB_ID" --output "$SUPERMIMIC_OUTPUT_ROOT/runs" "$@"
