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
"$SUPERMIMIC_PYTHON" -m supermimic.experiment --mode pilot \
  --name "pilot-$SLURM_JOB_ID" --output "$SUPERMIMIC_OUTPUT_ROOT/runs" "$@" &
train_pid=$!
# Keep the batch shell for scratch cleanup and forward checkpoint signals.
trap 'kill -USR1 "$train_pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$train_pid" 2>/dev/null || true' TERM
while true; do
  if wait "$train_pid"; then train_status=0; else train_status=$?; fi
  if ! kill -0 "$train_pid" 2>/dev/null; then break; fi
done
exit "$train_status"
