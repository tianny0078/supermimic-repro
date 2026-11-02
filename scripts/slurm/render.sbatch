#!/usr/bin/env bash
#SBATCH --job-name=sm-video
#SBATCH --partition=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/slurm/runtime.sh"
export SUPERMIMIC_SCRATCH
export MUJOCO_GL=egl
"$SUPERMIMIC_PYTHON" scripts/render_video.py "$@"
