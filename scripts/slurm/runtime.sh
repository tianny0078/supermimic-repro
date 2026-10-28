#!/usr/bin/env bash
# Source inside a Slurm allocation; preserve Slurm's device visibility.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit GPU work with sbatch or srun}"
SUPERMIMIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SUPERMIMIC_OUTPUT_ROOT="${SUPERMIMIC_OUTPUT_ROOT:-/mnt/data/$USER/supermimic-repro}"
SUPERMIMIC_PYTHON="${SUPERMIMIC_PYTHON:-/home/$USER/.venvs/supermimic-repro/bin/python}"
SUPERMIMIC_CACHE_ROOT="${SUPERMIMIC_CACHE_ROOT:-/home/$USER/.cache/supermimic-repro}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
if [[ -f "$SUPERMIMIC_ROOT/SOURCE_REVISION" ]]; then
  export SUPERMIMIC_SOURCE_REVISION="$(cat "$SUPERMIMIC_ROOT/SOURCE_REVISION")"
fi
export WANDB_MODE=disabled
export WARP_CACHE_PATH="$SUPERMIMIC_CACHE_ROOT/warp"
export TORCHINDUCTOR_CACHE_DIR="$SUPERMIMIC_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$SUPERMIMIC_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$SUPERMIMIC_CACHE_ROOT/cuda"
export MPLCONFIGDIR="$SUPERMIMIC_CACHE_ROOT/matplotlib"
mkdir -p "$SUPERMIMIC_OUTPUT_ROOT/logs" "$SUPERMIMIC_OUTPUT_ROOT/tests" "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"
# Stage repeated reads in RAM; outputs remain on persistent bulk storage.
: "${SUPERMIMIC_SOURCE_REVISION:?Use scripts/submit_experiment.py to freeze the source}"
SUPERMIMIC_SCRATCH="$(mktemp -d "/dev/shm/${USER}_supermimic_${SLURM_JOB_ID}_XXXXXX")"
trap 'rm -rf -- "$SUPERMIMIC_SCRATCH"' EXIT
mkdir -p "$SUPERMIMIC_SCRATCH/source" "$SUPERMIMIC_SCRATCH/data/npz"
cp -a "$SUPERMIMIC_ROOT/." "$SUPERMIMIC_SCRATCH/source/"
cp -a "$SUPERMIMIC_ROOT/data/npz/lafan1" "$SUPERMIMIC_SCRATCH/data/npz/lafan1"
if [[ "${SUPERMIMIC_STAGE_RAW:-0}" == 1 ]]; then
  mkdir -p "$SUPERMIMIC_SCRATCH/data/raw/lafan1"
  cp -aL "$SUPERMIMIC_ROOT/data/raw/lafan1/g1" "$SUPERMIMIC_SCRATCH/data/raw/lafan1/g1"
fi
rm "$SUPERMIMIC_SCRATCH/source/data"
ln -s "$SUPERMIMIC_SCRATCH/data" "$SUPERMIMIC_SCRATCH/source/data"
SUPERMIMIC_ROOT="$SUPERMIMIC_SCRATCH/source"
export PYTHONPATH="$SUPERMIMIC_ROOT/src"
echo "[INFO] source $SUPERMIMIC_SOURCE_REVISION; staged LAFAN1 in $SUPERMIMIC_SCRATCH; outputs $SUPERMIMIC_OUTPUT_ROOT"
cd "$SUPERMIMIC_ROOT"
