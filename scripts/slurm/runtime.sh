#!/usr/bin/env bash
# Source inside a Slurm allocation; preserve Slurm's device visibility.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit GPU work with sbatch or srun}"
SUPERMIMIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SUPERMIMIC_OUTPUT_ROOT="${SUPERMIMIC_OUTPUT_ROOT:-/mnt/data/$USER/supermimic-repro}"
SUPERMIMIC_PYTHON="${SUPERMIMIC_PYTHON:-/mnt/data/$USER/supermimic-repro-env/bin/python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SUPERMIMIC_ROOT/src"
if [[ -f "$SUPERMIMIC_ROOT/SOURCE_REVISION" ]]; then
  export SUPERMIMIC_SOURCE_REVISION="$(cat "$SUPERMIMIC_ROOT/SOURCE_REVISION")"
fi
export WANDB_MODE=disabled
export WARP_CACHE_PATH="$SUPERMIMIC_OUTPUT_ROOT/cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$SUPERMIMIC_OUTPUT_ROOT/cache/torchinductor"
export MPLCONFIGDIR="$SUPERMIMIC_OUTPUT_ROOT/cache/matplotlib"
mkdir -p "$SUPERMIMIC_OUTPUT_ROOT/logs" "$SUPERMIMIC_OUTPUT_ROOT/tests" "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"
cd "$SUPERMIMIC_ROOT"
