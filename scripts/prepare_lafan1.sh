#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SUPERMIMIC_PYTHON="${SUPERMIMIC_PYTHON:-/mnt/data/$USER/supermimic-repro-env/bin/python}"
SUPERMIMIC_OUTPUT_ROOT="${SUPERMIMIC_OUTPUT_ROOT:-/mnt/data/$USER/supermimic-repro}"
LAFAN1_ROOT="${LAFAN1_ROOT:-/mnt/data/datasets/lafan1_g1/raw}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
"$SUPERMIMIC_PYTHON" scripts/data/download_lafan1.py --output "$LAFAN1_ROOT"
mkdir -p "$SUPERMIMIC_OUTPUT_ROOT/data/raw/lafan1"
if [[ ! -e "$SUPERMIMIC_OUTPUT_ROOT/data/raw/lafan1/g1" ]]; then
  ln -s "$LAFAN1_ROOT" "$SUPERMIMIC_OUTPUT_ROOT/data/raw/lafan1/g1"
fi
if [[ ! -e data ]]; then
  ln -s "$SUPERMIMIC_OUTPUT_ROOT/data" data
fi
"$SUPERMIMIC_PYTHON" scripts/data/csv_to_npz_batch.py \
  --input-dir "$LAFAN1_ROOT" --output-dir "$SUPERMIMIC_OUTPUT_ROOT/data/npz/lafan1" \
  --input-fps 30 --output-fps 50 --workers "${SUPERMIMIC_DATA_WORKERS:-8}" --pattern '*.csv'
