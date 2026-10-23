#!/usr/bin/env bash
#SBATCH --job-name=sm-preflight
#SBATCH --partition=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/slurm/runtime.sh"
"$SUPERMIMIC_PYTHON" - <<'PY'
import json, os, torch, warp
assert torch.cuda.is_available(), 'CUDA unavailable inside Slurm allocation'
x = torch.randn(256, 256, device='cuda')
assert torch.isfinite(x @ x).all()
torch.cuda.synchronize()
print(json.dumps({'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'gpu_memory_gib': torch.cuda.get_device_properties(0).total_memory / 2**30, 'warp': warp.__version__}, indent=2))
PY
