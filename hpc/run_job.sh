#!/bin/bash
#SBATCH --job-name=energy_sweep
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --gpu-freq=1065
#SBATCH --time=24:00:00
#SBATCH --output=job_output_%j.txt

set -euo pipefail

# 1. Manually add CUDA to the path (adjust the path if your 'ls' command found a specific version like cuda-12.1)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# 2. Preflight checks (helps debug CUDA init issues quickly)
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvcc
command -v ncu
nvidia-smi -L
nvidia-smi

# Run CUDA binaries before activating conda to avoid runtime library conflicts.

echo "[Stage] param_identification"
cd /home/gargia/coalesced-memory-tb/param_identification

make clean
ARCH=sm_90 make
make run-h100

if [ ! -f results/cache_miss_rates.csv ]; then
	echo "Error: param_identification/results/cache_miss_rates.csv was not generated"
	exit 1
fi

echo "[Stage] calibration"
cd /home/gargia/coalesced-memory-tb/calibration

make clean
make sweep_csv BLOCKS=264 THREADS=256
make sweep_ncu BLOCKS=264 THREADS=256

# Activate your personal Conda installation for Python analysis/plotting.
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
	source ~/miniconda3/etc/profile.d/conda.sh
	conda activate gpu_energy
else
	source ~/miniconda3/bin/activate gpu_energy
fi

cd /home/gargia/coalesced-memory-tb/param_identification
make analyze

cd /home/gargia/coalesced-memory-tb/calibration
make analyze_metrics
python plot_energy.py
