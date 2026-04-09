#!/bin/bash
#SBATCH --job-name=energy_sweep
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=job_output_%j.txt

# 1. Manually add CUDA to the path (adjust the path if your 'ls' command found a specific version like cuda-12.1)
export PATH=/usr/local/cuda/bin:$PATH

# 2. Activate your personal Conda installation
source ~/miniconda3/bin/activate gpu_energy

cd /home/gargia/GPU_energy_SOP/coalesced-memory-tb/param_identification

make clean
ARCH=sm_90 make
make run-a100
make analyze

cd /home/gargia/GPU_energy_SOP/coalesced-memory-tb

make clean
make sweep_csv BLOCKS=264
python plot_energy.py
