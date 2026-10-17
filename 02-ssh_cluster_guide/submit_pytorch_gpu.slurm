#!/bin/bash
#SBATCH --job-name=mace_gpu_train
#SBATCH --account=chem_theory          # Group allocation account
#SBATCH --partition=gpu                # GPU partition (NVIDIA A100 / H100 nodes)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.err

set -euo pipefail

module purge
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate qchem-onboarding
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_VISIBLE_DEVICES=0

echo "GPU Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"
nvidia-smi || true

python3 mace_md_pipeline.py
bash 02_train_mace_slurm.sh --dry-run
echo "GPU Job finished at $(date)"
