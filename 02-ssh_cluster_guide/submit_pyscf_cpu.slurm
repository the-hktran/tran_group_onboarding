#!/bin/bash
#SBATCH --job-name=pyscf_rhf_cpu
#SBATCH --account=chem_theory          # Group allocation account on MSU HPC²
#SBATCH --partition=normal             # CPU partition (e.g., orion / morrill)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # 8 OpenMP threads for PySCF integral evaluation
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.err

set -euo pipefail

module purge
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate qchem-onboarding
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYSCF_MAX_MEMORY=28000
export PYSCF_TMPDIR="${TMPDIR:-/tmp}/${USER}_${SLURM_JOB_ID:-local}"
mkdir -p "$PYSCF_TMPDIR"
trap 'rm -rf "$PYSCF_TMPDIR"' EXIT

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"
python3 01_water_single_point_and_mulliken.py
echo "Job completed cleanly at $(date)"
