#!/usr/bin/env bash
# Module 07 Example 02: Launch MACE Equivariant Potential Training with SWA Loss Scheduling
# Usage: bash 02_train_mace_slurm.sh [--dry-run]
set -euo pipefail

DRY_RUN="${1:-}"

CMD=(
    mace_run_train
    --name="water_mace_model"
    --train_file="water_md_train.xyz"
    --valid_file="water_md_valid.xyz"
    --energy_key="REF_energy"
    --forces_key="REF_forces"
    --E0s="average"
    --model="MACE"
    --hidden_irreps="64x0e + 64x1o"
    --r_max=5.0
    --correlation=3
    --loss="weighted"
    --energy_weight=1.0
    --forces_weight=100.0
    --max_num_epochs=250
    --swa
    --start_swa=180
    --batch_size=10
    --device="cuda"
    --seed=42
)

if [[ "$DRY_RUN" == "--dry-run" ]] || ! command -v mace_run_train >/dev/null 2>&1; then
    echo "[Dry-Run / Offline Mode] MACE training command verified:"
    printf '  %s \\\n' "${CMD[@]}"
else
    "${CMD[@]}"
fi