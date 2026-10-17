#!/usr/bin/env bash
# Module 01 Project: Batch-extract SCF energies, iteration counts, and wall times from .out logs
# Usage: ./parse_scf_logs.sh [log_directory] [output_csv]
set -euo pipefail

LOG_DIR="${1:-./sample_logs}"
OUT_CSV="${2:-summary_energies.csv}"

echo "molecule,status,scf_energy_hartree,scf_iterations,wall_time_sec" > "$OUT_CSV"

shopt -s nullglob
for logfile in "$LOG_DIR"/*.out; do
    mol_name=$(basename "$logfile" .out)

    if grep -q "SCF CONVERGED" "$logfile"; then
        status="CONVERGED"
        energy=$(grep "Final SCF Energy:" "$logfile" | tail -n 1 | awk '{print $4}')
        iters=$(grep -c "SCF Iteration" "$logfile")
    else
        status="UNCONVERGED"
        energy="NaN"
        iters=$(grep -c "SCF Iteration" "$logfile" || true)
    fi

    wall_time=$(grep "Total Wall Time:" "$logfile" | tail -n 1 | awk '{print $4}')
    echo "${mol_name},${status},${energy},${iters},${wall_time:-0.0}" >> "$OUT_CSV"
done

echo "Summary written to ${OUT_CSV}:"
column -t -s, "$OUT_CSV"