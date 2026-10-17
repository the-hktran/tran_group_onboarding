#!/usr/bin/env bash
# ==============================================================================
# Module 02 Project (Student Starter with #TODOs): Merge Slurm Array PES Outputs
# Usage: ./collect_pes_results_todo.bash [results_dir] [output_csv]
# ==============================================================================
set -euo pipefail

RESULTS_DIR="${1:-./pes_results}"
OUT_CSV="${2:-water_pes_scan.csv}"

# TODO 1: Write the CSV header "task_id,r_oh_angstrom,e_rhf_hartree,e_mp2_hartree" to "$OUT_CSV"
echo "task_id,r_oh_angstrom,e_rhf_hartree,e_mp2_hartree" > "$OUT_CSV"

shopt -s nullglob
files=("$RESULTS_DIR"/pes_point_*.csv)
if [ ${#files[@]} -eq 0 ]; then
    echo "Error: No pes_point_*.csv files found in $RESULTS_DIR" >&2
    exit 1
fi

# TODO 2: Concatenate all "${files[@]}", sort numerically by task_id (column 1, comma-delimited),
#         and append to "$OUT_CSV" (hint: sort -t, -k1,1n)
# YOUR CODE HERE:


# TODO 3: Find the row with the minimum MP2 energy (column 4, general numeric sort -k4,4g)
#         and print the equilibrium bond length R_OH and minimum MP2 energy
# YOUR CODE HERE: