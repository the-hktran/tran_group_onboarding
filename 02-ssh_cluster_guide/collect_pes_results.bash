#!/usr/bin/env bash
# Module 02 Project Post-Processor: Merge per-task Slurm array outputs into sorted CSV
# Usage: ./collect_pes_results.sh [results_dir] [output_csv]
set -euo pipefail

RESULTS_DIR="${1:-./pes_results}"
OUT_CSV="${2:-water_pes_scan.csv}"

echo "task_id,r_oh_angstrom,e_rhf_hartree,e_mp2_hartree" > "$OUT_CSV"

shopt -s nullglob
files=("$RESULTS_DIR"/pes_point_*.csv)
if [ ${#files[@]} -eq 0 ]; then
    echo "Error: No pes_point_*.csv files found in $RESULTS_DIR" >&2
    exit 1
fi

cat "${files[@]}" | sort -t, -k1,1n >> "$OUT_CSV"

min_row=$(tail -n +2 "$OUT_CSV" | sort -t, -k4,4g | head -n 1)
min_r=$(echo "$min_row" | awk -F, '{print $2}')
min_e=$(echo "$min_row" | awk -F, '{print $4}')
echo "Merged ${#files[@]} PES points into ${OUT_CSV}."
echo "Minimum MP2 energy: ${min_e} Eh at R_OH = ${min_r} Angstrom"