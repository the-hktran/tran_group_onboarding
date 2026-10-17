#!/usr/bin/env bash
# ==============================================================================
# Module 01 Student Starter Script: Automated Quantum Chemistry SCF Log Scraper
# File: parse_scf_log.bash
# Usage: ./parse_scf_log.bash [log_directory] [output_csv]
#
# Instructions:
#   Complete each of the # TODO blocks below using standard Linux command-line
#   utilities (echo, basename, grep, tail, awk) to parse all *.out files in
#   $LOG_DIR and compile a summary CSV table.
#
#   When you are finished, test your implementation against the sample logs:
#     chmod +x parse_scf_log.bash
#     ./parse_scf_log.bash . summary_energies.csv
# ==============================================================================
set -euo pipefail

LOG_DIR="${1:-./sample_logs}"
OUT_CSV="${2:-summary_energies.csv}"

# ------------------------------------------------------------------------------
# TODO 1: Write the CSV header line to "$OUT_CSV" (overwriting any existing file)
# Expected header format:
#   molecule,status,scf_energy_hartree,scf_iterations,wall_time_sec
# Hint: Use `echo "..." > "$OUT_CSV"`
# ------------------------------------------------------------------------------
echo "molecule,status,scf_energy_hartree,scf_iterations,wall_time_sec" > "$OUT_CSV"

shopt -s nullglob
for logfile in "$LOG_DIR"/*.out; do

    # --------------------------------------------------------------------------
    # TODO 2: Extract the molecule name from "$logfile" by stripping the
    # directory path and the ".out" suffix.
    # Hint: Use `basename "$logfile" .out`
    # --------------------------------------------------------------------------
    mol_name="TODO_MOLECULE_NAME"

    # --------------------------------------------------------------------------
    # TODO 3: Check whether the string "SCF CONVERGED" appears anywhere in
    # "$logfile" using `grep -q`.
    # --------------------------------------------------------------------------
    if false; then # <-- TODO 3: Replace `false` with `grep -q "SCF CONVERGED" "$logfile"`
        status="CONVERGED"

        # ----------------------------------------------------------------------
        # TODO 4: Extract the numerical SCF energy (4th whitespace-delimited
        # field) from the line matching "Final SCF Energy:" in "$logfile".
        # Sample log line:
        #   Final SCF Energy: -74.96302314 Eh
        # Hint: Pipe `grep "Final SCF Energy:" "$logfile" | tail -n 1 | awk '{print $4}'`
        # ----------------------------------------------------------------------
        energy="TODO_ENERGY"

        # ----------------------------------------------------------------------
        # TODO 5: Count the total number of SCF iterations by counting how many
        # lines contain "SCF Iteration" in "$logfile".
        # Hint: Use `grep -c "SCF Iteration" "$logfile"`
        # ----------------------------------------------------------------------
        iters="TODO_ITERS"
    else
        # ----------------------------------------------------------------------
        # TODO 6: Handle unconverged calculations:
        #   - Set status="UNCONVERGED"
        #   - Set energy="NaN"
        #   - Count how many "SCF Iteration" lines were completed before failure
        #     (use `grep -c "SCF Iteration" "$logfile" || true` so `set -e` does
        #     not exit if 0 iterations are found).
        # ----------------------------------------------------------------------
        status="UNCONVERGED"
        energy="NaN"
        iters="0"
    fi

    # --------------------------------------------------------------------------
    # TODO 7: Extract the numerical runtime in seconds (4th field) from the line
    # matching "Total Wall Time:" in "$logfile".
    # Sample log line:
    #   Total Wall Time: 0.42 s
    # Hint: Pipe `grep "Total Wall Time:" "$logfile" | tail -n 1 | awk '{print $4}'`
    # --------------------------------------------------------------------------
    wall_time="0.0"

    # --------------------------------------------------------------------------
    # TODO 8: Append the formatted CSV row to "$OUT_CSV":
    #   ${mol_name},${status},${energy},${iters},${wall_time:-0.0}
    # Hint: Use `>> "$OUT_CSV"` (append, do not overwrite!)
    # --------------------------------------------------------------------------
    echo "${mol_name},${status},${energy},${iters},${wall_time:-0.0}" >> "$OUT_CSV"

done

echo "Summary written to ${OUT_CSV}:"
if command -v column >/dev/null 2>&1; then
    column -t -s, "$OUT_CSV"
else
    cat "$OUT_CSV"
fi