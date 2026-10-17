#!/usr/bin/env bash
# Automated verification script for Module 01 (parse_scf_logs.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_CSV="$(mktemp /tmp/test_summary_XXXXXX.csv)"
trap 'rm -f "$TMP_CSV"' EXIT

bash "$SCRIPT_DIR/parse_scf_logs.sh" "$SCRIPT_DIR" "$TMP_CSV" >/dev/null

header=$(head -n 1 "$TMP_CSV")
expected_header="molecule,status,scf_energy_hartree,scf_iterations,wall_time_sec"
if [[ "$header" != "$expected_header" ]]; then
    echo "FAIL: Header mismatch. Got '$header', expected '$expected_header'"
    exit 1
fi

grep -q "^h2o_sto3g_converged,CONVERGED,-74.96302314,7,0.42$" "$TMP_CSV" || {
    echo "FAIL: Missing or incorrect h2o_sto3g_converged row in CSV"
    cat "$TMP_CSV"
    exit 1
}

grep -q "^nh3_def2svp_converged,CONVERGED,-56.15449921,8,1.85$" "$TMP_CSV" || {
    echo "FAIL: Missing or incorrect nh3_def2svp_converged row in CSV"
    cat "$TMP_CSV"
    exit 1
}

grep -q "^fe_complex_unconverged,UNCONVERGED,NaN,10,14.70$" "$TMP_CSV" || {
    echo "FAIL: Missing or incorrect fe_complex_unconverged row in CSV"
    cat "$TMP_CSV"
    exit 1
}

echo "PASS: Module 01 log parser verified on all 3 sample logs!"