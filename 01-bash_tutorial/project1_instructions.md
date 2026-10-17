# Project 1 Instructions: Automated Quantum Chemistry Log Scraper

## Objective
Write and verify a command-line Bash script (`parse_scf_logs.sh`) that scans a directory of quantum chemistry output logs (`*.out`), identifies which calculations converged and which failed to converge, and compiles a clean summary table (`summary_energies.csv`).

## Background
When running high-throughput potential energy scans or catalyst screening campaigns on [MSU HPC² clusters](https://docs.hpc.msstate.edu/cluster/index.html), you will routinely generate hundreds of `.out` log files. Manually opening each file to check whether the Self-Consistent Field (SCF) cycle converged is error-prone. A 20-line shell script using `grep` and `awk` can audit an entire directory in milliseconds.

## Specifications
1. **Input Arguments**:
   * `$1`: Directory containing `.out` files (defaults to `./sample_logs`).
   * `$2`: Output CSV filename (defaults to `summary_energies.csv`).
2. **CSV Header**:
   ```text
   molecule,status,scf_energy_hartree,scf_iterations,wall_time_sec
   ```
3. **Verification**:
   ```bash
   chmod +x parse_scf_logs.sh test_parser.sh
   ./test_parser.sh
   ```
