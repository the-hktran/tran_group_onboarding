# Module 01: Bash, Linux Command Line & Git for Computational Chemists

High-performance computing (HPC) clusters run Linux without a graphical desktop. Whether you are launching 500 single-point energy evaluations across a potential energy surface or checking if a geometry optimization converged, fluency in `bash` lets you automate hours of manual work in seconds.

## 1. Navigating the Filesystem & Inspecting Large Output Logs

* **Where am I and what is here?**: `pwd` prints the current directory; `ls -lah` lists files with human-readable sizes and hidden dotfiles (`.bashrc`, `.ssh`).
* **Creating & moving directories**: `mkdir -p project/calculations/scan_01`, `cp template.inp project/calculations/scan_01/`, `mv raw.out converged.out`.
* **Inspecting multi-gigabyte quantum chemistry logs safely**:
  * Never open a 2 GB output file in a heavy editor. Use `less output.out` (press `G` to jump to the end, `/SCF` to search forward, `?Energy` to search backward, `q` to quit).
  * Use `head -n 25 output.out` to inspect the input deck header and `tail -n 30 output.out` (or `tail -f output.out` during a live calculation) to watch SCF iterations converge.

## 2. Pipes, `grep`, `awk`, and `sed`

* **Extracting SCF Energies with `grep`**:
  ```bash
  grep "converged SCF energy" *.out
  grep -n "RMS Gradient" h2o_opt.out
  ```
* **Column Extraction & Math with `awk`**:
  `awk` splits each line on whitespace (`$1`, `$2`, ..., `$NF` for the last column):
  ```bash
  # Print filename and final energy in Hartree, and convert to eV
  grep "Final SCF Energy:" *.out | awk '{printf "%-25s %14.8f Eh (%10.4f eV)\n", $1, $4, $4 * 27.211386}'
  ```
* **Generating Input Decks with `sed`**:
  Replace a placeholder bond length `R_OH` across 20 geometries:
  ```bash
  for r in $(seq 0.80 0.05 1.80); do
      sed "s/R_OH/${r}/g" h2o_template.xyz > "h2o_${r}.xyz"
  done
  ```

## 3. Example Project 1: Automated SCF Log Scraper (`parse_scf_logs.sh`)

Open `parse_scf_logs.sh` and complete the `# TODO` blocks so the script loops over a directory of `.out` log files, checks whether `SCF CONVERGED` is present, extracts the final energy (in $\text{E}_\text{h}$), counts the number of SCF cycles, and writes a sorted CSV report (`summary_energies.csv`) while flagging unconverged jobs.

