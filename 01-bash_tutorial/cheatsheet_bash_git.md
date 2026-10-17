# Module 01 Quick Reference: Bash One-Liners & Git Workflow Cheatsheet

Keep this cheatsheet handy when navigating [Mississippi State University HPC² clusters](https://docs.hpc.msstate.edu/cluster/index.html) or managing your `onboarding/<last-name>` Git branch.

## 1. Essential Bash One-Liners for Quantum Chemistry Logs

| Task | Command |
| :--- | :--- |
| Find all converged SCF logs in current directory | `grep -l "SCF CONVERGED" *.out` |
| Find all **unconverged** SCF logs | `grep -L "SCF CONVERGED" *.out` |
| Extract final SCF energy (in $\text{E}_\text{h}$) from all `.out` files | `grep "Final SCF Energy:" *.out \| awk '{print $1, $4}'` |
| Convert Hartree energies to $\text{eV}$ ($1\text{ E}_\text{h} = 27.211386\text{ eV}$) | `grep "Final SCF Energy:" *.out \| awk '{printf "%s %.6f eV\n", $1, $4*27.211386}'` |
| Count SCF iterations in each `.out` file | `grep -c "SCF Iteration" *.out` |
| Sort molecules from lowest (most stable) to highest energy | `sort -t, -k3,3g summary_energies.csv` |
| Watch an active calculation log in real time | `tail -f slurm-12345.out` |
| Check disk usage of subdirectories sorted by size | `du -sh * \| sort -h` |
| Replace basis set across 20 input files in place | `sed -i 's/sto-3g/cc-pVTZ/g' scan_*.inp` |

## 2. Shell Safety Flags (`set -euo pipefail`)

Always start your production Bash and [Slurm](https://slurm.schedmd.com/quickstart.html) scripts with:
```bash
#!/usr/bin/env bash
set -euo pipefail
```
* **`-e`**: Exit immediately if any command returns a non-zero exit code.
* **`-u`**: Treat unset variables (e.g., a typo like `$SLURM_ARRAY_TASK_DI`) as an error instead of expanding to an empty string.
* **`-o pipefail`**: Ensure a pipeline `cmd1 | cmd2` fails if `cmd1` fails.

## 3. Git Branching & Pull Request Workflow

```bash
# 1. Clone the group onboarding repo and create your feature branch
git clone git@github.com:<your-org>/group-onboarding-bootcamp.git
cd group-onboarding-bootcamp
git checkout -b onboarding/your-last-name

# 2. Check modified and untracked files (verify no .chk or .npy scratch files are staged!)
git status

# 3. Stage specific project files and commit with a descriptive message
git add parse_scf_logs.sh test_parser.sh
git commit -m "Complete Module 01 SCF log parser and verification tests"

# 4. Sync with upstream main before pushing
git fetch origin
git rebase origin/main

# 5. Push your branch and open a Pull Request on GitHub
git push -u origin onboarding/your-last-name
```
