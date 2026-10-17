# Computational Chemistry & Electronic Structure Onboarding Repository Blueprint

This blueprint defines a complete, modular GitHub repository (`group-onboarding-bootcamp`) designed to onboard new undergraduate, graduate, and postdoctoral researchers into a computational quantum chemistry and electronic structure theory group. It takes students from basic Linux shell navigation and high-performance computing (HPC) cluster workflows on [Mississippi State University HPC² clusters](https://www.google.com/search?q=https%3A%2F%2Fwww.hpc.msstate.edu%2Fcomputing%2Fhpc.php) through vectorized [NumPy](https://www.google.com/search?q=https%3A%2F%2Fnumpy.org%2Fneps%2Fnep-0029-deprecation_policy.html) tensor contractions, coding a Restricted Hartree–Fock (RHF) self-consistent field (SCF) solver with Pulay's [Direct Inversion in the Iterative Subspace (DIIS)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FTutorials%2F03_Hartree-Fock%2F3b_rhf-diis.ipynb) from scratch, running production calculations in [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Findex.html), and constructing entangled bath orbitals via Schmidt decomposition as a bridge to quantum embedding research.

## 1. Complete Repository Directory Layout

```text
group-onboarding-bootcamp/
├── README.md                                      # Syllabus, 4-week timeline, and Git PR workflow
├── environment.yml                                # Reproducible Conda/Miniforge environment spec
├── pyproject.toml                                 # Pytest configuration and Ruff linter settings
├── .gitignore                                     # Ignores scratch files (*.chk, *.npy, slurm-*.out)
├── .github/
│   └── workflows/
│       └── ci-autograde.yml                       # Automated pytest runner on student Pull Requests
│
├── 01_bash_and_git/
│   ├── README.md                                  # Walkthrough: shell navigation, pipes, awk/grep, Git
│   ├── cheatsheet_bash_git.md                     # Quick reference for bash one-liners and Git branches
│   ├── sample_logs/
│   │   ├── h2o_sto3g_converged.out                # Sample converged SCF output log
│   │   ├── nh3_def2svp_converged.out              # Sample converged SCF output log
│   │   └── fe_complex_unconverged.out             # Sample unconverged SCF output log (for debugging)
│   └── project1_log_parser/
│       ├── INSTRUCTIONS.md                        # Project 1 prompt: automated log scraper
│       ├── parse_scf_logs.sh                      # Student starter script (Bash + awk/grep)
│       └── test_parser.sh                         # Verification script checking CSV output
│
├── 02_cluster_and_slurm/
│   ├── README.md                                  # Walkthrough: SSH keys, ProxyJump, tmux, Lmod, Slurm
│   ├── templates/
│   │   ├── ssh_config.example                     # ~/.ssh/config template for cluster login nodes
│   │   ├── bashrc_cluster.example                 # Module loads, OMP_NUM_THREADS, scratch aliases
│   │   ├── submit_pyscf_cpu.slurm                 # Annotated multi-core CPU Slurm batch script
│   │   └── submit_pytorch_gpu.slurm               # Annotated A100 GPU Slurm batch script
│   └── project2_slurm_array/
│       ├── INSTRUCTIONS.md                        # Project 2 prompt: parallel bond-stretch job array
│       ├── run_single_point.py                    # Python driver taking geometry index from $SLURM_ARRAY_TASK_ID
│       ├── submit_pes_array.slurm                 # Student starter Slurm array script (#SBATCH --array=0-20)
│       └── collect_pes_results.sh                 # Post-processing script to merge array outputs
│
├── 03_python_scientific_computing/
│   ├── README.md                                  # Walkthrough: vectorization, eigh, np.einsum, pytest
│   ├── notebooks/
│   │   ├── 01_numpy_broadcasting_and_linalg.ipynb # Matrix diagonalization, S^(-1/2), and condition numbers
│   │   └── 02_einsum_integral_transforms.ipynb    # O(N^8) vs O(N^5) 4-index AO-to-MO ERI transformation
│   └── project3_huckel_and_morse/
│       ├── INSTRUCTIONS.md                        # Project 3 prompt: Hückel solver & Morse PES fitting
│       ├── huckel_solver.py                       # Student starter code for pi-conjugated systems
│       ├── fit_morse_pes.py                       # Non-linear least squares fit of diatomic dissociation
│       └── test_project3.py                       # Pytest suite verifying Hückel eigenvalues & Morse parameters
│
├── 04_coding_hartree_fock/
│   ├── README.md                                  # Walkthrough: Roothaan-Hall equations, SCF loop, and DIIS
│   ├── data/
│   │   ├── generate_ao_integrals.py               # Exports S, T, V, ERI, and E_nuc from PySCF to .npz
│   │   ├── h2o_sto3g_integrals.npz                # Precomputed H2O / STO-3G AO integrals (offline fallback)
│   │   └── heh_plus_sto3g_integrals.npz           # Precomputed HeH+ / STO-3G AO integrals
│   ├── src/
│   │   ├── __init__.py
│   │   ├── integrals.py                           # Helper to load .npz or compute PySCF integrals on the fly
│   │   ├── rhf_solver.py                          # Project 4A: Scaffolded Restricted Hartree-Fock solver
│   │   ├── diis.py                                # Project 4B: Scaffolded Pulay DIIS accelerator
│   │   └── mp2_extension.py                       # Bonus Project 4C: 10-line MP2 correlation energy via einsum
│   └── tests/
│       └── test_rhf.py                            # Pytest suite comparing student RHF/DIIS/MP2 against PySCF
│
└── 05_electronic_structure_calculations/
    ├── README.md                                  # Walkthrough: PySCF workflows, basis sets, RHF vs UHF, DMET
    ├── examples/
    │   ├── 01_water_single_point_and_mulliken.py  # Running RHF/DFT, inspecting MO coefficients & charges
    │   ├── 02_basis_set_convergence.py            # Comparing STO-3G, 6-31G*, cc-pVDZ, cc-pVTZ, aug-cc-pVTZ
    │   ├── 03_h2_dissociation_rhf_vs_uhf.py       # Plotting RHF vs UHF spin symmetry breaking (Coulson-Fischer)
    │   └── 04_orca_qchem_input_templates/         # Standalone input deck examples for Q-Chem and ORCA
    ├── project5_ci_mp2_ccsd_and_be/
    │   ├── INSTRUCTIONS.md                        # Project 5 prompt: CI, MP2, CCSD & Cumulant Bootstrap Embedding
    │   ├── correlated_methods_ci_mp2_ccsd.py      # Project 5A: FCI/CISD, MP2, and Stanton-Gauss CCSD from scratch
    │   ├── schmidt_bath_construction.py           # Project 5B: Schmidt decomposition + cumulant-based BE energy
    │   └── test_ci_mp2_ccsd_be.py                 # Pytest suite verifying CCSD==FCI, size-extensivity & BE cumulant
│
└── 06_md_and_surface_hopping/
    ├── README.md                                  # Walkthrough: Velocity Verlet BOMD, NACVs, and Tully FSSH
    ├── examples/
    │   ├── 01_pyscf_ase_bomd_water.py             # Ab initio BOMD of H2O using PySCF gradients + ASE VelocityVerlet
    │   └── 02_avoided_crossing_surfaces.py        # Visualizing diabatic vs adiabatic curves and NACV spikes
    ├── project6_fssh_tully/
    │   ├── INSTRUCTIONS.md                        # Project 6 prompt: Morse NVE MD + Tully Model I FSSH
    │   ├── fssh_tully_solver.py                   # Student starter & reference code for Velocity Verlet and FSSH
    │   └── test_md_fssh.py                        # Pytest suite checking NVE energy drift, NACVs, and FSSH hops
│
└── 07_mace_and_quantum_dynamics/
    ├── README.md                                  # Walkthrough: MACE architecture, MD active learning, and MCTDH
    ├── examples/
    │   ├── 01_mace_foundation_md.py               # Running MACE-OFF23 / MACE-MP-0 with ASE VelocityVerlet
    │   └── 02_train_mace_slurm.sh                 # Slurm GPU script for mace_run_train with SWA loss scheduling
    ├── project7_mace_and_mctdh/
    │   ├── mace_md_pipeline.py                    # Project 7A: Export MD extxyz datasets & committee uncertainty
    │   ├── mctdh_2d_solver.py                     # Project 7B: 2D MCTDH wavepacket solver with time-dependent SPFs
    │   └── test_mace_mctdh.py                     # Pytest suite verifying extxyz forces and MCTDH vs exact DVR
│
└── 08_tddmrg_and_heom/
    ├── README.md                                  # Walkthrough: MPS tensor networks (TD-DMRG/TEBD) and HEOM
    └── project8_tddmrg_and_heom/
        ├── tddmrg_mps_solver.py                   # Project 8A: 1D MPS exciton/polaron transport with SVD truncation
        ├── heom_spin_boson_solver.py              # Project 8B: Tanimura-Kubo HEOM spin-boson decoherence solver
        └── test_tddmrg_heom.py                    # Pytest suite verifying MPS vs exact diagonalization & HEOM tiers
```

## 2. File-by-File Overview and Pedagogical Progression

| Module | File Path | Primary Concept Taught | Hands-On Student Deliverable |
| :--- | :--- | :--- | :--- |
| **Root** | `README.md` | Group workflow, Git branching, onboarding milestones | Fork/branch creation (`onboarding/<username>`) and first PR |
| **Root** | `environment.yml` | Reproducible scientific environments via [Miniforge](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fconda-forge%2Fminiforge) | Local and cluster Conda environment setup |
| **Module 01** | `01_bash_and_git/README.md` | Shell navigation, pipes, `grep`, `awk`, `sed`, permissions | Interactive shell drills on sample quantum chemistry logs |
| **Module 01** | `project1_log_parser/parse_scf_logs.sh` | Automated text extraction and exit-code error checking | Working CLI script outputting `summary_energies.csv` |
| **Module 02** | `02_cluster_and_slurm/README.md` | SSH keys, `~/.ssh/config`, `tmux`, Lmod, [Slurm](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fquickstart.html), `/scratch` | Passwordless SSH login to [MSU HPC²](https://www.google.com/search?q=https%3A%2F%2Fdocs.hpc.msstate.edu%2Fcluster%2Findex.html) (`Orion`/`Morrill`) |
| **Module 02** | `project2_slurm_array/submit_pes_array.slurm` | [Slurm job arrays](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fsbatch.html) (`$SLURM_ARRAY_TASK_ID`) and scratch I/O | 21-point parallel $\text{H}_2\text{O}$ stretch scan run on compute nodes |
| **Module 03** | `03_python_scientific_computing/README.md` | [NumPy](https://www.google.com/search?q=https%3A%2F%2Fnumpy.org%2Fneps%2Fnep-0029-deprecation_policy.html) broadcasting, `scipy.linalg.eigh`, `np.einsum` | Converting $\mathcal{O}(N^8)$ loops to $\mathcal{O}(N^5)$ tensor contractions |
| **Module 03** | `project3_huckel_and_morse/huckel_solver.py` | Generalized eigenvalue problems and [Pytest](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytest.org%2Fen%2Flatest%2F) unit tests | Verified Hückel $\pi$-orbital solver and Morse curve fitter |
| **Module 04** | `04_coding_hartree_fock/README.md` | Roothaan–Hall equations, $S^{-1/2}$ orthogonalization, [DIIS](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FTutorials%2F03_Hartree-Fock%2F3b_rhf-diis.ipynb) | Step-by-step derivation from AO integrals to converged SCF |
| **Module 04** | `src/rhf_solver.py` & `src/diis.py` | Implementing RHF, Pulay DIIS, and $\mathcal{O}(N^5)$ MP2 from scratch | Passing `pytest` suite within $10^{-8}\text{ E}_\text{h}$ of [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Fscf.html) |
| **Module 05** | `05_electronic_structure_calculations/README.md` | Production [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fquickstart.html), basis sets, RHF vs. UHF static correlation | Dissociation curves of $\text{H}_2$ and $\text{H}_2\text{O}$ PES comparison |
| **Module 05** | `project5_capstone_embedding/schmidt_bath_construction.py` | Löwdin orthogonalization and SVD of 1-RDM (Schmidt bath) | Constructing exact fragment+bath embedding subspaces |
| **Module 06** | `06_md_and_surface_hopping/README.md` | [Velocity Verlet BOMD](https://www.google.com/search?q=https%3A%2F%2Fdocs.matlantis.com%2Fatomistic-simulation-tutorial%2Fen%2F6_1_md-nve.html), NACVs ([pyscf.nac](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fpyscf_api_docs%2Fpyscf.nac.html)), & [Tully FSSH](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Facsodf%2Farticle%2F7%2F50%2F45810%2F461699%2FPedagogical-Overview-of-the-Fewest-Switches) | Understanding adiabatic vs. nonadiabatic dynamics, frustrated hops, and decoherence |
| **Module 06** | `project6_fssh_tully/fssh_tully_solver.py` & `test_md_fssh.py` | Coding Velocity Verlet NVE and Tully Model I surface hopping | Simulating excited-state branching ratios and passing conservation unit tests |

## 3. Root Configuration Files

### `README.md` (Root Repository Syllabus)

```markdown
# Welcome to the Research Group Onboarding Bootcamp

Welcome to the group! Before diving into research on quantum embedding, machine-learned potentials, or quantum dynamics, every new group member completes this self-paced, 4-week onboarding bootcamp. Rather than reading manuals passively, you will learn by building real tools—culminating in writing your own Restricted Hartree–Fock (RHF) electronic structure code from scratch and running production calculations on the cluster.

## How to Use This Repository

1. **Create Your Branch**: Clone this repository and create a personal working branch named `onboarding/<your-last-name>`:
   ```bash
   git clone git@github.com:<your-org>/group-onboarding-bootcamp.git
   cd group-onboarding-bootcamp
   git checkout -b onboarding/your-last-name
   ```
2. **Set Up Your Python Environment**: Install [Miniforge](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fconda-forge%2Fminiforge) and build the group environment:
   ```bash
   conda env create -f environment.yml
   conda activate qchem-onboarding
   ```
3. **Work Through Modules 01–05**: Each folder contains a `README.md` tutorial and a hands-on project folder (`project1_...` through `project5_...`).
4. **Open Pull Requests for Code Review**: As you complete each module's project, commit your work and open a Pull Request (PR). GitHub Actions will automatically run `pytest` on your code, and a senior group member will pair-review your PR with you.

## 4-Week Onboarding Schedule

* **Week 1 (Modules 01 & 02)**: Linux command line, regex log parsing in Bash, SSH keys, and submitting Slurm job arrays on the cluster.
* **Week 2 (Module 03)**: Scientific Python, NumPy linear algebra (`eigh`), 4-index tensor contractions (`np.einsum`), and unit testing (`pytest`).
* **Week 3 (Module 04)**: Coding Restricted Hartree–Fock (RHF), Pulay DIIS convergence acceleration, and MP2 from scratch.
* **Week 4 (Module 05)**: Running production calculations in PySCF, analyzing RHF vs. UHF bond dissociation, and performing a Schmidt decomposition of the 1-RDM to construct quantum embedding bath orbitals.
```

### `environment.yml`

```yaml
name: qchem-onboarding
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy>=1.26
  - scipy>=1.12
  - matplotlib>=3.8
  - pandas>=2.2
  - jupyterlab>=4.1
  - pytest>=8.0
  - ruff>=0.3
  - pyscf>=2.5
  - h5py>=3.10
  - pip
```

## 4. Module 01: Bash, Linux Command Line & Git (`01_bash_and_git/`)

### `01_bash_and_git/README.md` Walkthrough Outline

* **Why Chemists Need the Shell**: Compute clusters do not run graphical desktops by default. Managing thousands of single-point geometries, checking SCF convergence across directories, and automating workflows requires fluency in `bash`.
* **Core Navigation & File Inspection**: `pwd`, `ls -lah`, `cd`, `mkdir -p`, `cp -r`, `mv`, and safe deletion practices (`rm -i`), plus viewing large output files without loading gigabytes into RAM using `less`, `head -n 20`, and `tail -f`.
* **Search, Pipes & Text Processing for Quantum Chemistry**:
  * Finding converged energies with `grep`: `grep "SCF Done" *.out` or `grep "converged SCF energy" *.out`.
  * Extracting specific columns with `awk`:
    ```bash
    grep "converged SCF energy" *.out | awk '{print $1, $5}'
    ```
  * Batch replacing basis sets or functionals across template inputs with `sed`:
    ```bash
    sed 's/BASIS_PLACEHOLDER/cc-pVTZ/g' template.inp > run_ccpvtz.inp
    ```
* **Environment Variables & Permissions**: Understanding `chmod +x script.sh`, `#!/usr/bin/env bash`, `set -euo pipefail`, `$PATH`, `$PYTHONPATH`, and controlling multi-threading via `export OMP_NUM_THREADS=4`.

### `01_bash_and_git/project1_log_parser/parse_scf_logs.sh` (Starter & Solution Structure)

```bash
#!/usr/bin/env bash
# Project 1: Parse a directory of quantum chemistry output files (.out)
# Usage: ./parse_scf_logs.sh ../sample_logs summary_energies.csv
set -euo pipefail

LOG_DIR="${1:-../sample_logs}"
OUT_CSV="${2:-summary_energies.csv}"

echo "molecule,status,scf_energy_hartree,iterations,wall_time_sec" > "$OUT_CSV"

for logfile in "$LOG_DIR"/*.out; do
    mol_name=$(basename "$logfile" .out)
    
    # TODO 1: Check whether "SCF CONVERGED" appears in $logfile using grep -q
    if grep -q "SCF CONVERGED" "$logfile"; then
        status="CONVERGED"
        # TODO 2: Use grep and awk to extract the final SCF energy (in Hartree)
        energy=$(grep "Final SCF Energy:" "$logfile" | tail -n 1 | awk '{print $4}')
        # TODO 3: Count how many "SCF Iteration" lines occurred
        iters=$(grep -c "SCF Iteration" "$logfile")
    else
        status="FAILED"
        energy="NaN"
        iters=$(grep -c "SCF Iteration" "$logfile" || true)
    fi
    
    # TODO 4: Extract wall time in seconds from "Total Wall Time: X.XX s"
    wall_time=$(grep "Total Wall Time:" "$logfile" | awk '{print $4}')
    
    echo "${mol_name},${status},${energy},${iters},${wall_time}" >> "$OUT_CSV"
done

echo "Wrote summary to $OUT_CSV"
```

## 5. Module 02: SSH, HPC Clusters & Slurm (`02_cluster_and_slurm/`)

### `02_cluster_and_slurm/README.md` Walkthrough Outline

* **Connecting via SSH & Key-Based Authentication**:
  * Generating an Ed25519 key pair (`ssh-keygen -t ed25519 -C "your_netid@msstate.edu"`), copying the public key (`ssh-copy-id`), and configuring `~/.ssh/config` so `ssh orion` or `ssh morrill` connects cleanly ([NOAA/MSU SSH Guide](https://www.google.com/search?q=https%3A%2F%2Fdocs.rdhpcs.noaa.gov%2Fconnecting%2Findex.html)).
  * Using `tmux` (`tmux new -s research`, `Ctrl+b d`, `tmux attach -t research`) so interactive transfers or compilation jobs survive dropped Wi-Fi connections.
* **The Golden Rule of HPC Clusters**:
  * **Never run calculations on the login node.** Login nodes are shared by hundreds of users for editing files and submitting jobs. Any calculation taking more than a few seconds or multiple cores belongs in a [Slurm](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fquickstart.html) job (`sbatch` or `salloc`).
* **Cluster Storage Hierarchy ([MSU HPC² Documentation](https://www.google.com/search?q=https%3A%2F%2Fdocs.hpc.msstate.edu%2Fcluster%2Findex.html))**:
  * **Home (`/home/$USER`)**: Small quota, backed up. Store source code, Git repos, and small scripts here.
  * **Work / Project (`/work/...`)**: High-capacity Lustre parallel filesystem ([MSU-HPC User Guide](https://www.google.com/search?q=https%3A%2F%2Fdocs.rdhpcs.noaa.gov%2Fsystems%2FMSU-HPC_user_guide.html)) for active project datasets and output logs.
  * **Node-Local Scratch (`$TMPDIR` or `/scratch`)**: Fast local NVMe/SSD on the compute node. Always point 2-electron integral scratch files (`PYSCF_TMPDIR=$TMPDIR`) here so heavy disk I/O does not bottleneck the shared network filesystem.
* **Environment Modules (`Lmod`) & Slurm Commands**:
  * `module avail`, `module load python/3.11`, `module list`, `module purge`.
  * Submitting jobs with [sbatch](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fsbatch.html), checking queue status with [squeue -u \$USER](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fsqueue.html), canceling with `scancel <jobid>`, and inspecting memory/CPU efficiency of finished jobs with `seff <jobid>` or `sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State`.

### `02_cluster_and_slurm/templates/submit_pyscf_cpu.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=pyscf_rhf_test
#SBATCH --account=chem_theory          # Replace with group allocation account
#SBATCH --partition=normal             # Cluster partition (e.g., orion / morrill CPU queue)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # 8 OpenMP threads for PySCF integral evaluation
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.err

set -euo pipefail

# 1. Load clean environment and activate Conda
module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qchem-onboarding

# 2. Configure OpenMP threading and fast node-local scratch for integrals
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYSCF_TMPDIR="${TMPDIR:-/tmp}/${USER}_${SLURM_JOB_ID}"
mkdir -p "$PYSCF_TMPDIR"

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "Using ${OMP_NUM_THREADS} threads and scratch dir ${PYSCF_TMPDIR}"

# 3. Run calculation
python3 run_calculation.py

# 4. Clean up temporary integral files
rm -rf "$PYSCF_TMPDIR"
echo "Job finished at $(date)"
```

### `02_cluster_and_slurm/project2_slurm_array/INSTRUCTIONS.md` Summary

Students write a Slurm job array script (`#SBATCH --array=0-20`) that passes `${SLURM_ARRAY_TASK_ID}` to `run_single_point.py`. Each array task computes a single O–H bond length $R \in [0.70, 1.70]\text{ \AA}$ of $\text{H}_2\text{O}$ independently in parallel, writes `results/energy_${SLURM_ARRAY_TASK_ID}.csv`, and then a lightweight bash script concatenates and sorts the 21 points into a complete potential energy curve.

## 6. Module 03: Scientific Python & Tensor Contractions (`03_python_scientific_computing/`)

### `03_python_scientific_computing/README.md` Walkthrough Outline

* **Why Loops Kill Quantum Chemistry Performance**: In an atomic orbital (AO) basis of size $N$, the 2-electron repulsion integral (ERI) tensor $(\mu\nu|\lambda\sigma)$ has $N^4$ elements. Transforming ERIs from the AO basis to the molecular orbital (MO) basis via naive 8-nested Python `for`-loops scales as $\mathcal{O}(N^8)$ and takes hours even for $N=15$.
* **Factorized $\mathcal{O}(N^5)$ Transformation with `np.einsum`**:
  $$(pq|rs) = \sum_{\mu\nu\lambda\sigma} C_{\mu p} C_{\nu q} C_{\lambda r} C_{\sigma s} (\mu\nu|\lambda\sigma)$$
  By contracting one index at a time using `np.einsum(..., optimize=True)`, [NumPy](https://www.google.com/search?q=https%3A%2F%2Fnumpy.org%2Fneps%2Fnep-0029-deprecation_policy.html) dispatches the contraction to compiled BLAS matrix multiplications (`DGEMM`), reducing scaling from $\mathcal{O}(N^8)$ to $4 \times \mathcal{O}(N^5)$ and running in milliseconds.
* **Hermitian Eigenvalue Problems**: Using [SciPy](https://www.google.com/search?q=https%3A%2F%2Fdocs.scipy.org%2Fdoc%2Fscipy%2Fdev%2Ftoolchain.html)'s `scipy.linalg.eigh(F, S)` vs. explicit symmetric orthogonalization $S^{-1/2} = U s^{-1/2} U^\dagger$, and handling near-linear dependencies in diffuse basis sets by dropping eigenvalues $s_k < 10^{-7}$.

### `03_python_scientific_computing/project3_huckel_and_morse/huckel_solver.py`

```python
# Project 3A: Hückel Molecular Orbital Solver for Conjugated Pi Systems
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


def build_huckel_hamiltonian(
    n_carbons: int,
    alpha: float = -11.4,
    beta: float = -0.8,
    cyclic: bool = False,
) -> np.ndarray:
    """Construct the N x N Hückel Hamiltonian matrix without Python for-loops."""
    if n_carbons < 2:
        raise ValueError("n_carbons must be at least 2.")

    # Construct H using vectorized indexing without explicit loops
    H = np.full((n_carbons, n_carbons), 0.0, dtype=np.float64)
    np.fill_diagonal(H, alpha)
    idx = np.arange(n_carbons - 1)
    H[idx, idx + 1] = beta
    H[idx + 1, idx] = beta

    if cyclic and n_carbons > 2:
        H[0, -1] = beta
        H[-1, 0] = beta

    return H


def solve_pi_system(H: np.ndarray, n_electrons: int) -> dict[str, float | np.ndarray]:
    """Diagonalize H and compute total pi-electron energy and HOMO-LUMO gap."""
    evals, evecs = eigh(H)
    n_occ = n_electrons // 2
    total_energy = 2.0 * float(np.sum(evals[:n_occ]))
    if n_electrons % 2 == 1:
        total_energy += float(evals[n_occ])
    homo_lumo_gap = float(evals[n_occ] - evals[n_occ - 1])
    return {
        "orbital_energies": evals,
        "mo_coefficients": evecs,
        "total_pi_energy": total_energy,
        "homo_lumo_gap": homo_lumo_gap,
    }
```

## 7. Module 04: Coding Restricted Hartree–Fock from Scratch (`04_coding_hartree_fock/`)

This is the centerpiece project of the onboarding bootcamp, modeled after classic quantum chemistry pedagogy (Szabo & Ostlund and the [Psi4NumPy Hartree–Fock Tutorials](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FTutorials%2F03_Hartree-Fock%2F3b_rhf-diis.ipynb)). Students write a complete, self-contained Restricted Hartree–Fock (RHF) solver and Pulay DIIS accelerator and verify it to $10^{-8}\text{ E}_\text{h}$ against [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Fscf.html).

### `04_coding_hartree_fock/README.md` Step-by-Step Mathematical Walkthrough

1. **Extract Atomic Orbital (AO) Integrals**:
   For a molecule with $N_{\text{elec}}$ electrons ($n_{\text{occ}} = N_{\text{elec}} / 2$ doubly occupied spatial orbitals) and $N_{\text{bf}}$ basis functions $\{\chi_\mu\}$, we obtain:
   * Overlap matrix: $S_{\mu\nu} = \langle \chi_\mu | \chi_\nu \rangle$
   * Kinetic energy matrix: $T_{\mu\nu} = \langle \chi_\mu | -\frac{1}{2}\nabla^2 | \chi_\nu \rangle$
   * Nuclear attraction matrix: $V_{\mu\nu} = \langle \chi_\mu | -\sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|} | \chi_\nu \rangle$
   * Two-electron repulsion integrals (ERIs) in chemist's notation: $(\mu\nu|\lambda\sigma) = \iint \chi_\mu(\mathbf{r}_1)\chi_\nu(\mathbf{r}_1) \frac{1}{r_{12}} \chi_\lambda(\mathbf{r}_2)\chi_\sigma(\mathbf{r}_2)\,d\mathbf{r}_1 d\mathbf{r}_2$
   * Core Hamiltonian: $H^{\text{core}} = T + V$

2. **Symmetric Orthogonalization of the AO Basis ($S^{-1/2}$)**:
   Because atomic orbitals on different centers overlap ($S \neq I$), the Roothaan–Hall equation $FC = SC\epsilon$ is a generalized eigenvalue problem. Diagonalizing $S = U s U^\dagger$ allows us to construct the symmetric orthogonalization matrix:
   $$X = S^{-1/2} = U s^{-1/2} U^\dagger$$
   Transforming into the orthogonalized basis via $C' = X^{-1} C$ turns $FC = SC\epsilon$ into a standard eigenvalue equation $F' C' = C' \epsilon$, where $F' = X^\dagger F X$.

3. **Initial Core Guess & Density Matrix**:
   Setting $F^{(0)} = H^{\text{core}}$, we transform $F'^{(0)} = X^\dagger H^{\text{core}} X$, diagonalize to obtain initial MO coefficients $C^{(0)} = X C'^{(0)}$, and form the one-particle occupied AO density matrix:
   $$D_{\mu\nu} = \sum_{i=1}^{n_{\text{occ}}} C_{\mu i} C_{\nu i}$$

4. **Building the Fock Matrix via `np.einsum`**:
   At iteration $k$, the Coulomb matrix $J$ and Exchange matrix $K$ are contracted with the density matrix $D$:
   $$J_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma} (\mu\nu|\lambda\sigma), \qquad K_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma} (\mu\lambda|\nu\sigma)$$
   $$F_{\mu\nu} = H_{\mu\nu}^{\text{core}} + 2 J_{\mu\nu} - K_{\mu\nu}$$
   In vectorized Python:
   ```python
   J = np.einsum("pqrs,rs->pq", eri, D, optimize=True)
   K = np.einsum("prqs,rs->pq", eri, D, optimize=True)
   F = H_core + 2.0 * J - K
   ```

5. **Evaluating the Electronic & Total Energy**:
   $$E_{\text{elec}} = \sum_{\mu\nu} D_{\mu\nu} \left( H_{\mu\nu}^{\text{core}} + F_{\mu\nu} \right), \qquad E_{\text{total}} = E_{\text{elec}} + E_{\text{nuc}}$$

6. **Convergence Acceleration via Pulay DIIS**:
   Simple fixed-point iteration often oscillates or converges slowly. Following [Pulay's DIIS algorithm](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FTutorials%2F03_Hartree-Fock%2F3b_rhf-diis.ipynb), at each step $i$ we store the Fock matrix $F_i$ and the orthonormalized orbital gradient error vector:
   $$e_i = X^\dagger \left( F_i D_i S - S D_i F_i \right) X$$
   We solve the linear system $\sum_j B_{ij} c_j - \lambda = 0$ subject to $\sum_i c_i = 1$, where $B_{ij} = \text{Tr}(e_i^\dagger e_j)$, and extrapolate $F_{\text{DIIS}} = \sum_i c_i F_i$ before diagonalizing.

### `04_coding_hartree_fock/src/rhf_solver.py` (Student Template + Reference Implementation)

```python
# Module 04: Restricted Hartree-Fock (RHF) Solver from Scratch
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


@dataclass
class RHFResult:
    converged: bool
    e_total: float
    e_elec: float
    e_nuc: float
    orbital_energies: np.ndarray
    mo_coeff: np.ndarray
    density_matrix: np.ndarray
    n_iterations: int
    energy_history: list[float]


def symmetric_orthogonalization(S: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Compute X = S^(-1/2) = U @ diag(s^(-1/2)) @ U.T for overlap matrix S."""
    evals, evecs = eigh(S)
    keep = evals > tol
    s_inv_sqrt = 1.0 / np.sqrt(evals[keep])
    X = (evecs[:, keep] * s_inv_sqrt) @ evecs[:, keep].T
    return X


def compute_density_matrix(C: np.ndarray, n_occ: int) -> np.ndarray:
    """Compute unscaled occupied density matrix D_uv = sum_{i=1}^{n_occ} C_{u,i} C_{v,i}."""
    C_occ = C[:, :n_occ]
    return np.einsum("ui,vi->uv", C_occ, C_occ, optimize=True)


def build_fock_matrix(H_core: np.ndarray, eri: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Construct RHF Fock matrix F = H_core + 2*J - K using np.einsum."""
    J = np.einsum("uvls,ls->uv", eri, D, optimize=True)
    K = np.einsum("ulvs,ls->uv", eri, D, optimize=True)
    return H_core + 2.0 * J - K


def compute_orbital_gradient(
    F: np.ndarray, D: np.ndarray, S: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Compute orthonormalized Pulay DIIS residual e = X.T @ (FDS - SDF) @ X."""
    fds = F @ D @ S
    return X.T @ (fds - fds.T) @ X


def diis_extrapolate(
    fock_list: list[np.ndarray], error_list: list[np.ndarray]
) -> np.ndarray:
    """Solve the Pulay DIIS linear equations to extrapolate the Fock matrix."""
    n_diis = len(fock_list)
    B = np.full((n_diis + 1, n_diis + 1), -1.0, dtype=np.float64)
    B[-1, -1] = 0.0
    for i in range(n_diis):
        for j in range(i, n_diis):
            val = float(np.einsum("ab,ab->", error_list[i], error_list[j]))
            B[i, j] = val
            B[j, i] = val

    rhs = np.zeros(n_diis + 1, dtype=np.float64)
    rhs[-1] = -1.0
    coeffs = np.linalg.solve(B, rhs)[:-1]

    F_diis = np.zeros_like(fock_list[0])
    for c, F_mat in zip(coeffs, fock_list):
        F_diis += c * F_mat
    return F_diis


def run_rhf(
    S: np.ndarray,
    T: np.ndarray,
    V: np.ndarray,
    eri: np.ndarray,
    e_nuc: float,
    n_elec: int,
    max_iter: int = 100,
    e_tol: float = 1e-9,
    d_tol: float = 1e-7,
    use_diis: bool = True,
    diis_max_space: int = 6,
) -> RHFResult:
    """Execute the Roothaan-Hall Restricted Hartree-Fock SCF procedure."""
    if n_elec % 2 != 0:
        raise ValueError("RHF requires an even number of electrons.")
    n_occ = n_elec // 2

    H_core = T + V
    X = symmetric_orthogonalization(S)

    # Core guess: diagonalize H_core in orthogonalized AO basis
    F_prime = X.T @ H_core @ X
    eps, C_prime = eigh(F_prime)
    C = X @ C_prime
    D = compute_density_matrix(C, n_occ)
    e_elec_old = 0.0

    fock_list: list[np.ndarray] = []
    error_list: list[np.ndarray] = []
    energy_history: list[float] = []

    for iteration in range(1, max_iter + 1):
        F = build_fock_matrix(H_core, eri, D)
        e_elec = float(np.einsum("uv,uv->", D, H_core + F))
        e_total = e_elec + e_nuc
        energy_history.append(e_total)

        grad = compute_orbital_gradient(F, D, S, X)
        rms_grad = float(np.sqrt(np.mean(grad**2)))
        delta_e = abs(e_elec - e_elec_old)

        if iteration > 1 and delta_e < e_tol and rms_grad < d_tol:
            return RHFResult(
                converged=True,
                e_total=e_total,
                e_elec=e_elec,
                e_nuc=e_nuc,
                orbital_energies=eps,
                mo_coeff=C,
                density_matrix=D,
                n_iterations=iteration,
                energy_history=energy_history,
            )

        if use_diis:
            fock_list.append(F.copy())
            error_list.append(grad.copy())
            if len(fock_list) > diis_max_space:
                fock_list.pop(0)
                error_list.pop(0)
            if iteration >= 2:
                F = diis_extrapolate(fock_list, error_list)

        F_prime = X.T @ F @ X
        eps, C_prime = eigh(F_prime)
        C = X @ C_prime
        D = compute_density_matrix(C, n_occ)
        e_elec_old = e_elec

    return RHFResult(
        converged=False,
        e_total=e_elec_old + e_nuc,
        e_elec=e_elec_old,
        e_nuc=e_nuc,
        orbital_energies=eps,
        mo_coeff=C,
        density_matrix=D,
        n_iterations=max_iter,
        energy_history=energy_history,
    )
```

### `04_coding_hartree_fock/tests/test_rhf.py` (Automated Verification against PySCF)

```python
# Automated Pytest verification comparing student RHF solver against PySCF
import numpy as np
import pytest
from pyscf import gto, scf
from src.rhf_solver import run_rhf, symmetric_orthogonalization


@pytest.mark.parametrize(
    "atom_spec,basis",
    [
        ("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "sto-3g"),
        ("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "6-31g"),
        ("N 0 0 0; H 0 0 1.01; H 0.95 0 -0.33; H -0.47 0.82 -0.33", "sto-3g"),
    ],
)
def test_rhf_matches_pyscf(atom_spec: str, basis: str) -> None:
    mol = gto.M(atom=atom_spec, basis=basis, verbose=0)
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    e_nuc = float(mol.energy_nuc())

    # Verify X.T @ S @ X == I
    X = symmetric_orthogonalization(S)
    np.testing.assert_allclose(X.T @ S @ X, np.eye(S.shape[0]), atol=1e-10)

    # Run student RHF solver
    res = run_rhf(S, T, V, eri, e_nuc, n_elec=mol.nelectron, use_diis=True)
    assert res.converged, "Student RHF solver failed to converge."

    # Run reference PySCF RHF
    mf = scf.RHF(mol).run()
    assert np.isclose(res.e_total, mf.e_tot, atol=1e-8)
    np.testing.assert_allclose(res.orbital_energies, mf.mo_energy, atol=1e-6)
```

## 8. Module 05: Production Electronic Structure & Bridge to Quantum Embedding (`05_electronic_structure_calculations/`)

### `05_electronic_structure_calculations/README.md` Walkthrough Outline

* **From Your Own Code to [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fquickstart.html)**: Now that students know what happens inside every SCF cycle, they learn how to run fast, density-fitted RHF, UHF, DFT (`B3LYP`, `PBE0`, `wB97X-D`), MP2, and Coupled Cluster (`CCSD`, `CCSD(T)`) calculations in [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Fscf.html) (alongside input templates for [ORCA](https://www.google.com/search?q=https%3A%2F%2Fwww.faccts.de%2Forca%2F) and [Q-Chem](https://www.google.com/search?q=https%3A%2F%2Fmanual.q-chem.com%2Flatest%2Fqchem_manual.pdf)).
* **When Mean-Field Theory Fails (RHF vs. UHF on $\text{H}_2$ Dissociation)**:
  * Students run `examples/03_h2_dissociation_rhf_vs_uhf.py` to plot the $\text{H}_2$ potential energy curve from $0.5\text{ \AA}$ to $4.0\text{ \AA}$.
  * They observe firsthand why RHF overestimates the dissociation limit (spurious ionic terms $\text{H}^+\text{H}^-$) and how Unrestricted Hartree–Fock ([UHF](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FTutorials%2F03_Hartree-Fock%2F3c_unrestricted-hartree-fock.ipynb)) lowers the energy past the Coulson–Fischer point at the cost of spin contamination ($\langle \hat{S}^2 \rangle \approx 1$).
* **Capstone Project 5 — Schmidt Decomposition & Quantum Embedding Bath Orbitals (`schmidt_bath_construction.py`)**:
  * To transition students directly into the group's research on **Density Matrix Embedding Theory (DMET) and Bootstrap Embedding (BE)**, students perform a [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Findex.html) RHF calculation on an $\text{H}_{10}$ ring in a minimal basis.
  * They transform the 1-electron reduced density matrix ($P = C_{\text{occ}} C_{\text{occ}}^\dagger$) into Löwdin symmetrically orthogonalized atomic orbitals ($\tilde{P} = S^{1/2} P S^{1/2}$).
  * Selecting a 2-atom fragment $A$ ($N_A = 2$ orbitals) and environment $B$ ($N_B = 8$ orbitals), they take the off-diagonal coupling block $\tilde{P}_{BA}$ and compute its Singular Value Decomposition (SVD):
    $$\tilde{P}_{BA} = U \Sigma V^\dagger$$
  * They verify numerically that **at most $N_A = 2$ singular values are non-zero**, proving that the entire 8-orbital environment's entanglement with the 2-orbital fragment is captured exactly by just 2 bath orbitals—reducing a 10-orbital correlated calculation to a 4-orbital active embedding space ($2\text{ fragment} + 2\text{ bath}$).

### `05_electronic_structure_calculations/project5_capstone_embedding/schmidt_bath_construction.py`

```python
# Project 5 Capstone: Construct DMET/Bootstrap Embedding Bath Orbitals via SVD
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh, svd
from pyscf import gto, scf


def build_h_ring(n_atoms: int = 10, radius: float = 1.8) -> gto.Mole:
    """Create a symmetric ring of hydrogen atoms."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_atoms, endpoint=False)
    coords = [
        f"H {radius * np.cos(a):.6f} {radius * np.sin(a):.6f} 0.0" for a in angles
    ]
    return gto.M(atom="; ".join(coords), basis="sto-3g", verbose=0)


def construct_embedding_subspace(
    mol: gto.Mole, frag_indices: list[int], svd_tol: float = 1e-8
) -> dict[str, np.ndarray | int]:
    """Compute Löwdin-orthogonalized 1-RDM and extract entangled bath orbitals via SVD."""
    mf = scf.RHF(mol).run()
    S = mol.intor("int1e_ovlp")

    # 1. Construct S^(1/2) for Löwdin orthogonalized AO basis (OAO)
    evals, evecs = eigh(S)
    S_half = (evecs * np.sqrt(evals)) @ evecs.T

    # 2. Form spin-free 1-RDM in OAO basis: P_oao = C_oao,occ @ C_oao,occ.T (idempotent: P^2 = P)
    n_occ = mol.nelectron // 2
    C_oao_occ = S_half @ mf.mo_coeff[:, :n_occ]
    P_oao = C_oao_occ @ C_oao_occ.T

    # 3. Partition OAO indices into Fragment (A) and Environment (B)
    n_ao = S.shape[0]
    env_indices = [i for i in range(n_ao) if i not in frag_indices]
    P_env_frag = P_oao[np.ix_(env_indices, frag_indices)]

    # 4. Perform SVD on the environment-fragment density block P_{BA} = U * Sigma * V^T
    U_env, singular_vals, _ = svd(P_env_frag, full_matrices=False)
    active_bath_mask = singular_vals > svd_tol
    n_bath = int(np.sum(active_bath_mask))
    U_bath = U_env[:, active_bath_mask]

    # 5. Assemble the N_ao x (N_frag + N_bath) embedding rotation matrix R_emb
    n_frag = len(frag_indices)
    R_emb = np.zeros((n_ao, n_frag + n_bath), dtype=np.float64)
    for col, ao_idx in enumerate(frag_indices):
        R_emb[ao_idx, col] = 1.0
    for col in range(n_bath):
        R_emb[env_indices, n_frag + col] = U_bath[:, col]

    return {
        "P_oao": P_oao,
        "singular_values": singular_vals,
        "n_frag": n_frag,
        "n_bath": n_bath,
        "R_emb": R_emb,
        "P_emb": R_emb.T @ P_oao @ R_emb,
    }


if __name__ == "__main__":
    mol_ring = build_h_ring(n_atoms=10, radius=1.8)
    emb = construct_embedding_subspace(mol_ring, frag_indices=[0, 1])
    print(f"Total AOs: {mol_ring.nao_nr()}")
    print(f"Fragment orbitals: {emb['n_frag']} -> Entangled bath orbitals: {emb['n_bath']}")
    print(f"Schmidt singular values: {np.round(emb['singular_values'], 6)}")
```


## 9. Module 06: Adiabatic Molecular Dynamics & Tully's Surface Hopping (`06_md_and_surface_hopping/`)

See `06_md_and_surface_hopping_guide.md`, `fssh_tully_solver.py`, and `test_md_fssh.py` for the full walkthrough and reference implementation covering:
* **Adiabatic Born–Oppenheimer MD (BOMD)**: Symplectic [Velocity Verlet integration](https://www.google.com/search?q=https%3A%2F%2Fdocs.matlantis.com%2Fatomistic-simulation-tutorial%2Fen%2F6_1_md-nve.html) on a Morse potential and [PySCF/ASE interfaces](https://www.google.com/search?q=https%3A%2F%2Fyang.chem.wisc.edu%2Fwp-content%2Fuploads%2Fsites%2F1145%2F2023%2F12%2FHow-to-CNEO-MD.pdf).
* **Nonadiabatic Coupling Vectors (NACVs)**: Analytical vs. finite-difference derivative couplings $\mathbf{d}_{jk}(\mathbf{R}) = \langle \Phi_j | \nabla_{\mathbf{R}} \Phi_k \rangle$ ([pyscf.nac](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fpyscf_api_docs%2Fpyscf.nac.html), [SHARC](https://www.google.com/search?q=https%3A%2F%2Fsharc-md.org%2F), and [Newton-X](https://www.google.com/search?q=https%3A%2F%2Fnewtonx.org%2F)).
* **Tully's Fewest-Switches Surface Hopping ([FSSH](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Facsodf%2Farticle%2F7%2F50%2F45810%2F461699%2FPedagogical-Overview-of-the-Fewest-Switches))**: Coherent RK4 electronic sub-stepping, Tully switching probabilities $g_{k \to j}$, momentum rescaling along $\mathbf{d}_{kj}$, frustrated hop handling, and Granucci–Persico decoherence damping on Tully's Single Avoided Crossing (Model I).


## 10. Module 07: Machine-Learned Potentials (MACE) & Quantum Dynamics (MCTDH) (`07_mace_and_quantum_dynamics/`)

See `07_mace_and_quantum_dynamics_guide.md`, `mace_md_pipeline.py`, `mctdh_2d_solver.py`, and `test_mace_mctdh.py` for the full walkthrough and reference implementations covering:
* **Using & Training [MACE](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2F) on MD Data**: Higher-order equivariant message passing ([ACEsuit/mace](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Facesuit%2Fmace)), formatting Extended XYZ (`extxyz`) files with `IsolatedAtom` reference energies (`REF_energy`, `REF_forces`), running [mace\_run\_train](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2Fen%2Flatest%2Fguide%2Ftraining.html) with Stochastic Weight Averaging (`--swa`), and active-learning committee force uncertainty $\sigma_F$.
* **Multidimensional Quantum Dynamics & [MCTDH](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F11%2F114107%2F315442%2FA-rectangular-collocation-multi-configuration-time)**: Overcoming the $\mathcal{O}(N^f)$ direct-product DVR grid bottleneck via Tucker tensor decomposition with time-dependent [Single-Particle Functions (SPFs)](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F148%2F4%2F044115%2F75514%2FA-new-collocation-based-multi-configuration-time), deriving the Dirac–Frenkel equations of motion, comparing mean-field Time-Dependent Hartree ($n=1$) against correlated MCTDH ($n=4$), and connecting to [Multi-Layer MCTDH (ML-MCTDH) and TD-DMRG](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F17%2F174106%2F200048%2FComparison-of-the-multi-layer-multi-configuration).


## 11. Module 08: Tensor Network Dynamics (TD-DMRG) & Open Quantum Systems (HEOM) (`08_tddmrg_and_heom/`)

See `08_tddmrg_and_heom_guide.md`, `tddmrg_mps_solver.py`, `heom_spin_boson_solver.py`, and `test_tddmrg_heom.py` for the full walkthrough and reference implementations covering:
* **Matrix Product States (MPS) & [TD-DMRG](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2Fabs%2F10.1021%2Facs.jctc.8b00628)**: Factorizing an $L$-site chain into rank-3 tensors $A^{[k]s_k}_{\alpha_{k-1},\alpha_k}$, propagating with two-site Suzuki–Trotter [TEBD](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1706.09537) / [TDVP](https://www.google.com/search?q=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC8889605%2F) gates, compressing bond dimension $M$ via SVD, and tracking bipartite von Neumann entanglement entropy $S_{\text{vN}}(t)$ and mean-squared displacement $\text{MSD}(t)$.
* **Hierarchical Equations of Motion ([HEOM](https://www.google.com/search?q=https%3A%2F%2Fqutip.org%2Fdocs%2F4.7%2Fguide%2Fguide-heom.html))**: Propagating the Tanimura–Kubo hierarchy of Auxiliary Density Operators $\hat{\rho}_n(t)$ for a spin qubit coupled to an overdamped Drude–Lorentz phonon bath ($J(\omega) = 2\lambda\gamma\omega / (\omega^2 + \gamma^2)$), capturing non-Markovian memory and phonon back-flow beyond Redfield/Lindblad master equations.
