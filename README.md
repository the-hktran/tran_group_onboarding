# Computational Chemistry Research Group Onboarding Repository (`group-onboarding-bootcamp`)

Welcome to the research group! Before diving into research on quantum embedding ([Bootstrap Embedding](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F11nN9UOAku31tvnLTWJ3peB9QWuAi8jSb%2Fview%3Fusp%3Ddrivesdk) and [QuEmb](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Ftroyvvgroup%2Fquemb)), [machine-learned potentials](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Fopen%3Fid%3D1N8Zvvx-_w_A-D8v-S_jp7FlaDg7_DsSpn6BoEt8q97Y), or quantum dynamics, every new student completes this self-paced, 7-week onboarding curriculum. Rather than reading manuals passively, you will learn by building real tools—culminating in writing your own Restricted Hartree–Fock (RHF), Coupled Cluster (CCSD), Bootstrap Embedding, Surface Hopping, MACE, MCTDH, TD-DMRG, and HEOM solvers from scratch and running production calculations on our high-performance computing (HPC) clusters.

## Complete Repository File Manifest (All 62 Generated Files)

| File Name | Type | Module & Purpose |
| :--- | :--- | :--- |
| `README.md` | Markdown | Root syllabus, complete 62-file manifest, and 7-week onboarding schedule |
| `tran_group_onboarding_repo_blueprint.md` | Markdown | Master architectural blueprint and pedagogical overview |
| `environment.yml` | YAML | Reproducible [Miniforge](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fconda-forge%2Fminiforge) environment (`numpy`, `scipy`, `pyscf`, `pytest`) |
| `pyproject.toml` | TOML | Project metadata, [Pytest](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytest.org%2Fen%2Flatest%2F) test discovery, and Ruff linter configuration |
| `.gitignore` | Git Config | Ignores quantum chemistry scratch files (`*.chk`, `*.npy`, `slurm-*.out`) |
| `ci-autograde.yml` | YAML | GitHub Actions workflow running `ruff`, `test_parser.sh`, and `pytest` on PRs |
| `01_bash_tutorial.md` | Markdown | Module 01 walkthrough: shell navigation, pipes, `grep`, `awk`, `sed`, permissions |
| `cheatsheet_bash_git.md` | Markdown | Module 01 quick reference for Bash one-liners, `set -euo pipefail`, and Git PRs |
| `h2o_sto3g_converged.out` | Log File | Module 01 sample converged RHF/STO-3G water output log |
| `nh3_def2svp_converged.out` | Log File | Module 01 sample converged RHF/def2-SVP ammonia output log |
| `fe_complex_unconverged.out` | Log File | Module 01 sample unconverged transition-metal SCF log for debugging |
| `project1_instructions.md` | Markdown | Module 01 Project prompt: automated SCF log scraper |
| `parse_scf_logs.sh` | Bash Script | Module 01 Project: batch-extract SCF energies, iterations, and wall times |
| `test_parser.sh` | Bash Script | Module 01 automated verification script for `parse_scf_logs.sh` |
| `02_ssh_cluster_guide.md` | Markdown | Module 02 walkthrough: SSH keys, `~/.ssh/config`, `tmux`, Lmod, [Slurm](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fquickstart.html), `/scratch` |
| `ssh_config.example` | SSH Config | Module 02 `~/.ssh/config` template for [MSU HPC²](https://www.google.com/search?q=https%3A%2F%2Fdocs.hpc.msstate.edu%2Fcluster%2Findex.html) (`Orion`, `Hercules`, `Morrill`) |
| `bashrc_cluster.example` | Bash Config | Module 02 `~/.bashrc` template with safe `OMP_NUM_THREADS=1` & Slurm aliases |
| `submit_pyscf_cpu.slurm` | Slurm Script | Module 02 template for multi-core CPU [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Fscf.html) jobs using node-local `$TMPDIR` |
| `submit_pytorch_gpu.slurm` | Slurm Script | Module 02 template for single-GPU NVIDIA A100 jobs |
| `project2_instructions.md` | Markdown | Module 02 Project prompt: parallel 21-point $\text{H}_2\text{O}$ stretch Slurm job array |
| `submit_pes_array.slurm` | Slurm Script | Module 02 Project: parallel [Slurm job array](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fsbatch.html) (`#SBATCH --array=0-20`) |
| `run_single_point.py` | Python | Module 02 Project driver evaluating $\text{H}_2\text{O}$ stretch point from `$SLURM_ARRAY_TASK_ID` |
| `collect_pes_results.sh` | Bash Script | Module 02 Project post-processor merging array outputs into `water_pes_scan.csv` |
| `03_python_basics.md` | Markdown | Module 03 walkthrough: [NumPy](https://www.google.com/search?q=https%3A%2F%2Fnumpy.org%2Fneps%2Fnep-0029-deprecation_policy.html) broadcasting, `scipy.linalg.eigh`, `np.einsum` |
| `01_numpy_broadcasting_and_linalg.ipynb` | Notebook | Module 03 interactive notebook on vectorized $E_{\text{nuc}}$ and $S^{-1/2}$ condition numbers |
| `02_einsum_integral_transforms.ipynb` | Notebook | Module 03 interactive notebook benchmarking $\mathcal{O}(N^8)$ vs. $\mathcal{O}(N^5)$ ERI transforms |
| `project3_instructions.md` | Markdown | Module 03 Project prompt: vectorized Hückel solver & Morse curve fitting |
| `huckel_solver.py` | Python | Module 03A Project: loop-free Hückel $\pi$-electron solver & $\mathcal{O}(N^5)$ ERI transformer |
| `fit_morse_pes.py` | Python | Module 03B Project: non-linear least-squares Morse PES fitter & harmonic $\tilde{\nu}_e$ |
| `test_project3.py` | Python | Module 03 Automated Test Suite ([Pytest](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytest.org%2Fen%2Flatest%2F)) |
| `04_hartree_fock_walkthrough.md` | Markdown | Module 04 walkthrough: Roothaan–Hall equations, $S^{-1/2}$, SCF loop, and [Pulay DIIS](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FTutorials%2F03_Hartree-Fock%2F3b_rhf-diis.ipynb) |
| `generate_ao_integrals.py` | Python | Module 04 utility exporting $\text{HeH}^+$ and $\text{H}_2\text{O}$ AO integrals to `.npz` archives |
| `heh_plus_sto3g_integrals.npz` | NumPy `.npz` | Module 04 precomputed STO-3G AO integrals ($S, T, V, \text{ERI}, E_{\text{nuc}}$) for $\text{HeH}^+$ |
| `h2o_sto3g_integrals.npz` | NumPy `.npz` | Module 04 precomputed 7-AO STO-3G AO integrals for $\text{H}_2\text{O}$ |
| `integrals.py` | Python | Module 04 helper loading `.npz` integral archives |
| `rhf_solver.py` | Python | Module 04A Project: self-contained RHF + Pulay DIIS + MP2 solver from scratch |
| `diis.py` | Python | Module 04B Project: object-oriented `PulayDIIS` commutator extrapolation manager |
| `mp2_extension.py` | Python | Module 04C Bonus Project: canonical MP2 and Grimme SCS-MP2 correlation module |
| `test_rhf.py` | Python | Module 04 Automated Test Suite ([Pytest](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytest.org%2Fen%2Flatest%2F)) |
| `05_electronic_structure_guide.md` | Markdown | Module 05 walkthrough: CI, MP2, CCSD, and [Bootstrap Embedding](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2F10.1021%2Facs.jctc.4c01267) matching loops & cumulant energy |
| `01_water_single_point_and_mulliken.py` | Python | Module 05 example: single-point RHF & Mulliken vs. Löwdin population analysis |
| `02_basis_set_convergence.py` | Python | Module 05 example: basis set convergence (`STO-3G` to `cc-pVQZ`) & Helgaker CBS extrapolation |
| `03_h2_dissociation_rhf_vs_uhf.py` | Python | Module 05 example: $\text{H}_2$ dissociation curve, Coulson–Fischer point, & UHF $\langle \hat{S}^2 \rangle$ |
| `04_orca_qchem_input_templates.md` | Markdown | Module 05 production input deck templates for [ORCA](https://www.google.com/search?q=https%3A%2F%2Fwww.faccts.de%2Forca%2F) and [Q-Chem](https://www.google.com/search?q=https%3A%2F%2Fmanual.q-chem.com%2Flatest%2Fqchem_manual.pdf) |
| `project5_instructions.md` | Markdown | Module 05 Project prompt: CI/MP2/CCSD and Bootstrap Embedding matching loops |
| `correlated_methods_ci_mp2_ccsd.py` | Python | Module 05A Project: FCI/CISD, MP2, & Stanton–Gauss spin-orbital CCSD from scratch |
| `schmidt_bath_construction.py` | Python | Module 05B Project: Schmidt SVD bath, Newton–Raphson BE matching loop, & cumulant BE energy |
| `test_ci_mp2_ccsd_be.py` | Python | Module 05 Automated Test Suite ([Pytest](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytest.org%2Fen%2Flatest%2F)) |
| `06_md_and_surface_hopping_guide.md` | Markdown | Module 06 walkthrough: [Velocity Verlet BOMD](https://www.google.com/search?q=https%3A%2F%2Fdocs.matlantis.com%2Fatomistic-simulation-tutorial%2Fen%2F6_1_md-nve.html), NACVs ([pyscf.nac](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fpyscf_api_docs%2Fpyscf.nac.html)), & [Tully FSSH](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Facsodf%2Farticle%2F7%2F50%2F45810%2F461699%2FPedagogical-Overview-of-the-Fewest-Switches) |
| `01_pyscf_ase_bomd_water.py` | Python | Module 06 example: 3D microcanonical (NVE) Velocity Verlet BOMD of $\text{H}_2\text{O}$ |
| `02_avoided_crossing_surfaces.py` | Python | Module 06 example: scanning adiabatic surfaces $E_{0,1}(R)$ & NACV spikes $d_{01}(R)$ |
| `project6_instructions.md` | Markdown | Module 06 Project prompt: Morse NVE MD and Tully Model I surface hopping |
| `fssh_tully_solver.py` | Python | Module 06 Project: Velocity Verlet + Tully FSSH with momentum rescaling & decoherence |
| `test_md_fssh.py` | Python | Module 06 Automated Test Suite ([Pytest](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytest.org%2Fen%2Flatest%2F)) |
| `07_mace_and_quantum_dynamics_guide.md` | Markdown | Module 07 walkthrough: [MACE](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2F) equivariant potentials & [MCTDH](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F11%2F114107%2F315442%2FA-rectangular-collocation-multi-configuration-time) quantum dynamics |
| `01_mace_foundation_md.py` | Python | Module 07 example: MACE active-learning MD workflow & committee force uncertainty |
| `02_train_mace_slurm.sh` | Bash Script | Module 07 example: [mace\_run\_train](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2Fen%2Flatest%2Fguide%2Ftraining.html) launcher with two-stage SWA loss scheduling |
| `project7_instructions.md` | Markdown | Module 07 Project prompt: MACE `extxyz` pipeline and 2D MCTDH wavepacket propagation |
| `mace_md_pipeline.py` | Python | Module 07A Project: MD snapshot `extxyz` generator & committee uncertainty $\sigma_F$ |
| `mctdh_2d_solver.py` | Python | Module 07B Project: 2D [MCTDH](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F11%2F114107%2F315442%2FA-rectangular-collocation-multi-configuration-time) wavepacket solver vs. exact DVR & TDH |
| `test_mace_mctdh.py` | Python | Module 07 Automated Test Suite ([Pytest](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytest.org%2Fen%2Flatest%2F)) |
| `08_tddmrg_and_heom_guide.md` | Markdown | Module 08 walkthrough: [TD-DMRG](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2Fabs%2F10.1021%2Facs.jctc.8b00628) / MPS ([TEBD](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1706.09537) & [TDVP](https://www.google.com/search?q=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC8889605%2F)) and [HEOM](https://www.google.com/search?q=https%3A%2F%2Fqutip.org%2Fdocs%2F4.7%2Fguide%2Fguide-heom.html) open quantum systems |
| `project8_instructions.md` | Markdown | Module 08 Project prompt: 1D MPS exciton/polaron transport & non-Markovian HEOM |
| `tddmrg_mps_solver.py` | Python | Module 08A Project: 1D Matrix Product State (MPS) exciton/polaron transport solver |
| `heom_spin_boson_solver.py` | Python | Module 08B Project: Non-Markovian [HEOM](https://www.google.com/search?q=https%3A%2F%2Fqutip.org%2Fdocs%2F4.7%2Fguide%2Fguide-heom.html) spin–boson decoherence solver |
| `test_tddmrg_heom.py` | Python | Module 08 Automated Test Suite ([Pytest](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytest.org%2Fen%2Flatest%2F)) |

## 7-Week Onboarding Schedule

1. **Week 1 (Modules 01 & 02 — Linux, Bash, SSH & Slurm)**: Set up SSH keys, configure `~/.ssh/config`, complete `01_bash_tutorial.md` and `parse_scf_logs.sh` (verified by `test_parser.sh`), and run the 21-point parallel job array via `submit_pes_array.slurm`, `run_single_point.py`, and `collect_pes_results.sh`.
2. **Week 2 (Module 03 — Scientific Python & Tensor Contractions)**: Complete `03_python_basics.md`, work through `01_numpy_broadcasting_and_linalg.ipynb` and `02_einsum_integral_transforms.ipynb`, and pass all tests in `test_project3.py` for `huckel_solver.py` and `fit_morse_pes.py`.
3. **Week 3 (Module 04 — Coding Hartree–Fock from Scratch)**: Follow `04_hartree_fock_walkthrough.md` to code `rhf_solver.py`, `diis.py`, and `mp2_extension.py` using precomputed AO integrals from `generate_ao_integrals.py` and `integrals.py`, and pass `test_rhf.py`.
4. **Week 4 (Module 05 — Correlated Electronic Structure & Bootstrap Embedding)**: Complete `05_electronic_structure_guide.md`, run the examples (`01_water_single_point_and_mulliken.py`, `02_basis_set_convergence.py`, `03_h2_dissociation_rhf_vs_uhf.py`), implement MP2/CISD/CCSD in `correlated_methods_ci_mp2_ccsd.py`, and implement the Schmidt decomposition, Newton–Raphson density matching loop, and cumulant energy functional in `schmidt_bath_construction.py` (verified by `test_ci_mp2_ccsd_be.py`).
5. **Week 5 (Module 06 — Molecular Dynamics & Nonadiabatic Surface Hopping)**: Complete `06_md_and_surface_hopping_guide.md`, run `01_pyscf_ase_bomd_water.py` and `02_avoided_crossing_surfaces.py`, implement Velocity Verlet and Tully's FSSH in `fssh_tully_solver.py`, and pass `test_md_fssh.py`.
6. **Week 6 (Module 07 — Machine-Learned Potentials with MACE & Quantum Dynamics with MCTDH)**: Complete `07_mace_and_quantum_dynamics_guide.md`, generate MD training datasets and committee uncertainty metrics in `mace_md_pipeline.py`, propagate a 2D coupled wavepacket in `mctdh_2d_solver.py`, and pass `test_mace_mctdh.py`.
7. **Week 7 (Module 08 — Tensor Network Dynamics with TD-DMRG & Open Quantum Systems with HEOM)**: Complete `08_tddmrg_and_heom_guide.md`, simulate 1D polaron/exciton wavepacket spreading in `tddmrg_mps_solver.py`, model non-Markovian spin–phonon qubit decoherence in `heom_spin_boson_solver.py`, and pass `test_tddmrg_heom.py`.
