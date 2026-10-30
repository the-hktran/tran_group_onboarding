# Tran Group Onboarding (`tran_group_onboarding`)

Welcome to the Tran Research Group! This repository contains a self-paced onboarding curriculum covering Linux command-line workflows, high-performance computing (HPC) cluster job submission, scientific Python, electronic structure theory, molecular dynamics, and open-system quantum dynamics.

Each module is self-contained and uses interactive Jupyter notebooks (`# TODO` coding exercises with automated checkpoint assertions at the bottom of each notebook) powered by [PySCF](https://www.google.com/search?q=https://pyscf.org/user/gto.html), NumPy, SciPy, and Matplotlib.

## Quick Start & Environment Setup

1. **Follow the setup guide**: Read `03-python_basics/vs_code_and_conda_setup_guide.md` to configure VS Code, Conda/Miniforge, and Jupyter.
2. **Create and activate the environment**:
   ```bash
   conda create -n tran-onboarding python=3.11 numpy scipy matplotlib pandas pytest h5py jupyterlab -c conda-forge
   conda activate tran-onboarding
   pip install pyscf ase mace-torch
   ```
3. **Work through the modules in order (`01` through `07`)**: Complete the `_todo` scripts in Modules `01`–`02` and fill in the `# TODO` functions in each Jupyter notebook (`03`–`07`) until all verification assertions pass.

## Repository Structure & Curriculum

| Directory | Files | Topics Covered |
| :--- | :--- | :--- |
| **`01-bash_tutorial/`** | `01_bash_tutorial.md`, `cheatsheet_bash_git.md`, `project1_instructions.md`, `parse_scf_logs_todo.bash`, `parse_scf_logs.bash`, `test_parser.bash`, `sample_logs/` (`h2o_sto3g_converged.out`, `nh3_def2svp_converged.out`, `fe_complex_unconverged.out`) | Shell navigation, pipes, `grep`/`awk`/`sed`, Git basics, and automated scraping of quantum chemistry SCF output logs |
| **`02-ssh_cluster_guide/`** | `02_ssh_cluster_guide.md`, `ssh_config.example`, `bashrc_cluster.example`, `submit_pyscf_cpu.slurm`, `submit_pytorch_gpu.slurm`, `project2_instructions.md`, `run_single_point_todo.py`, `run_single_point.py`, `collect_pes_results_todo.bash`, `collect_pes_results.bash` | SSH key configuration, Slurm CPU/GPU batch scripts, node-local scratch management, and running a parallel potential energy surface (PES) scan |
| **`03-python_basics/`** | `vs_code_and_conda_setup_guide.md`, `03_modern_quantum_chemistry_ch1_lab.ipynb` | Conda/VS Code setup, NumPy broadcasting, `np.einsum` tensor contractions, and linear algebra fundamentals from Chapter 1 of Szabo & Ostlund |
| **`04-hartree_fock/`** | `04_hartree_fock_scf_notebook.ipynb` | Extracting AO integrals (`int1e_ovlp`, `int1e_kin`, `int1e_nuc`, `int2e`) with [PySCF](https://www.google.com/search?q=https://github.com/pyscf/pyscf-doc/blob/master/examples/gto/20-ao_integrals.py), symmetric orthogonalization ($S^{-1/2}$), Roothaan–Hall SCF loop, [Pulay DIIS](https://www.google.com/search?q=https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3b_rhf-diis.ipynb), dipole moments, and Mulliken/Löwdin populations ([Crawford Project 03](https://www.google.com/search?q=https://pycrawfordprogproj.readthedocs.io/en/latest/Project_03/Project_03.html)) |
| **`05-correlated_electronic_structure/`** | `05a`–`05g` notebooks (see table below) | Post-Hartree–Fock wavefunction correlation, quantum embedding, periodic solids, analytical nuclear gradients, production [PySCF](https://www.google.com/search?q=https://pyscf.org/user/scf.html)/[ORCA 6](https://www.faccts.de/docs/orca/6.0/manual/_downloads/f5a34d500b44971b2b057d96d7f899ca/orca.pdf) workflows, multireference methods, and Green's function methods |
| **`06-md_and_surface_hopping/`** | `06a_md_and_surface_hopping_notebook.ipynb`, `06b_mace_mlip_md_notebook.ipynb` | Velocity Verlet Born–Oppenheimer MD, Tully's Fewest Switches Surface Hopping (FSSH) with decoherence corrections, and machine-learned interatomic potentials (MACE MD & active-learning committee uncertainty) |
| **`07-quantum_dynamics/`** | `07a`–`07d` notebooks (see table below) | Exact and open-system quantum dynamics: MCTDH, TD-DMRG, HEOM, Lindblad master equation, weak/strong-coupling polaron theory, and Redfield vibrational relaxation |
| **Root Configs** | `pyproject.toml`, `ci-autograde.yml`, `gitignore.txt`, `README.md` | Python package metadata, Ruff/Pytest settings, GitHub Actions CI autograding workflow, and Git ignore rules |

## Section 05: Correlated Electronic Structure Modules (`05-correlated_electronic_structure/`)

| Notebook | Focus & Key Exercises |
| :--- | :--- |
| **`05a_correlated_wavefunction_methods_notebook.ipynb`** | Two-step $\mathcal{O}(N^5)$ AO-to-MO integral transformation, Configuration Interaction (CISD), [Møller–Plesset perturbation theory (MP2 \& SCS-MP2)](https://www.google.com/search?q=https://pycrawfordprogproj.readthedocs.io/en/latest/Project_04/Project_04.html), and [spin-orbital CCSD](https://www.google.com/search?q=https://pycrawfordprogproj.readthedocs.io/en/latest/Project_05/Project_05.html) with DIIS |
| **`05b_special_correlated_methods_notebook.ipynb`** | [Bootstrap Embedding (BE)](https://www.google.com/search?q=https://pubs.acs.org/doi/10.1021/acs.jctc.4c01267) ([QuEmb](https://www.google.com/search?q=https://github.com/troyvvgroup/quemb)) via Löwdin localization, Schmidt SVD bath construction, and chemical potential/density matching, plus 1D Density Matrix Renormalization Group (DMRG) |
| **`05c_periodic_electronic_structure_notebook.ipynb`** | Crystal lattices, reciprocal lattice vectors, $k$-point Monkhorst–Pack meshes, plane-wave vs. crystalline Gaussian basis sets, and [periodic KRHF band structures](https://www.google.com/search?q=https://pyscf.org/user/pbc/scf.html) |
| **`05d_analytical_gradients_notebook.ipynb`** | Hellmann–Feynman vs. Pulay forces, energy-weighted density matrix $W_{\mu\nu}$, RHF analytical nuclear gradients, and [non-variational MP2 gradients](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00015) via Coupled-Perturbed Hartree–Fock (CPHF) and Handy–Schaefer $Z$-vector equations |
| **`05e_using_pyscf_and_orca_notebook.ipynb`** | Running production RHF, MP2, CISD, CCSD(T), density fitting (`density_fit()`), and UHF stability analysis in [PySCF](https://www.google.com/search?q=https://pyscf.org/user/scf.html), plus generating and parsing [ORCA 6](https://www.faccts.de/docs/orca/6.0/manual/_downloads/f5a34d500b44971b2b057d96d7f899ca/orca.pdf) (`DLPNO-CCSD(T)`, `OPT FREQ`, TD-DFT) jobs |
| **`05f_multireference_casscf_mrci_mrcc_notebook.ipynb`** | Static vs. dynamic correlation, natural orbital occupations ($n_p$), $\text{CASSCF}(n_e, n_o)$ orbital rotation gradients, uncontracted and internally contracted MRCI + Davidson ($+Q$) size-extensivity corrections, and state-universal/Mk-MRCC |
| **`05g_greens_functions_gw_bse_dmft_notebook.ipynb`** | Matsubara and retarded one-particle Green's functions, Dyson's equation, Hedin's pentagon, $G_0W_0$ quasiparticle energies, the Bethe–Salpeter Equation (BSE) for neutral excitons, and Dynamical Mean-Field Theory (DMFT) on the Hubbard dimer |

## Section 07: Quantum Dynamics Modules (`07-quantum_dynamics/`)

| Notebook | Focus & Key Exercises |
| :--- | :--- |
| **`07a_quantum_dynamics_mctdh_tddmrg_heom_notebook.ipynb`** | Wavepacket propagation using Multi-Configuration Time-Dependent Hartree (MCTDH), Time-Dependent DMRG (TD-DMRG / TEBD on matrix product states), and Hierarchical Equations of Motion (HEOM) |
| **`07b_lindblad_coupled_harmonic_oscillators.ipynb`** | Open quantum system dynamics of coupled harmonic oscillators via the Gorini–Kossakowski–Sudarshan–Lindblad (GKSL) master equation and Liouville-space vectorization |
| **`07c_polaron_weak_and_strong_coupling_notebook.ipynb`** | Holstein and Fröhlich electron–phonon Hamiltonians, Lang–Firsov canonical transformation, Franck–Condon bandwidth renormalization, weak-coupling Golden Rule mobility vs. strong-coupling Marcus small-polaron hopping, and Pekar variational large polarons |
| **`07d_redfield_vibrational_relaxation_notebook.ipynb`** | Vibrational energy relaxation ($T_1$) and pure dephasing ($T_2^*$) of an anharmonic chromophore in an excited electronic state using the Bloch–Redfield master equation, Bose–Einstein bath correlation functions, and the secular (Davies) approximation |

## How to Complete Exercises

1. **Bash & Cluster Projects (`01` & `02`)**: Read `project1_instructions.md` and `project2_instructions.md`, complete the `_todo` scripts (`parse_scf_logs_todo.bash`, `run_single_point_todo.py`, `collect_pes_results_todo.bash`), and verify your parser with `bash 01-bash_tutorial/test_parser.bash`.
2. **Jupyter Notebooks (`03`–`07`)**: Open each `.ipynb` notebook, implement every function marked with `# TODO` (replacing `raise NotImplementedError(...)`), and run the final verification cell to confirm all unit checkpoints pass.
