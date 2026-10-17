# Module 07: Machine-Learned Interatomic Potentials (MACE) & Multidimensional Quantum Dynamics (MCTDH)

Simulating realistic chemical and materials systems requires overcoming two coupled bottlenecks: the **electronic bottleneck** (evaluating potential energy surfaces millions of times) and the **nuclear bottleneck** (propagating high-dimensional quantum wavepackets where tunneling, zero-point energy, and vibronic coupling matter). This module teaches you how to use and train **[MACE](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2F)** ($E(3)$-equivariant message-passing force fields) on molecular dynamics data, and how to solve the time-dependent Schrödinger equation using **Multi-Configuration Time-Dependent Hartree ([MCTDH](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F11%2F114107%2F315442%2FA-rectangular-collocation-multi-configuration-time))** tensor networks.

## 1. Equivariant Interatomic Potentials with [MACE](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Facesuit%2Fmace)

Traditional empirical force fields use fixed functional forms that cannot break bonds, while standard ab initio molecular dynamics (BOMD) scales as $\mathcal{O}(N_{\text{elec}}^3\text{–}N_{\text{elec}}^7)$ per time step. **[MACE](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2F)** (Higher-Order Equivariant Message Passing, developed by [Batatia, Csányi, and co-workers](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Facesuit%2Fmace)) bridges this gap by combining the systematic many-body expansion of the Atomic Cluster Expansion (ACE) with $E(3)$-equivariant graph neural networks ([J. Appl. Phys. overview of MACE-MP-0](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjap%2Farticle%2F139%2F4%2F041101%2F3378116%2FFine-tuning-universal-machine-learned-interatomic)):

* **$E(3)$ Equivariance ($L = 0, 1, 2$)**: Internal node features transform as spherical tensors $h_{i, kLM}^{(t)}$ under 3D rotations, translations, and inversions. Tensor products coupled via Clebsch–Gordan coefficients construct higher-body-order ($\nu$-body) messages in just 2 message-passing layers.
* **Conservative Forces by Analytical Backpropagation**: MACE predicts per-atom energies $E = \sum_i \varepsilon_i$, and atomic forces are obtained as exact analytical gradients via [PyTorch](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytorch.org%2Fdocs%2Fstable%2Findex.html) `autograd`:
  $$\mathbf{F}_i = -\nabla_{\mathbf{R}_i} E_{\text{MACE}}(\mathbf{R})$$
  guaranteeing strictly conservative dynamics ($\oint \mathbf{F}\cdot d\mathbf{R} = 0$) in NVE trajectories.
* **Using Pre-Trained Foundation Models (`MACE-MP-0` & `MACE-OFF23`)**:
  ```python
  from ase.build import molecule
  from ase.md.verlet import VelocityVerlet
  from ase import units
  from mace.calculators import mace_mp, mace_off

  atoms = molecule("H2O")
  # Use MACE-OFF23 for organic/molecular systems or mace_mp() for materials
  atoms.calc = mace_off(model="medium", device="cuda", default_dtype="float64")
  dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
  dyn.run(1000)
  ```

## 2. Training MACE on Molecular Dynamics Data & Active Learning

### Step 1: Curating Extended XYZ (`extxyz`) Datasets from MD Trajectories
Never train an interatomic potential only on equilibrium minimum-energy structures—an MD trajectory will immediately visit stretched bonds and fail! Instead:
1. Sample diverse configurations across temperatures ($300\text{ K}\text{–}1000\text{ K}$) or normal-mode rattle displacements.
2. Include **isolated single-atom reference energies (`config_type=IsolatedAtom`)** for every chemical element in your system so MACE learns the smooth atomization energy rather than huge total electronic energies.
3. Label energies (`REF_energy` in $\text{eV}$) and forces (`REF_forces` in $\text{eV/\AA}$) using a consistent level of electronic structure theory (e.g., DFT, MP2, CCSD, or quantum embedding) and store them in Extended XYZ format ([MACE Training Tutorial](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FACEsuit%2Fmace-tutorials%2Fblob%2Fmain%2FT01_MACE_Practice_I.ipynb)).

### Step 2: Training with `mace_run_train`
Once `train.xyz` and `valid.xyz` are generated (see `mace_md_pipeline.py`), launch [mace\_run\_train](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2Fen%2Flatest%2Fguide%2Ftraining.html) (or [fine-tune a foundation model](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2Fen%2Flatest%2Fguide%2Ffinetuning.html) using `--foundation_model="medium"` and optional [CUDA cuEquivariance acceleration](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2Fen%2Flatest%2Fguide%2Fcuda_acceleration.html)):

```bash
mace_run_train \
    --name="water_mace_model" \
    --train_file="water_md_train.xyz" \
    --valid_file="water_md_valid.xyz" \
    --test_file="water_md_test.xyz" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --E0s="average" \
    --model="MACE" \
    --hidden_irreps="64x0e + 64x1o" \
    --r_max=5.0 \
    --correlation=3 \
    --loss="weighted" \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --max_num_epochs=250 \
    --swa \
    --start_swa=180 \
    --batch_size=10 \
    --device="cuda" \
    --seed=42
```

* **Two-Stage Loss Scheduling (`--swa`)**: Early epochs weight forces heavily (`--forces_weight=100.0`) because each configuration provides $3N_{\text{atoms}}$ force components vs. $1$ scalar energy. Stochastic Weight Averaging (`--swa`) switches to high energy weighting in the final epochs to achieve $< 1\text{ meV/atom}$ energy RMSE.
* **Active Learning (Query by Committee)**: Train a committee of $M=3\text{–}4$ MACE models with different random seeds (`--seed=1,2,3`). During a MACE-driven MD trajectory, monitor the committee force disagreement:
  $$\sigma_{F, i}(\mathbf{R}) = \sqrt{\frac{1}{M} \sum_{m=1}^{M} \left\| \mathbf{F}_i^{(m)}(\mathbf{R}) - \bar{\mathbf{F}}_i(\mathbf{R}) \right\|^2}$$
  Whenever $\max_i \sigma_{F, i}(\mathbf{R})$ exceeds a threshold (e.g., $0.05\text{–}0.15\text{ eV/\AA}$), flag that geometry for a new ab initio calculation and add it to the training set.

## 3. Nuclear Quantum Dynamics & Multi-Configuration Time-Dependent Hartree ([MCTDH](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F11%2F114107%2F315442%2FA-rectangular-collocation-multi-configuration-time))

### The Exponential Curse of Dimensionality in Grid-Based Quantum Dynamics
Once a fast potential energy surface $V(q_1, \dots, q_f)$ is available, solving the nuclear time-dependent Schrödinger equation (TDSE) exactly on a direct-product Discrete Variable Representation (DVR) grid with $N$ points per coordinate expands the wavepacket as:
$$\Psi_{\text{grid}}(q_1, \dots, q_f, t) = \sum_{i_1=1}^{N} \cdots \sum_{i_f=1}^{N} C_{i_1 \dots i_f}(t) \prod_{\kappa=1}^{f} \chi_{i_\kappa}^{(\kappa)}(q_\kappa)$$
The coefficient tensor $C_{i_1 \dots i_f}(t)$ has $N^f$ complex amplitudes. Even for a 6-atom molecule ($f = 12$ vibrational modes) with $N = 16$ grid points per mode, $16^{12} \approx 2.8 \times 10^{14}$ complex numbers require $> 4.5\text{ PB}$ of RAM!

### The MCTDH Ansatz: Time-Dependent Tucker Tensor Decomposition
The **[Multi-Configuration Time-Dependent Hartree (MCTDH)](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F11%2F114107%2F315442%2FA-rectangular-collocation-multi-configuration-time)** method compresses this $N^f$ tensor by introducing a small number $n_\kappa \ll N$ of orthonormal, **time-dependent [Single-Particle Functions (SPFs)](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F148%2F4%2F044115%2F75514%2FA-new-collocation-based-multi-configuration-time)** $\varphi_{j_\kappa}^{(\kappa)}(q_\kappa, t)$ along each degree of freedom $\kappa$:
$$\Psi_{\text{MCTDH}}(q_1, \dots, q_f, t) = \sum_{j_1=1}^{n_1} \cdots \sum_{j_f=1}^{n_f} A_{j_1 \dots j_f}(t) \prod_{\kappa=1}^{f} \varphi_{j_\kappa}^{(\kappa)}(q_\kappa, t) = \sum_J A_J(t) \Phi_J(\mathbf{q}, t)$$
where each SPF is represented on the underlying $N$-point primitive DVR grid:
$$\varphi_{j_\kappa}^{(\kappa)}(q_\kappa, t) = \sum_{i_\kappa=1}^{N} c_{i_\kappa, j_\kappa}^{(\kappa)}(t) \, \chi_{i_\kappa}^{(\kappa)}(q_\kappa)$$

### Dirac–Frenkel Equations of Motion
Applying the **Dirac–Frenkel variational principle** $\langle \delta\Psi | i\hbar\partial_t - \hat{H} | \Psi \rangle = 0$ with the orthogonality and gauge constraints $\langle \varphi_j^{(\kappa)}(t) | \varphi_l^{(\kappa)}(t) \rangle = \delta_{jl}$ and $\langle \varphi_j^{(\kappa)}(t) | \dot{\varphi}_l^{(\kappa)}(t) \rangle = 0$ yields two coupled sets of equations (in atomic units, $\hbar = 1$):

1. **Core Tensor Evolution ($A_J$)**:
   $$i \dot{A}_J(t) = \sum_L \langle \Phi_J(t) | \hat{H} | \Phi_L(t) \rangle \, A_L(t)$$
2. **Single-Particle Function Evolution ($\varphi_j^{(\kappa)}$)**:
   $$i \dot{\varphi}_j^{(\kappa)}(q_\kappa, t) = \left( 1 - \hat{P}^{(\kappa)}(t) \right) \sum_{k, l=1}^{n_\kappa} \left( \boldsymbol{\rho}^{(\kappa)}(t) \right)^{-1}_{jk} \langle \hat{H} \rangle_{kl}^{(\kappa)} \, \varphi_l^{(\kappa)}(q_\kappa, t)$$
   where:
   * $\hat{P}^{(\kappa)} = \sum_{m=1}^{n_\kappa} |\varphi_m^{(\kappa)}\rangle\langle\varphi_m^{(\kappa)}|$ projects out the space already spanned by the current SPFs (preserving orthonormality $\langle \varphi_j^{(\kappa)} | \varphi_l^{(\kappa)} \rangle = \delta_{jl}$ for all $t$),
   * $\rho_{kl}^{(\kappa)} = \langle \Psi_k^{(\kappa)} | \Psi_l^{(\kappa)} \rangle$ is the single-particle reduced density matrix (whose eigenvalues, the **natural populations**, measure inter-mode entanglement and indicate whether $n_\kappa$ is large enough), and
   * $\langle \hat{H} \rangle_{kl}^{(\kappa)} = \langle \Psi_k^{(\kappa)} | \hat{H} | \Psi_l^{(\kappa)} \rangle$ is the mean-field Hamiltonian operator acting on mode $\kappa$.

### Sum-of-Products (SOP) Potentials, ML-MCTDH & TD-DMRG
To evaluate $\langle \Phi_J | \hat{H} | \Phi_L \rangle$ without $N^f$ multidimensional quadrature, $\hat{H}$ must be written as a **Sum of Products (SOP)** of 1D operators:
$$\hat{H} = \sum_{r=1}^{s} c_r \prod_{\kappa=1}^{f} \hat{h}_r^{(\kappa)}(q_\kappa)$$
In your research workflow, high-dimensional [MACE](https://www.google.com/search?q=https%3A%2F%2Fmace-docs.readthedocs.io%2F) potential energy surfaces can be compressed into SOP or tensor-network form via **POTFIT**, **Canonical Polyadic Decomposition (CPD)**, or **Tensor Cross Interpolation (TCI)**, and propagated for dozens to hundreds of modes using **Multi-Layer MCTDH ([ML-MCTDH](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F17%2F174106%2F200048%2FComparison-of-the-multi-layer-multi-configuration))** or **Time-Dependent DMRG ([TD-DMRG](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F17%2F174106%2F200048%2FComparison-of-the-multi-layer-multi-configuration))**.

## 4. Example Projects 7A & 7B

1. **Project 7A (`mace_md_pipeline.py`)**: Generate a training/validation `extxyz` dataset from MD snapshots of $\text{H}_2\text{O}$ (including `IsolatedAtom` references), implement the committee force uncertainty metric $\sigma_{F, i}(\mathbf{R})$ for active learning, and inspect the `mace_run_train` workflow.
2. **Project 7B (`mctdh_2d_solver.py`)**: Run and inspect a complete 2D MCTDH wavepacket propagation code for two bilinearly coupled anharmonic oscillators ($\hat{H} = \hat{h}_1(q_1) + \hat{h}_2(q_2) + c \, q_1 q_2$). Compare $n=1$ SPF (mean-field Time-Dependent Hartree, TDH) against $n=4$ SPFs (MCTDH) and the exact $N^2$ DVR grid solution, verifying that $n=4$ SPFs recover the exact autocorrelation function $C(t) = \langle \Psi(0) | \Psi(t) \rangle$ and inter-mode entanglement!
