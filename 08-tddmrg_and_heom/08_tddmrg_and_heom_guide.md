# Module 08: Tensor Network Dynamics (TD-DMRG / MPS) & Open Quantum Systems (HEOM)

This module introduces two cornerstone quantum dynamics methods directly used in our group's research on **polaron transport in organic semiconductors** and **spin–phonon decoherence in molecular qubits**:
1. **Time-Dependent Density Matrix Renormalization Group ([TD-DMRG](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2Fabs%2F10.1021%2Facs.jctc.8b00628))** on **Matrix Product States (MPS)** for simulating 1D and quasi-1D correlated exciton/polaron chains without exponential memory growth.
2. **Hierarchical Equations of Motion ([HEOM](https://www.google.com/search?q=https%3A%2F%2Fqutip.org%2Fdocs%2F4.7%2Fguide%2Fguide-heom.html))** for numerically exact, non-perturbative open quantum system dynamics of a spin or electronic subsystem coupled to a dissipative phonon bath.

## 1. Matrix Product States (MPS) & Time-Dependent DMRG ([TD-DMRG](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2Fabs%2F10.1021%2Facs.jctc.8b00628))

### Factorizing the Exponential Hilbert Space into a Tensor Train
For a chain of $L$ sites (each with local physical dimension $d$, such as $d=2$ for a spin/qubit or $d = d_{\text{el}} \times N_{\text{phonon}}$ for a vibronic Holstein site), the exact wavefunction tensor $C_{s_1 s_2 \dots s_L}$ has $d^L$ components. A **Matrix Product State (MPS)** decomposes this rank-$L$ tensor into a 1D chain of rank-3 site tensors $A^{[k]s_k}_{\alpha_{k-1}, \alpha_k}$ via successive Singular Value Decompositions (SVDs):
$$|\Psi(t)\rangle = \sum_{s_1, \dots, s_L=1}^{d} \sum_{\alpha_1, \dots, \alpha_{L-1}=1}^{M} A^{[1]s_1}_{1, \alpha_1}(t) \, A^{[2]s_2}_{\alpha_1, \alpha_2}(t) \cdots A^{[L]s_L}_{\alpha_{L-1}, 1}(t) \, |s_1 s_2 \cdots s_L\rangle$$
where the virtual indices $\alpha_k$ have maximum **bond dimension** $M$. Instead of $d^L$ parameters, an MPS stores only $\mathcal{O}(L \, d \, M^2)$ parameters!

### Schmidt Decomposition, Entanglement Entropy & Bond Dimension $M$
Cutting the chain at bond $k$ partitions the system into Left ($1 \dots k$) and Right ($k+1 \dots L$) blocks:
$$|\Psi\rangle = \sum_{\alpha=1}^{\chi_k} \sigma_\alpha \, |\Phi_\alpha^{\text{L}}\rangle \otimes |\Phi_\alpha^{\text{R}}\rangle$$
The singular values $\{\sigma_\alpha\}$ ($\sum_\alpha \sigma_\alpha^2 = 1$) determine the **bipartite von Neumann entanglement entropy**:
$$S_{\text{vN}}(k) = -\sum_{\alpha=1}^{\chi_k} \sigma_\alpha^2 \ln\left(\sigma_\alpha^2\right)$$
As long as $S_{\text{vN}}$ obeys an area law (or grows slowly in time), truncating the SVD at bond dimension $M$ (discarding singular values $\sigma_\alpha < \epsilon_{\text{SVD}}$) yields a numerically exact representation.

### Propagating an MPS in Time: Two-Site [TEBD](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1706.09537) & [TDVP](https://www.google.com/search?q=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC8889605%2F)
To evolve $|\Psi(t)\rangle$ under a nearest-neighbor Hamiltonian $\hat{H} = \sum_{k=1}^{L-1} \hat{h}_{k, k+1}$:
1. **Suzuki–Trotter Decomposition**: Split the short-time propagator into odd and even bond layers:
   $$e^{-i \hat{H} \Delta t} = \prod_{k \text{ odd}} e^{-i \hat{h}_{k, k+1} \Delta t / 2} \prod_{k \text{ even}} e^{-i \hat{h}_{k, k+1} \Delta t} \prod_{k \text{ odd}} e^{-i \hat{h}_{k, k+1} \Delta t / 2} + \mathcal{O}(\Delta t^3)$$
2. **Two-Site Gate Contraction & SVD Compression**:
   * Contract adjacent site tensors $A^{[k]}$ and $A^{[k+1]}$ into a two-site tensor $\Theta_{\alpha_{k-1}, s_k, s_{k+1}, \alpha_{k+1}}$.
   * Apply the $d^2 \times d^2$ two-site unitary gate $U_{k, k+1} = \exp(-i \hat{h}_{k, k+1} \tau)$ to the physical indices $(s_k, s_{k+1})$.
   * Reshape the updated $\tilde{\Theta}$ into a matrix of shape $(\chi_{k-1} d) \times (d \chi_{k+1})$, compute its SVD $\tilde{\Theta} = U \Sigma V^\dagger$, keep the largest $M$ singular values, and split $U \Sigma^{1/2}$ and $\Sigma^{1/2} V^\dagger$ back into updated tensors $\tilde{A}^{[k]}$ and $\tilde{A}^{[k+1]}$.
3. **Generalization to Long-Range Hamiltonians ([TDVP](https://www.google.com/search?q=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC8889605%2F))**: When $\hat{H}$ is written as a Matrix Product Operator (MPO) with long-range electron–phonon or Coulomb couplings, applying the Dirac–Frenkel variational principle directly to the MPS manifold yields the 1-site and 2-site **Time-Dependent Variational Principle (TDVP)** sweep algorithms ([comparison of ML-MCTDH and TD-DMRG](https://www.google.com/search?q=https%3A%2F%2Fpubs.aip.org%2Faip%2Fjcp%2Farticle%2F154%2F17%2F174106%2F200048%2FComparison-of-the-multi-layer-multi-configuration)).

## 2. Open Quantum System Dynamics via [HEOM](https://www.google.com/search?q=https%3A%2F%2Fqutip.org%2Fdocs%2F4.7%2Fguide%2Fguide-heom.html)

When a molecular spin qubit or chromophore ($\hat{H}_S$) interacts with a macroscopic thermal phonon bath ($\hat{H}_B$), tracking every bath oscillator wavepacket explicitly is neither necessary nor feasible. Instead, we evolve the reduced density matrix $\hat{\rho}_S(t) = \text{Tr}_B[\hat{\rho}_{\text{tot}}(t)]$.

### Why Markovian Master Equations (Lindblad / Redfield) Fail in Materials
Standard Redfield and Lindblad equations assume weak system–bath coupling ($\lambda \ll \Delta$) and instantaneous bath memory ($\tau_c \to 0$). In molecular spin qubits and organic semiconductors, local ligand vibrations have correlation times $\tau_c = 1/\gamma$ comparable to electronic or spin precession timescales, causing strong **non-Markovian memory effects** and coherent phonon back-flow.

### The Spin–Boson Hamiltonian & Drude–Lorentz Bath Correlation Function
Consider a two-level spin or electronic system coupled via operator $\hat{S} = \hat{\sigma}_z$ to a harmonic bath:
$$\hat{H} = \hat{H}_S + \sum_k \hbar\omega_k \hat{b}_k^\dagger \hat{b}_k + \hat{S} \otimes \sum_k g_k (\hat{b}_k^\dagger + \hat{b}_k), \qquad \hat{H}_S = \frac{\varepsilon}{2}\hat{\sigma}_z + \frac{\Delta}{2}\hat{\sigma}_x$$
For an overdamped **Drude–Lorentz spectral density** $J(\omega) = \frac{2 \lambda \gamma \omega}{\omega^2 + \gamma^2}$ (where $\lambda$ is the reorganization energy and $\gamma$ is the bath cutoff frequency), the bath correlation function $C(t) = \langle \hat{B}(t)\hat{B}(0)\rangle_B$ in the high-temperature regime ($k_B T \gtrsim \hbar\gamma$) takes the single-exponential form:
$$C(t) = \left( c_{\text{R}} + i c_{\text{I}} \right) e^{-\gamma t}, \qquad c_{\text{R}} = 2 \lambda k_B T, \qquad c_{\text{I}} = -\lambda \gamma$$
*(At cryogenic temperatures, additional Matsubara or Padé exponential poles $\sum_{m=1}^K c_m e^{-\nu_m t}$ are included identically; see the [QuTiP HEOM Guide](https://www.google.com/search?q=https%3A%2F%2Fqutip.org%2Fdocs%2F4.7%2Fguide%2Fguide-heom.html) and [QuantumToolbox.jl HEOM](https://www.google.com/search?q=https%3A%2F%2Fqutip.org%2FQuantumToolbox.jl%2Fstable%2Fusers_guide%2FHEOM)).*

### Tanimura–Kubo Hierarchy of Auxiliary Density Operators (ADOs)
By differentiating the Feynman–Vernon influence functional with respect to time, Tanimura and Kubo showed that $\hat{\rho}_0(t) \equiv \hat{\rho}_S(t)$ couples to a hierarchy of **Auxiliary Density Operators (ADOs)** $\{\hat{\rho}_n(t)\}_{n=0}^{N_{\text{tier}}}$ (in units where $\hbar = 1$):
$$\frac{d}{dt}\hat{\rho}_n(t) = -\left( i [\hat{H}_S, \hat{\rho}_n(t)] + n \gamma \hat{\rho}_n(t) \right) - i [\hat{S}, \hat{\rho}_{n+1}(t)] - i \, n \left( c_{\text{R}} [\hat{S}, \hat{\rho}_{n-1}(t)] + i c_{\text{I}} \{\hat{S}, \hat{\rho}_{n-1}(t)\} \right)$$
where $[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}$ is the commutator, $\{\hat{A}, \hat{B}\} = \hat{A}\hat{B} + \hat{B}\hat{A}$ is the anticommutator, and the hierarchy is truncated by setting $\hat{\rho}_{N_{\text{tier}}+1}(t) = 0$.

* **Tier $n = 0$**: The physical reduced density matrix $\hat{\rho}_0(t)$ with $\text{Tr}[\hat{\rho}_0(t)] = 1$.
* **Tiers $n \ge 1$**: Encode $n$-phonon non-Markovian memory correlations between the spin/exciton and the bath!

## 3. Example Projects 8A & 8B

1. **Project 8A (`tddmrg_mps_solver.py`)**: Build a 1D Matrix Product State (MPS) solver with two-site Suzuki–Trotter TEBD gates and SVD bond-dimension truncation. Simulate coherent exciton/polaron wavepacket spreading on an $L$-site chain, track the half-chain entanglement entropy $S_{\text{vN}}(t)$, and benchmark against exact Hilbert-space diagonalization.
2. **Project 8B (`heom_spin_boson_solver.py`)**: Implement the Tanimura–Kubo [HEOM](https://www.google.com/search?q=https%3A%2F%2Fqutip.org%2Fdocs%2F4.7%2Fguide%2Fguide-heom.html) hierarchy for a spin qubit coupled to a Drude–Lorentz phonon bath. Compare non-Markovian coherence revivals and thermalization ($N_{\text{tier}} = 6$) against the lowest-tier perturbative approximation ($N_{\text{tier}} = 1$).
