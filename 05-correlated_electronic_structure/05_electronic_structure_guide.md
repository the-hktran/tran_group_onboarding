# Module 05: Correlated Wavefunction Methods (CI, MP2, CCSD) & Bootstrap Embedding (Schmidt Bath, Matching Loops & Cumulant Energy)

Hartree–Fock (Module 04) replaces instantaneous electron–electron repulsion with an average mean field, missing the **correlation energy** $E_{\text{corr}} = E_{\text{exact}} - E_{\text{HF}}$. While $E_{\text{corr}}$ is typically $< 1\%$ of the total energy of a molecule, it is on the exact same energy scale as chemical bond breaking, catalytic barriers, and spin gaps. This module walks you through the hierarchy of correlated wavefunction methods—**Configuration Interaction (CI)**, **Second-Order Møller–Plesset Perturbation Theory (MP2)**, and **Coupled Cluster Singles and Doubles (CCSD)**—and then shows how **Bootstrap Embedding ([BE](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2F10.1021%2Facs.jctc.4c01267))** breaks their $\mathcal{O}(N^5\text{–}N^7)$ scaling wall via the **Schmidt decomposition**, **self-consistent density matching loops**, and the **cumulant-based energy functional** ([QuEmb Toolbox](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Ftroyvvgroup%2Fquemb)).

## 1. Configuration Interaction (CI: CISD & Full CI)

### The Linear Variational Wavefunction Expansion
Starting from the Restricted Hartree–Fock reference determinant $|\Phi_0\rangle$, we can generate all excited Slater determinants by promoting occupied spin-orbitals ($i, j, k, \dots$) to virtual spin-orbitals ($a, b, c, \dots$):
$$|\Psi_{\text{CI}}\rangle = c_0 |\Phi_0\rangle + \sum_{ia} c_i^a |\Phi_i^a\rangle + \frac{1}{4}\sum_{ijab} c_{ij}^{ab} |\Phi_{ij}^{ab}\rangle + \cdots$$

* **Brillouin's Theorem**: Because the RHF Fock matrix is diagonal ($F_{ia} = 0$), singly excited determinants do not couple directly to the Hartree–Fock reference: $\langle \Phi_0 | \hat{H} | \Phi_i^a \rangle = F_{ia} = 0$. They only mix with $|\Phi_0\rangle$ indirectly through the doubles $|\Phi_{ij}^{ab}\rangle$.
* **Slater–Condon Matrix Elements**: Using chemist's MO integrals $(pq|rs)$, the correlation energy of any intermediate-normalized CI/CC wavefunction ($\langle \Phi_0 | \Psi \rangle = 1$) depends purely on the singles and doubles coefficients:
  $$E_{\text{corr}} = \langle \Phi_0 | \hat{H} - E_{\text{HF}} | \Psi \rangle = \sum_{ij}^{\text{occ}} \sum_{ab}^{\text{virt}} C_{ij}^{ab} \left[ 2(ia|jb) - (ib|ja) \right]$$
* **Full CI (FCI) vs. Truncated CISD**:
  * **Full CI** includes all $\binom{N_{\text{bf}}}{N_\alpha}\binom{N_{\text{bf}}}{N_\beta}$ determinants. It is exact within the chosen basis set, variationally bounded, and **size-extensive**, but scales factorially.
  * **CISD** truncates the linear expansion at singles and doubles ($\mathcal{O}(N^6)$). Because a linear sum $(1 + \hat{C}_1 + \hat{C}_2)|\Phi_0\rangle$ on two non-interacting $\text{H}_2$ monomers lacks the simultaneous quadruple excitation $\hat{C}_2(A)\hat{C}_2(B)|\Phi_0(A)\Phi_0(B)\rangle$, **truncated CISD is not size-extensive**: $E_{\text{CISD}}(A + B) > E_{\text{CISD}}(A) + E_{\text{CISD}}(B)$ ([Psi4NumPy CI Tutorial](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FConfiguration-Interaction%2FCI_DL.py)).

## 2. Second-Order Møller–Plesset Perturbation Theory ([MP2](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FTutorials%2F05_Moller-Plesset%2F5a_conventional-mp2.ipynb))

Møller–Plesset perturbation theory partitions the electronic Hamiltonian into the sum of one-electron Fock operators $\hat{H}^{(0)} = \sum_p \epsilon_p \hat{a}_p^\dagger \hat{a}_p$ and the fluctuation potential $\hat{V}_{\text{fluct}} = \hat{H} - \hat{H}^{(0)}$.

* **First-Order Doubles Amplitudes & Second-Order Energy**:
  For a closed-shell RHF reference with spatial MO energies $\{\epsilon_p\}$ and MO two-electron integrals $(ia|jb) = \sum_{\mu\nu\lambda\sigma} C_{\mu i} C_{\nu a} C_{\lambda j} C_{\sigma b} (\mu\nu|\lambda\sigma)$:
  $$t_{ij}^{ab(1)} = -\frac{(ia|jb)}{\epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j}, \qquad E_{\text{MP2}} = \sum_{i,j=1}^{n_{\text{occ}}} \sum_{a,b=n_{\text{occ}}+1}^{N_{\text{bf}}} \frac{(ia|jb)\left[ 2(ia|jb) - (ib|ja) \right]}{\epsilon_i + \epsilon_j - \epsilon_a - \epsilon_b}$$
* **Scaling & Limitations**: Transforming $(ia|jb)$ via quarter-transformations (`np.einsum` with `optimize=True`) scales as $\mathcal{O}(N^5)$. MP2 is strictly **size-extensive** and captures $\sim 80\text{–}90\%$ of dynamic correlation near equilibrium, but diverges when $\epsilon_{\text{LUMO}} - \epsilon_{\text{HOMO}} \to 0$ (e.g., bond dissociation or transition metals).

## 3. Coupled Cluster Singles and Doubles ([CCSD](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FCoupled-Cluster%2FSpin_Orbitals%2FCCSD%2FCCSD_DIIS.py))

To restore size-extensivity at finite truncation order, Coupled Cluster theory uses an **exponential ansatz**:
$$|\Psi_{\text{CCSD}}\rangle = e^{\hat{T}} |\Phi_0\rangle, \qquad \hat{T} = \hat{T}_1 + \hat{T}_2 = \sum_{ia} t_i^a \hat{a}_a^\dagger \hat{a}_i + \frac{1}{4}\sum_{ijab} t_{ij}^{ab} \hat{a}_a^\dagger \hat{a}_b^\dagger \hat{a}_j \hat{a}_i$$
Expanding $e^{\hat{T}} = 1 + \hat{T}_1 + \left(\hat{T}_2 + \frac{1}{2}\hat{T}_1^2\right) + \left(\hat{T}_1\hat{T}_2 + \frac{1}{6}\hat{T}_1^3\right) + \left(\frac{1}{2}\hat{T}_2^2 + \dots\right)$ automatically includes **disconnected quadruple excitations** $\frac{1}{2}\hat{T}_2^2$, guaranteeing exact size-extensivity!

* **Similarity-Transformed Hamiltonian & Amplitude Equations**:
  Multiplying $\hat{H} e^{\hat{T}} |\Phi_0\rangle = E_{\text{CCSD}} e^{\hat{T}} |\Phi_0\rangle$ on the left by $e^{-\hat{T}}$ defines the similarity-transformed Hamiltonian $\bar{H} = e^{-\hat{T}} \hat{H} e^{\hat{T}}$, whose Baker–Campbell–Hausdorff (BCH) expansion terminates **exactly** at the fourth nested commutator because $\hat{H}$ is a two-body operator:
  $$\bar{H} = \hat{H} + [\hat{H}, \hat{T}] + \frac{1}{2!}[[\hat{H}, \hat{T}], \hat{T}] + \frac{1}{3!}[[[\hat{H}, \hat{T}], \hat{T}], \hat{T}] + \frac{1}{4!}[[[[\hat{H}, \hat{T}], \hat{T}], \hat{T}], \hat{T}]$$
  Projecting onto $\langle \Phi_0 |$, $\langle \Phi_i^a |$, and $\langle \Phi_{ij}^{ab} |$ gives the CCSD energy and nonlinear amplitude equations ($\mathcal{O}(N_{\text{occ}}^2 N_{\text{virt}}^4)$ scaling):
  $$E_{\text{CCSD}} = E_{\text{HF}} + \sum_{ia} f_{ia} t_i^a + \frac{1}{4}\sum_{ijab} \langle ij \| ab \rangle \left( t_{ij}^{ab} + 2 t_i^a t_j^b \right)$$
  $$\langle \Phi_i^a | e^{-\hat{T}} \hat{H} e^{\hat{T}} | \Phi_0 \rangle = 0, \qquad \langle \Phi_{ij}^{ab} | e^{-\hat{T}} \hat{H} e^{\hat{T}} | \Phi_0 \rangle = 0$$
* **CCSD Reduced Density Matrices (1-RDM & 2-RDM)**: Because $\bar{H}$ is non-Hermitian, computing exact expectation values and analytical RDMs $\gamma_{pq} = \langle \Phi_0 | (1 + \hat{\Lambda}) e^{-\hat{T}} \hat{a}_p^\dagger \hat{a}_q e^{\hat{T}} | \Phi_0 \rangle$ requires solving the linear left-eigenvector ($\Lambda$) equations ([PySCF CCSD RDM guide](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Findex.html); [BE2-CCSD(T) density matrices, JCTC 2025](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2Fabs%2F10.1021%2Facs.jctc.5c01060)).

## 4. Bootstrap Embedding ([BE](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2F10.1021%2Facs.jctc.4c01267)): Schmidt Decomposition, Matching Loops & Cumulant Energy

How do we achieve CCSD or FCI accuracy on systems with hundreds or thousands of atoms without hitting the $\mathcal{O}(N^6\text{–}N^7)$ wall? **Bootstrap Embedding (BE)** ([Welborn, Tsuchimochi \& Van Voorhis 2016](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F11nN9UOAku31tvnLTWJ3peB9QWuAi8jSb%2Fview%3Fusp%3Ddrivesdk); [Tran, Ye \& Van Voorhis, JCTC 2024](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2F10.1021%2Facs.jctc.4c01267); implemented in [QuEmb](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Ftroyvvgroup%2Fquemb) and [J. Phys. Chem. A 2025](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fjpcafh%2Farticle%2F129%2F28%2F6538%2F3651585%2FQuEmb-A-Toolbox-for-Bootstrap-Embedding)) exploits the spatial locality of electron correlation through five steps:

### Step 1: Orthonormal Localized Orbitals (LOs) & Overlapping Fragments ($\text{BE}(n)$)
We transform the AO basis into an orthonormal **Localized Orbital (LO)** basis $\{|\phi_\mu\rangle\}_{\mu=1}^{N_{\text{LO}}}$ (via Löwdin $S^{-1/2}$ in minimal bases, or Intrinsic Atomic Orbitals + Projected Atomic Orbitals, IAO+PAO, in extended bases; see [LargeBasisBE.pdf](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F11nN9UOAku31tvnLTWJ3peB9QWuAi8jSb%2Fview%3Fusp%3Ddrivesdk)).
We partition the molecule into $N_{\text{frag}}$ **overlapping fragments** $\{\mathbb{F}_A\}_{A=1}^{N_{\text{frag}}}$:
* Each fragment $\mathbb{F}_A = \mathbb{C}_A \cup \mathbb{E}_A$ consists of **center orbitals** $\mathbb{C}_A$ (the central atom/group) surrounded by **edge orbitals** $\mathbb{E}_A$ (up to $n-1$ coordination shells in $\text{BE}(n)$).
* Crucially, the center sets $\{\mathbb{C}_A\}$ form an **exact, non-overlapping partition** of all localized orbitals in the molecule:
  $$\bigcup_{A=1}^{N_{\text{frag}}} \mathbb{C}_A = \{1, 2, \dots, N_{\text{LO}}\}, \qquad \mathbb{C}_A \cap \mathbb{C}_B = \emptyset \quad (A \neq B)$$

### Step 2: Schmidt Decomposition to Construct Bath Orbitals
For a given fragment $A$ with $N_f^A = |\mathbb{F}_A|$ fragment orbitals and complement $\bar{\mathbb{F}}_A$, we partition the rows of the $N_{\text{LO}} \times n_{\text{occ}}$ occupied RHF coefficient matrix $C_{\text{occ}}$ into fragment rows $C_{\text{occ}, \mathbb{F}_A}$ and complement rows $C_{\text{occ}, \bar{\mathbb{F}}_A}$ ([Appendix A of LargeBasisBE.pdf](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F11nN9UOAku31tvnLTWJ3peB9QWuAi8jSb%2Fview%3Fusp%3Ddrivesdk)), and perform a Singular Value Decomposition (SVD):
$$C_{\text{occ}, \bar{\mathbb{F}}_A} = U^A \, \Sigma^A \, (V^A)^\dagger$$
* **Fragment Orbitals (FOs, $N_f^A$)**: The localized orbitals $|\phi_\mu\rangle$ for $\mu \in \mathbb{F}_A$.
* **Bath Orbitals (BOs, $N_b^A \le N_f^A$)**: Columns $U^A_{:, q}$ with fractional singular values $\epsilon_{\text{SVD}} < \Sigma^A_{qq} < 1 - \epsilon_{\text{SVD}}$. By the Schmidt theorem, **at most $N_f^A$ bath orbitals** have non-zero entanglement with the fragment!
* **Frozen Environment Orbitals (EOs)**: Columns $U^A_{:, r}$ with $\Sigma^A_{rr} = 1$, which remain doubly occupied and unentangled with $\mathbb{F}_A$, contributing a spin-summed environment density matrix $P^{\text{env}; A}_{\mu\nu} = 2 \sum_{r \in \text{EO}} U^A_{\mu r} U^A_{\nu r}$.

The $N_{\text{LO}} \times (N_f^A + N_b^A)$ embedding transformation matrix from the full LO basis into the active $(N_f^A\text{ FO} + N_b^A\text{ BO})$ impurity space of fragment $A$ is:
$$T^A = \begin{pmatrix} I_{N_f^A} & 0 \\ 0 & U^A_{\text{bath}} \end{pmatrix}$$

### Step 3: Effective Fragment Hamiltonian
Projecting the full-system Hamiltonian into the active space of fragment $A$ via $T^A$ yields the undressed $(N_f^A + N_b^A)$-orbital fragment Hamiltonian:
$$\hat{H}^A = \sum_{p,q=1}^{N_f^A + N_b^A} h^{\text{eff}, A}_{pq} \sum_\sigma \hat{a}^{A\dagger}_{p\sigma} \hat{a}^A_{q\sigma} + \frac{1}{2} \sum_{p,q,r,s=1}^{N_f^A + N_b^A} V^A_{pqrs} \sum_{\sigma\tau} \hat{a}^{A\dagger}_{p\sigma} \hat{a}^{A\dagger}_{r\tau} \hat{a}^A_{s\tau} \hat{a}^A_{q\sigma}$$
where:
$$h^{\text{eff}, A}_{pq} = \left( (T^A)^\dagger \left[ h_{\text{LO}} + G_{\text{LO}}[P^{\text{env}; A}] \right] T^A \right)_{pq}, \qquad V^A_{pqrs} = \sum_{\mu\nu\lambda\sigma} T^A_{\mu p} T^A_{\nu q} T^A_{\lambda r} T^A_{\sigma s} V^{\text{LO}}_{\mu\nu\lambda\sigma}$$

### Step 4: The Bootstrap Embedding Density-Matching Loops
Because each fragment's bath orbitals are constructed from a mean-field (RHF) reference, electron correlation is least accurate on the **edge sites** $\mathbb{E}_A$ (which sit right next to the mean-field bath) and most accurate on the **center sites** $\mathbb{C}_A$ (which are buffered by the edge shell). Bootstrap Embedding remedies edge errors by constraining the edge 1-RDM elements of fragment $A$ to match the center 1-RDM elements of an overlapping fragment $B$ where those same orbitals are at the center ([Equations 22–26 of LargeBasisBE.pdf](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F11nN9UOAku31tvnLTWJ3peB9QWuAi8jSb%2Fview%3Fusp%3Ddrivesdk)):

1. **Matching Constraints**:
   * **Edge-to-Center Matching**: For each fragment $A$ and overlapping fragment $B \neq A$,
     $$P^A_{pq}(\boldsymbol{\Lambda}) - P^B_{pq}(\boldsymbol{\Lambda}) = 0 \qquad \forall \, p, q \in \mathbb{E}_A \cap \mathbb{C}_B$$
   * **Global Electron Number Conservation**: Across all fragment centers,
     $$\left( \sum_{A=1}^{N_{\text{frag}}} \sum_{p \in \mathbb{C}_A} P^A_{pp}(\boldsymbol{\Lambda}) \right) - N_e = 0$$
2. **Dressed Fragment Eigenvalue Equation**:
   Minimizing the constrained BE Lagrangian introduces local Lagrange multiplier potentials $\lambda^A_{pq}$ on the edge space $\mathbb{E}_A$ and a global chemical potential $\mu$ on the center space $\mathbb{C}_A$:
   $$\left[ \hat{H}^A + \sum_{p,q \in \mathbb{E}_A} \lambda^A_{pq} \sum_\sigma \hat{a}^{A\dagger}_{p\sigma} \hat{a}^A_{q\sigma} + \mu \sum_{p \in \mathbb{C}_A} \sum_\sigma \hat{a}^{A\dagger}_{p\sigma} \hat{a}^A_{p\sigma} \right] |\Psi^A(\boldsymbol{\Lambda})\rangle = E^A |\Psi^A(\boldsymbol{\Lambda})\rangle$$
3. **Newton–Raphson / Quasi-Newton Matching Loop**:
   Collecting all potentials into $\boldsymbol{\Lambda} = (\{\lambda^A_{pq}\}, \mu)$ and all matching + electron-count residuals into $\mathbf{R}(\boldsymbol{\Lambda})$:
   * **Initialize** $\boldsymbol{\Lambda}^{(0)} = \mathbf{0}$.
   * **For iteration $k = 0, 1, 2, \dots$ until $\|\mathbf{R}(\boldsymbol{\Lambda}^{(k)})\|_{\text{RMS}} < \epsilon_{\text{BE}}$**:
     1. Dress each fragment's one-electron matrix $\tilde{h}^A_{pq} = h^{\text{eff}, A}_{pq} + \lambda^A_{pq}\mathbf{1}_{(p,q)\in\mathbb{E}_A} + \mu\delta_{pq}\mathbf{1}_{p\in\mathbb{C}_A}$.
     2. Solve each fragment's correlated wavefunction $|\Psi^A(\boldsymbol{\Lambda}^{(k)})\rangle$ (via FCI or CCSD) to obtain $P^A(\boldsymbol{\Lambda}^{(k)})$.
     3. Assemble the residual vector $\mathbf{R}(\boldsymbol{\Lambda}^{(k)})$. If $\|\mathbf{R}\|_{\text{RMS}} < 10^{-8}$, break.
     4. Evaluate the Jacobian $J_{mn} = \partial R_m / \partial \Lambda_n$ (note that $\lambda^A$ only perturbs fragment $A$, so the Jacobian is block-sparse!) and take a damped/least-squares Newton step:
        $$\boldsymbol{\Lambda}^{(k+1)} = \boldsymbol{\Lambda}^{(k)} - \mathbf{J}^+ \mathbf{R}(\boldsymbol{\Lambda}^{(k)})$$

### Step 5: Calculating the Cumulant-Based Bootstrap Embedding Energy
Once the matching loop converges to $\boldsymbol{\Lambda}^*$, we extract the converged correlated fragment 1-RDM $P^A(\boldsymbol{\Lambda}^*)$ and 2-RDM $\Gamma^A(\boldsymbol{\Lambda}^*)$. **Crucially**, as noted in [Equation 27 of LargeBasisBE.pdf](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F11nN9UOAku31tvnLTWJ3peB9QWuAi8jSb%2Fview%3Fusp%3Ddrivesdk), the constraining potentials $(\boldsymbol{\lambda}^*, \mu^*)$ dress the wavefunction, **not** the physical Hamiltonian! We therefore contract the converged RDMs against the **undressed** fragment Fock matrix $F^{\text{HF}, A}$ and two-electron integrals $V^A_{pqrs}$ using the **two-body cumulant** ([Periodic Bootstrap Embedding, JCTC 2023](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2F10.1021%2Facs.jctc.3c00069); [Bootstrap Embedding in Extended Basis Sets, JCTC 2024](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2F10.1021%2Facs.jctc.4c01267)):

1. **Correlated 1-RDM Difference $\Delta P^A$ and Approximate Two-Body Cumulant $K^A$**:
   $$\Delta P^A_{pq} = P^A_{pq}(\boldsymbol{\Lambda}^*) - (P_{\text{HF}}^A)_{pq}, \qquad \tilde{K}^A_{pqrs} = \Gamma^A_{pqrs}(\boldsymbol{\Lambda}^*) - \left( P^A_{pq} P^A_{rs} - \frac{1}{2} P^A_{ps} P^A_{rq} \right)$$
   $$K^A_{pqrs} = \tilde{K}^A_{pqrs} + \Delta P^A_{pq} \Delta P^A_{rs} - \frac{1}{2} \Delta P^A_{ps} \Delta P^A_{rq}$$
2. **Total Cumulant-Based BE Energy**:
   $$E_{\text{BE}} = E_{\text{HF}} + \sum_{A=1}^{N_{\text{frag}}} \sum_{p \in \mathbb{C}_A} \left[ \sum_{q=1}^{N_f^A + N_b^A} F^{\text{HF}, A}_{pq} \, \Delta P^A_{pq} + \frac{1}{2} \sum_{q,r,s=1}^{N_f^A + N_b^A} V^A_{pqrs} \, K^A_{pqrs} \right]$$

## 5. Example Projects 5A & 5B

1. **Project 5A (`correlated_methods_ci_mp2_ccsd.py`)**: Implement and compare **MP2**, **CISD / FCI**, and **CCSD** from scratch on $\text{HeH}^+$ and on separated $\text{H}_2$ dimers to prove numerically that MP2 and CCSD are size-extensive while truncated CISD is not.
2. **Project 5B (`schmidt_bath_construction.py`)**: Implement the complete **Bootstrap Embedding** pipeline on an inhomogeneous 6-site correlated chain:
   * Perform the **Schmidt Decomposition** via SVD of $C_{\text{occ}, \bar{\mathbb{F}}_A}$ to build $T^A$ and $P^{\text{env}; A}$.
   * Run the **Newton–Raphson BE matching loop** (`run_bootstrap_embedding_matching_loop`) on overlapping 2-site $\text{BE}(2)$ fragments to match edge 1-RDMs $P^A_{\text{edge}}$ to neighbor center 1-RDMs $P^B_{\text{center}}$ and enforce $\sum_A P^A_{\text{center}} = N_e$.
   * Evaluate the **cumulant-based BE energy** and verify in `test_ci_mp2_ccsd_be.py` that moving from $\text{BE}(1)$ $\to$ unmatched $\text{BE}(2)$ $\to$ density-matched $\text{BE}(2)$ systematically drives the correlation energy toward the exact 6-site Full CI benchmark!
