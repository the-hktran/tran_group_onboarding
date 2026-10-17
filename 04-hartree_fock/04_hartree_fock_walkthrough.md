# Module 04: Coding Restricted Hartree–Fock (RHF) & Pulay DIIS from Scratch

In this module, you will write a complete Restricted Hartree–Fock (RHF) self-consistent field (SCF) program in Python (`rhf_solver.py`), accelerate it with Pulay's [Direct Inversion in the Iterative Subspace (DIIS)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FTutorials%2F03_Hartree-Fock%2F3b_rhf-diis.ipynb), add a Second-Order Møller–Plesset (MP2) correlation energy calculator, and test your implementation against [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Fscf.html) using `pytest` (`test_rhf.py`).

## Step-by-Step RHF Algorithm

### Step 1: Integrals & the Core Hamiltonian
Given a closed-shell molecule with $N_{\text{elec}}$ electrons ($n_{\text{occ}} = N_{\text{elec}}/2$ doubly occupied spatial orbitals) and atomic orbital basis functions $\{\chi_\mu\}$, we extract the 1-electron overlap ($S$), kinetic ($T$), and nuclear attraction ($V$) matrices, plus the 4-index 2-electron repulsion integral (ERI) tensor $(\mu\nu|\lambda\sigma)$ in chemist's notation. The one-electron core Hamiltonian is:
$$H_{\mu\nu}^{\text{core}} = T_{\mu\nu} + V_{\mu\nu}$$

### Step 2: Symmetric Orthogonalization ($X = S^{-1/2}$)
The Roothaan–Hall equation $FC = SC\epsilon$ is a generalized eigenvalue problem because atomic orbitals are non-orthogonal ($S_{\mu\nu} \neq \delta_{\mu\nu}$). By diagonalizing the Hermitian overlap matrix $S = U s U^\dagger$, we construct the symmetric orthogonalization matrix:
$$X = S^{-1/2} = U s^{-1/2} U^\dagger$$
which satisfies $X^\dagger S X = I$. Defining transformed MO coefficients $C' = X^{-1} C$ and the transformed Fock matrix $F' = X^\dagger F X$ converts $FC = SC\epsilon$ into a standard eigenvalue equation:
$$F' C' = C' \epsilon \quad \Longrightarrow \quad C = X C'$$

### Step 3: Core Guess & Occupied Density Matrix
To start the SCF cycle ($k = 0$), approximate $F^{(0)} \approx H^{\text{core}}$, diagonalize $F'^{(0)} = X^\dagger H^{\text{core}} X$, back-transform $C^{(0)} = X C'^{(0)}$, and slice the lowest $n_{\text{occ}}$ columns $C_{\text{occ}}$ to build the density matrix:
$$D_{\mu\nu} = \sum_{i=1}^{n_{\text{occ}}} C_{\mu i} C_{\nu i}$$

### Step 4: Building the Fock Matrix ($F = H^{\text{core}} + 2J - K$)
In each SCF iteration, contract the density matrix $D$ with the 4-index ERIs using `np.einsum`:
$$J_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma} (\mu\nu|\lambda\sigma), \qquad K_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma} (\mu\lambda|\nu\sigma)$$
$$F_{\mu\nu} = H_{\mu\nu}^{\text{core}} + 2 J_{\mu\nu} - K_{\mu\nu}$$
The electronic and total RHF energies at iteration $k$ are:
$$E_{\text{elec}}^{(k)} = \sum_{\mu\nu} D_{\mu\nu}^{(k)} \left( H_{\mu\nu}^{\text{core}} + F_{\mu\nu}^{(k)} \right), \qquad E_{\text{total}}^{(k)} = E_{\text{elec}}^{(k)} + E_{\text{nuc}}$$

### Step 5: Convergence Acceleration via Pulay DIIS
At converged SCF, the MO Fock matrix is diagonal in the occupied-virtual block, which is equivalent to the vanishing of the AO commutator $FDS - SDF = 0$. At iteration $k$, we compute the orthonormalized orbital gradient residual:
$$e_k = X^\dagger \left( F_k D_k S - S D_k F_k \right) X$$
Pulay's [DIIS](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fpsi4%2Fpsi4numpy%2Fblob%2Fmaster%2FTutorials%2F03_Hartree-Fock%2F3b_rhf-diis.ipynb) stores the last $m$ Fock matrices $\{F_i\}$ and error vectors $\{e_i\}$ and finds weights $\{c_i\}$ ($\sum_i c_i = 1$) that minimize $\|\sum_i c_i e_i\|^2$ by solving:
$$\begin{pmatrix} B_{11} & \cdots & B_{1m} & -1 \\ \vdots & \ddots & \vdots & \vdots \\ B_{m1} & \cdots & B_{mm} & -1 \\ -1 & \cdots & -1 & 0 \end{pmatrix} \begin{pmatrix} c_1 \\ \vdots \\ c_m \\ \lambda \end{pmatrix} = \begin{pmatrix} 0 \\ \vdots \\ 0 \\ -1 \end{pmatrix}, \qquad B_{ij} = \text{Tr}(e_i^\dagger e_j)$$
We then diagonalize the extrapolated Fock matrix $F_{\text{DIIS}} = \sum_{i=1}^m c_i F_i$ to update $C$ and $D$.

### Step 6: Bonus — MP2 Correlation Energy in 5 Lines of Code
Once your RHF calculation converges, transform the ERIs into the MO basis $(ia|jb)$ (where $i,j$ index occupied orbitals and $a,b$ index virtual orbitals) and evaluate the closed-shell MP2 correlation energy:
$$E_{\text{MP2}} = \sum_{i,j}^{\text{occ}} \sum_{a,b}^{\text{virt}} \frac{(ia|jb) \left[ 2(ia|jb) - (ib|ja) \right]}{\epsilon_i + \epsilon_j - \epsilon_a - \epsilon_b}$$
