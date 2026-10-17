# Module 03: Scientific Python, NumPy Tensor Contractions & Testing

In electronic structure theory and quantum dynamics, almost every equation is a matrix diagonalization or a multi-index tensor contraction. Writing idiomatic, vectorized [NumPy](https://www.google.com/search?q=https%3A%2F%2Fnumpy.org%2Fneps%2Fnep-0029-deprecation_policy.html) and [SciPy](https://www.google.com/search?q=https%3A%2F%2Fdocs.scipy.org%2Fdoc%2Fscipy%2Fdev%2Ftoolchain.html) code—and verifying it with [Pytest](https://www.google.com/search?q=https%3A%2F%2Fdocs.pytest.org%2Fen%2Flatest%2F)—is the single most important programming skill in our group.

## 1. Vectorization & Hermitian Eigenvalue Problems (`scipy.linalg.eigh`)

* Quantum mechanical observables (Hamiltonians $H$, Fock matrices $F$, overlap matrices $S$, density matrices $D$) are Hermitian ($A = A^\dagger$).
* Always use `scipy.linalg.eigh` (never `np.linalg.eig`) for Hermitian matrices: `eigh` guarantees real eigenvalues sorted in ascending order and strictly orthogonal eigenvectors, and runs significantly faster.

## 2. Contracting 4-Index Integrals with `np.einsum` ($\mathcal{O}(N^8) \to \mathcal{O}(N^5)$)

Consider transforming the 4-index two-electron repulsion integrals $(\mu\nu|\lambda\sigma)$ from the atomic orbital (AO) basis to the molecular orbital (MO) basis $(pq|rs)$:
$$(pq|rs) = \sum_{\mu\nu\lambda\sigma} C_{\mu p} C_{\nu q} C_{\lambda r} C_{\sigma s} (\mu\nu|\lambda\sigma)$$

* **The Wrong Way ($\mathcal{O}(N^8)$ nested loops)**: Eight nested `for` loops in Python require $N^8$ operations ($>2.5 \times 10^9$ iterations for just $N=15$ basis functions).
* **The Right Way ($\mathcal{O}(N^5)$ quarter-transformations via `np.einsum`)**:
  ```python
  # optimize=True automatically factors the contraction into four O(N^5) steps
  eri_mo = np.einsum("uvls,up,vq,lr,st->pqrt", eri_ao, C, C, C, C, optimize=True)
  ```
  Always pass `optimize=True` to `np.einsum` when contracting 3 or more tensors!

## 3. Example Project 3: Generalized Hückel Solver (`huckel_solver.py`)

Implement `build_huckel_hamiltonian` and `solve_pi_system` in `huckel_solver.py` without using any Python `for`-loops, and verify that benzene ($N=6$, cyclic) has doubly degenerate HOMO and LUMO levels with a total $\pi$-electron energy of $6\alpha + 8\beta$ (delocalization energy of $2\beta$ relative to three isolated double bonds).
