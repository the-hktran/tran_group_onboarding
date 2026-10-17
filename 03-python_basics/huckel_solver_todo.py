"""Module 03A Project (Student Starter with #TODOs): Vectorized Hückel MO Solver & ERI Transform."""
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

    # TODO 1: Allocate an (n_carbons, n_carbons) float64 matrix filled with zeros,
    #         place `alpha` on the main diagonal using np.fill_diagonal, and place `beta`
    #         on the first super-diagonal and sub-diagonal using vectorized index arrays
    #         idx = np.arange(n_carbons - 1) (NO Python for-loops!).
    #         If `cyclic` is True and n_carbons > 2, also set H[0, -1] = H[-1, 0] = beta.
    raise NotImplementedError("Complete TODO 1 in build_huckel_hamiltonian")


def solve_pi_system(H: np.ndarray, n_electrons: int) -> dict[str, float | np.ndarray]:
    """Diagonalize H and compute total pi-electron energy and HOMO-LUMO gap."""
    # TODO 2: Diagonalize the Hermitian matrix H using scipy.linalg.eigh(H).
    #         Compute total_pi_energy by summing 2 * evals[:n_occ] (plus evals[n_occ] if odd)
    #         and homo_lumo_gap = evals[n_occ] - evals[n_occ - 1].
    raise NotImplementedError("Complete TODO 2 in solve_pi_system")


def transform_eri_ao_to_mo(eri_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Transform 4-index ERIs from AO to MO basis in O(N^5) using np.einsum."""
    # TODO 3: Contract eri_ao (uvls) with four copies of MO coefficient matrix C
    #         (up, vq, lr, st -> pqrt) using np.einsum(..., optimize=True).
    raise NotImplementedError("Complete TODO 3 in transform_eri_ao_to_mo")