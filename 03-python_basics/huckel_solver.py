"""Module 03 Project: Vectorized Hückel MO Solver and AO-to-MO Integral Transformer."""
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

    H = np.zeros((n_carbons, n_carbons), dtype=np.float64)
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


def transform_eri_ao_to_mo(eri_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Transform 4-index ERIs from AO to MO basis in O(N^5) using np.einsum."""
    return np.einsum("uvls,up,vq,lr,st->pqrt", eri_ao, C, C, C, C, optimize=True)


if __name__ == "__main__":
    H_benzene = build_huckel_hamiltonian(6, alpha=0.0, beta=-1.0, cyclic=True)
    res = solve_pi_system(H_benzene, n_electrons=6)
    print("Benzene orbital energies (in units of |beta|):", np.round(res["orbital_energies"], 4))
    print(f"Total pi energy: {res['total_pi_energy']:.4f} |beta| (expected -8.0000)")