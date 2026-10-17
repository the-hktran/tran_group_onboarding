"""Module 04 Project: Restricted Hartree-Fock (RHF), Pulay DIIS, and MP2 from Scratch."""
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
    return (evecs[:, keep] * s_inv_sqrt) @ evecs[:, keep].T


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
    """Solve the Pulay DIIS linear system to extrapolate the Fock matrix."""
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

    # Initial core guess
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


def compute_mp2_correlation(
    eri_ao: np.ndarray, C: np.ndarray, eps: np.ndarray, n_occ: int
) -> float:
    """Compute closed-shell MP2 correlation energy via O(N^5) np.einsum transformation."""
    C_occ = C[:, :n_occ]
    C_virt = C[:, n_occ:]
    iajb = np.einsum(
        "uvls,ui,va,lj,sb->iajb", eri_ao, C_occ, C_virt, C_occ, C_virt, optimize=True
    )
    ibja = np.swapaxes(iajb, 1, 3)
    e_occ = eps[:n_occ]
    e_virt = eps[n_occ:]
    denom = (
        e_occ[:, None, None, None]
        - e_virt[None, :, None, None]
        + e_occ[None, None, :, None]
        - e_virt[None, None, None, :]
    )
    return float(np.sum(iajb * (2.0 * iajb - ibja) / denom))