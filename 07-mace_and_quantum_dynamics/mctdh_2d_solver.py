"""Module 07B Project: 2D Multi-Configuration Time-Dependent Hartree (MCTDH) Wavepacket Solver."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, expm


@dataclass
class MCTDHResult:
    times: np.ndarray
    autocorrelation: np.ndarray
    norm_history: np.ndarray
    natural_populations_mode1: np.ndarray  # Shape (n_steps+1, n_spf)
    energy_history: np.ndarray


def build_1d_sine_dvr(
    n_grid: int, q_min: float, q_max: float, mass: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct 1D Colbert-Miller / Sine DVR grid points and kinetic energy matrix T."""
    length = q_max - q_min
    dq = length / (n_grid + 1)
    q = q_min + np.arange(1, n_grid + 1) * dq

    idx = np.arange(1, n_grid + 1)
    i_mat, j_mat = np.meshgrid(idx, idx, indexing="ij")
    prefactor = (np.pi / length) ** 2 / (2.0 * mass)

    T = np.zeros((n_grid, n_grid), dtype=np.float64)
    # Off-diagonal elements
    mask = i_mat != j_mat
    diff = i_mat[mask] - j_mat[mask]
    summ = i_mat[mask] + j_mat[mask]
    sign = (-1.0) ** diff
    T[mask] = prefactor * sign * (
        1.0 / (np.sin(np.pi * diff / (2.0 * (n_grid + 1))) ** 2)
        - 1.0 / (np.sin(np.pi * summ / (2.0 * (n_grid + 1))) ** 2)
    )
    # Diagonal elements
    diag_vals = prefactor * (
        (2.0 * (n_grid + 1) ** 2 + 1.0) / 3.0
        - 1.0 / (np.sin(np.pi * idx / (n_grid + 1)) ** 2)
    )
    np.fill_diagonal(T, diag_vals)
    return q, T, dq


def build_coupled_2d_operators(
    n_grid: int = 24,
    q_min: float = -4.0,
    q_max: float = 4.0,
    omega1: float = 1.0,
    omega2: float = 1.1,
    anharm: float = 0.03,
    coupling_c: float = 0.25,
) -> dict[str, np.ndarray | float]:
    """Construct SOP 1D operators for H = h1(q1) + h2(q2) + c * q1 * q2."""
    q1, T1, dq1 = build_1d_sine_dvr(n_grid, q_min, q_max, mass=1.0)
    q2, T2, dq2 = build_1d_sine_dvr(n_grid, q_min, q_max, mass=1.0)

    v1 = 0.5 * (omega1**2) * (q1**2) + anharm * (q1**4)
    v2 = 0.5 * (omega2**2) * (q2**2) + anharm * (q2**4)

    h1 = T1 + np.diag(v1)
    h2 = T2 + np.diag(v2)
    q1_op = np.diag(q1)
    q2_op = np.diag(q2)

    return {
        "q1": q1,
        "q2": q2,
        "dq": dq1,
        "h1": h1,
        "h2": h2,
        "q1_op": q1_op,
        "q2_op": q2_op,
        "coupling_c": coupling_c,
    }


def mctdh_rhs(
    A: np.ndarray,
    phi1: np.ndarray,
    phi2: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    q1_op: np.ndarray,
    q2_op: np.ndarray,
    c_coup: float,
    reg_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate Dirac-Frenkel time derivatives (dA/dt, dphi1/dt, dphi2/dt) and <H>.

    Shapes:
      A:    (n1, n2) core tensor
      phi1: (N_grid, n1) orthonormal SPFs for mode 1 (phi1.conj().T @ phi1 = I)
      phi2: (N_grid, n2) orthonormal SPFs for mode 2 (phi2.conj().T @ phi2 = I)
    """
    # 1. Single-particle matrix elements in current SPF basis
    h1_spf = phi1.conj().T @ (h1 @ phi1)  # (n1, n1)
    h2_spf = phi2.conj().T @ (h2 @ phi2)  # (n2, n2)
    q1_spf = phi1.conj().T @ (q1_op @ phi1)  # (n1, n1)
    q2_spf = phi2.conj().T @ (q2_op @ phi2)  # (n2, n2)

    # 2. Action of H on core tensor A: H(A) = h1_spf @ A + A @ h2_spf.T + c * q1_spf @ A @ q2_spf.T
    HA = h1_spf @ A + A @ h2_spf.T + c_coup * (q1_spf @ A @ q2_spf.T)
    dA_dt = -1j * HA
    energy = float(np.real(np.vdot(A, HA)))

    # 3. Single-particle density matrices rho^(1) = A @ A^dagger, rho^(2) = A^T @ A^*
    rho1 = A @ A.conj().T
    rho2 = A.T @ A.conj()

    # Regularized inverse rho^(-1) to handle unoccupied SPFs smoothly
    evals1, evecs1 = eigh(rho1)
    rho1_inv = (
        evecs1
        @ np.diag(1.0 / (evals1 + reg_eps * np.exp(-evals1 / reg_eps)))
        @ evecs1.conj().T
    )

    evals2, evecs2 = eigh(rho2)
    rho2_inv = (
        evecs2
        @ np.diag(1.0 / (evals2 + reg_eps * np.exp(-evals2 / reg_eps)))
        @ evecs2.conj().T
    )

    # 4. Mean-field matrices Acting on Primitive Grid:
    # For mode 1: sum_l <H>^(1)_{k,l} phi1_l = h1(phi1) @ rho1.T + c * q1(phi1) @ (A @ q2_spf.T @ A^dagger).T
    # Note: Using convention <Psi_k^(1) | H | Psi_l^(1)>:
    # Only the off-separable coupling term c * q1 * q2 survives (1 - P^(1)) alongside h1(phi1) * rho1!
    mean_q2_1 = A @ q2_spf.T @ A.conj().T  # (n1, n1)
    rhs_phi1 = (h1 @ phi1) @ rho1 + c_coup * (q1_op @ phi1) @ mean_q2_1
    rhs_phi1 = rhs_phi1 @ rho1_inv
    # Apply orthogonal projector (1 - P^(1)) where P^(1) v = phi1 @ (phi1^dagger @ v)
    rhs_phi1 = rhs_phi1 - phi1 @ (phi1.conj().T @ rhs_phi1)
    dphi1_dt = -1j * rhs_phi1

    mean_q1_2 = A.T @ q1_spf.T @ A.conj()  # (n2, n2)
    rhs_phi2 = (h2 @ phi2) @ rho2 + c_coup * (q2_op @ phi2) @ mean_q1_2
    rhs_phi2 = rhs_phi2 @ rho2_inv
    rhs_phi2 = rhs_phi2 - phi2 @ (phi2.conj().T @ rhs_phi2)
    dphi2_dt = -1j * rhs_phi2

    return dA_dt, dphi1_dt, dphi2_dt, energy


def run_2d_mctdh(
    n_spf: int = 4,
    n_grid: int = 24,
    dt: float = 0.02,
    n_steps: int = 150,
    q1_shift: float = 0.8,
    q2_shift: float = -0.6,
) -> MCTDHResult:
    """Propagate a displaced 2D Gaussian wavepacket using MCTDH with n_spf Single-Particle Functions."""
    ops = build_coupled_2d_operators(n_grid=n_grid)
    q1 = ops["q1"]
    q2 = ops["q2"]
    h1, h2 = ops["h1"], ops["h2"]
    q1_op, q2_op = ops["q1_op"], ops["q2_op"]
    c_coup = float(ops["coupling_c"])

    # Build orthonormal initial SPFs: phi_0 = displaced Gaussian, phi_{1..n-1} = Gram-Schmidt excited states
    g1 = np.exp(-0.5 * (q1 - q1_shift) ** 2).astype(np.complex128)
    g1 /= np.linalg.norm(g1)
    _, evecs1 = eigh(h1)
    basis1 = np.column_stack([g1, evecs1[:, :n_spf]])
    phi1, _ = np.linalg.qr(basis1)
    if np.real(np.vdot(g1, phi1[:, 0])) < 0:
        phi1[:, 0] *= -1.0
    phi1 = phi1[:, :n_spf]

    g2 = np.exp(-0.5 * (q2 - q2_shift) ** 2).astype(np.complex128)
    g2 /= np.linalg.norm(g2)
    _, evecs2 = eigh(h2)
    basis2 = np.column_stack([g2, evecs2[:, :n_spf]])
    phi2, _ = np.linalg.qr(basis2)
    if np.real(np.vdot(g2, phi2[:, 0])) < 0:
        phi2[:, 0] *= -1.0
    phi2 = phi2[:, :n_spf]

    # Initial core tensor A: wavepacket starts purely in product state phi1_0 * phi2_0
    A = np.zeros((n_spf, n_spf), dtype=np.complex128)
    A[0, 0] = 1.0 + 0.0j

    psi0_grid = np.outer(phi1[:, 0], phi2[:, 0])

    times = np.zeros(n_steps + 1)
    autocorr = np.zeros(n_steps + 1, dtype=np.complex128)
    norms = np.zeros(n_steps + 1)
    nat_pops = np.zeros((n_steps + 1, n_spf))
    energies = np.zeros(n_steps + 1)

    _, _, _, e0 = mctdh_rhs(A, phi1, phi2, h1, h2, q1_op, q2_op, c_coup)
    autocorr[0] = 1.0 + 0.0j
    norms[0] = 1.0
    nat_pops[0] = np.sort(np.linalg.eigvalsh(A @ A.conj().T))[::-1]
    energies[0] = e0

    for step in range(1, n_steps + 1):
        # RK4 integration of coupled (A, phi1, phi2) system
        dA1, dp1_1, dp2_1, _ = mctdh_rhs(A, phi1, phi2, h1, h2, q1_op, q2_op, c_coup)
        dA2, dp1_2, dp2_2, _ = mctdh_rhs(
            A + 0.5 * dt * dA1,
            phi1 + 0.5 * dt * dp1_1,
            phi2 + 0.5 * dt * dp2_1,
            h1,
            h2,
            q1_op,
            q2_op,
            c_coup,
        )
        dA3, dp1_3, dp2_3, _ = mctdh_rhs(
            A + 0.5 * dt * dA2,
            phi1 + 0.5 * dt * dp1_2,
            phi2 + 0.5 * dt * dp2_2,
            h1,
            h2,
            q1_op,
            q2_op,
            c_coup,
        )
        dA4, dp1_4, dp2_4, _ = mctdh_rhs(
            A + dt * dA3,
            phi1 + dt * dp1_3,
            phi2 + dt * dp2_3,
            h1,
            h2,
            q1_op,
            q2_op,
            c_coup,
        )

        A = A + (dt / 6.0) * (dA1 + 2.0 * dA2 + 2.0 * dA3 + dA4)
        phi1 = phi1 + (dt / 6.0) * (dp1_1 + 2.0 * dp1_2 + 2.0 * dp1_3 + dp1_4)
        phi2 = phi2 + (dt / 6.0) * (dp2_1 + 2.0 * dp2_2 + 2.0 * dp2_3 + dp2_4)

        psi_t_grid = phi1 @ A @ phi2.T
        times[step] = step * dt
        autocorr[step] = np.vdot(psi0_grid, psi_t_grid)
        norms[step] = float(np.linalg.norm(psi_t_grid))
        nat_pops[step] = np.sort(np.real(np.linalg.eigvalsh(A @ A.conj().T)))[::-1]
        _, _, _, energies[step] = mctdh_rhs(A, phi1, phi2, h1, h2, q1_op, q2_op, c_coup)

    return MCTDHResult(
        times=times,
        autocorrelation=autocorr,
        norm_history=norms,
        natural_populations_mode1=nat_pops,
        energy_history=energies,
    )


def run_exact_2d_dvr(
    n_grid: int = 24,
    dt: float = 0.02,
    n_steps: int = 150,
    q1_shift: float = 0.8,
    q2_shift: float = -0.6,
) -> np.ndarray:
    """Propagate the exact 2D DVR wavepacket on the full N^2 product grid for reference."""
    ops = build_coupled_2d_operators(n_grid=n_grid)
    q1, q2 = ops["q1"], ops["q2"]
    h1, h2 = ops["h1"], ops["h2"]
    q1_op, q2_op = ops["q1_op"], ops["q2_op"]
    c_coup = float(ops["coupling_c"])

    I = np.eye(n_grid)
    H_full = np.kron(h1, I) + np.kron(I, h2) + c_coup * np.kron(q1_op, q2_op)
    U_dt = expm(-1j * H_full * dt)

    g1 = np.exp(-0.5 * (q1 - q1_shift) ** 2).astype(np.complex128)
    g1 /= np.linalg.norm(g1)
    g2 = np.exp(-0.5 * (q2 - q2_shift) ** 2).astype(np.complex128)
    g2 /= np.linalg.norm(g2)
    psi0 = np.kron(g1, g2)

    autocorr = np.zeros(n_steps + 1, dtype=np.complex128)
    psi = psi0.copy()
    autocorr[0] = 1.0 + 0.0j
    for step in range(1, n_steps + 1):
        psi = U_dt @ psi
        autocorr[step] = np.vdot(psi0, psi)
    return autocorr


if __name__ == "__main__":
    c_exact = run_exact_2d_dvr(n_grid=20, dt=0.02, n_steps=100)
    res_tdh = run_2d_mctdh(n_spf=1, n_grid=20, dt=0.02, n_steps=100)
    res_mctdh = run_2d_mctdh(n_spf=4, n_grid=20, dt=0.02, n_steps=100)

    err_tdh = np.max(np.abs(res_tdh.autocorrelation - c_exact))
    err_mctdh = np.max(np.abs(res_mctdh.autocorrelation - c_exact))
    print(f"Max |C(t) - C_exact(t)| for n_spf=1 (TDH):   {err_tdh:.6f}")
    print(f"Max |C(t) - C_exact(t)| for n_spf=4 (MCTDH): {err_mctdh:.6f}")
    print("Final natural populations (n_spf=4):", np.round(res_mctdh.natural_populations_mode1[-1], 6))