"""Module 08B Project: Hierarchical Equations of Motion (HEOM) Spin-Boson Qubit Decoherence Solver."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class HEOMResult:
    times: np.ndarray
    sigma_z: np.ndarray  # Population difference <sigma_z(t)>
    sigma_x: np.ndarray  # Coherence <sigma_x(t)>
    trace_history: np.ndarray  # Tr[rho_0(t)]
    purity_history: np.ndarray  # Tr[rho_0(t)^2]
    tier1_norm: np.ndarray  # Frobenius norm of tier-1 auxiliary density operator ||rho_1(t)||


def build_spin_qubit_operators(
    epsilon: float = 0.5, delta: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct 2x2 spin-qubit Hamiltonian H_S = (epsilon/2)*sigma_z + (delta/2)*sigma_x and Pauli matrices."""
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    H_s = 0.5 * epsilon * sigma_z + 0.5 * delta * sigma_x
    S_op = sigma_z  # System-bath coupling operator
    return H_s, S_op, sigma_x, sigma_z


def heom_rhs(
    ados: np.ndarray,
    H_s: np.ndarray,
    S_op: np.ndarray,
    gamma: float,
    c_real: float,
    c_imag: float,
) -> np.ndarray:
    """Evaluate Tanimura-Kubo HEOM time derivative d(ados)/dt for a single Drude-Lorentz bath mode.

    Parameters
    ----------
    ados : np.ndarray
        Array of shape (n_tiers + 1, 2, 2) containing [rho_0, rho_1, ..., rho_{N_tier}].
    """
    n_total = ados.shape[0]
    deriv = np.zeros_like(ados)

    for n in range(n_total):
        rho_n = ados[n]
        # 1. Liouvillian + bath decay term: -i [H_S, rho_n] - n * gamma * rho_n
        comm_h = H_s @ rho_n - rho_n @ H_s
        term = -1j * comm_h - (n * gamma) * rho_n

        # 2. Coupling to tier n + 1: -i [S, rho_{n+1}]
        if n + 1 < n_total:
            rho_up = ados[n + 1]
            comm_up = S_op @ rho_up - rho_up @ S_op
            term += -1j * comm_up

        # 3. Coupling to tier n - 1: -i * n * (c_real * [S, rho_{n-1}] + i * c_imag * {S, rho_{n-1}})
        if n >= 1:
            rho_down = ados[n - 1]
            comm_down = S_op @ rho_down - rho_down @ S_op
            anti_down = S_op @ rho_down + rho_down @ S_op
            term += -1j * n * (c_real * comm_down + 1j * c_imag * anti_down)

        deriv[n] = term

    return deriv


def run_heom_spin_boson(
    epsilon: float = 0.5,
    delta: float = 1.0,
    reorg_lambda: float = 0.15,
    bath_gamma: float = 0.25,
    temperature_kt: float = 0.5,
    n_tiers: int = 6,
    dt: float = 0.02,
    n_steps: int = 300,
    init_coherence: bool = True,
) -> HEOMResult:
    """Propagate the Tanimura-Kubo HEOM hierarchy using RK4 integration."""
    H_s, S_op, sigma_x, sigma_z = build_spin_qubit_operators(epsilon=epsilon, delta=delta)

    # Drude-Lorentz bath correlation function C(t) = (c_real + i * c_imag) * exp(-gamma * t)
    c_real = 2.0 * reorg_lambda * temperature_kt
    c_imag = -reorg_lambda * bath_gamma

    ados = np.zeros((n_tiers + 1, 2, 2), dtype=np.complex128)
    if init_coherence:
        # Start in coherent superposition |+> = (|0> + |1>) / sqrt(2) -> <sigma_x(0)> = 1
        ados[0] = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    else:
        # Start in excited spin state |0> -> <sigma_z(0)> = 1
        ados[0] = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

    times = np.zeros(n_steps + 1)
    sz_hist = np.zeros(n_steps + 1)
    sx_hist = np.zeros(n_steps + 1)
    tr_hist = np.zeros(n_steps + 1)
    purity_hist = np.zeros(n_steps + 1)
    tier1_norm = np.zeros(n_steps + 1)

    rho0 = ados[0]
    sz_hist[0] = float(np.real(np.trace(rho0 @ sigma_z)))
    sx_hist[0] = float(np.real(np.trace(rho0 @ sigma_x)))
    tr_hist[0] = float(np.real(np.trace(rho0)))
    purity_hist[0] = float(np.real(np.trace(rho0 @ rho0)))

    for step in range(1, n_steps + 1):
        k1 = heom_rhs(ados, H_s, S_op, bath_gamma, c_real, c_imag)
        k2 = heom_rhs(ados + 0.5 * dt * k1, H_s, S_op, bath_gamma, c_real, c_imag)
        k3 = heom_rhs(ados + 0.5 * dt * k2, H_s, S_op, bath_gamma, c_real, c_imag)
        k4 = heom_rhs(ados + dt * k3, H_s, S_op, bath_gamma, c_real, c_imag)
        ados = ados + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        rho0 = ados[0]
        times[step] = step * dt
        sz_hist[step] = float(np.real(np.trace(rho0 @ sigma_z)))
        sx_hist[step] = float(np.real(np.trace(rho0 @ sigma_x)))
        tr_hist[step] = float(np.real(np.trace(rho0)))
        purity_hist[step] = float(np.real(np.trace(rho0 @ rho0)))
        if n_tiers >= 1:
            tier1_norm[step] = float(np.linalg.norm(ados[1]))

    return HEOMResult(
        times=times,
        sigma_z=sz_hist,
        sigma_x=sx_hist,
        trace_history=tr_hist,
        purity_history=purity_hist,
        tier1_norm=tier1_norm,
    )


if __name__ == "__main__":
    res_markov = run_heom_spin_boson(n_tiers=1, dt=0.02, n_steps=250)
    res_exact = run_heom_spin_boson(n_tiers=6, dt=0.02, n_steps=250)
    diff = np.max(np.abs(res_exact.sigma_x - res_markov.sigma_x))
    print(f"Trace conservation drift (n_tiers=6): {np.max(np.abs(res_exact.trace_history - 1.0)):.2e}")
    print(f"Final qubit purity Tr[rho^2]:        {res_exact.purity_history[-1]:.4f}")
    print(f"Max non-Markovian coherence difference (n_tiers=6 vs n_tiers=1): {diff:.5f}")