"""Module 06 Project: Velocity Verlet Molecular Dynamics & Tully Fewest-Switches Surface Hopping."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MDTrajectory:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray


@dataclass
class FSSHTrajectoryResult:
    final_position: float
    final_momentum: float
    active_state: int
    transmitted: bool
    n_hops: int
    n_frustrated_hops: int
    max_energy_drift: float
    norm_drift: float


# ==============================================================================
# Part A: Adiabatic Velocity Verlet Molecular Dynamics on a Morse Oscillator
# ==============================================================================
def morse_potential_and_force(
    r: float, d_e: float = 0.1745, a: float = 1.0282, r_eq: float = 1.4011
) -> tuple[float, float]:
    """Compute Morse potential V(r) and force F(r) = -dV/dr in atomic units (H2 default)."""
    exp_term = np.exp(-a * (r - r_eq))
    v_pot = float(d_e * (1.0 - exp_term) ** 2)
    # dV/dr = 2 * d_e * (1 - exp_term) * (a * exp_term) -> F = -dV/dr
    force = float(-2.0 * a * d_e * (1.0 - exp_term) * exp_term)
    return v_pot, force


def run_velocity_verlet_morse(
    r0: float = 1.80,
    v0: float = 0.0,
    mass: float = 918.57,
    dt: float = 2.0,
    n_steps: int = 500,
) -> MDTrajectory:
    """Propagate 1D classical nuclear dynamics on a Morse potential via Velocity Verlet."""
    times = np.zeros(n_steps + 1)
    pos = np.zeros(n_steps + 1)
    vel = np.zeros(n_steps + 1)
    ke = np.zeros(n_steps + 1)
    pe = np.zeros(n_steps + 1)

    r = r0
    v = v0
    v_pot, f_curr = morse_potential_and_force(r)

    pos[0], vel[0], pe[0], ke[0] = r, v, v_pot, 0.5 * mass * v**2

    for step in range(1, n_steps + 1):
        # 1. Half-step velocity update
        v_half = v + 0.5 * (f_curr / mass) * dt
        # 2. Full-step position update
        r_new = r + v_half * dt
        # 3. Evaluate new force at r_new
        v_pot_new, f_new = morse_potential_and_force(r_new)
        # 4. Second half-step velocity update
        v_new = v_half + 0.5 * (f_new / mass) * dt

        r, v, f_curr = r_new, v_new, f_new
        times[step] = step * dt
        pos[step] = r
        vel[step] = v
        pe[step] = v_pot_new
        ke[step] = 0.5 * mass * v**2

    return MDTrajectory(
        times=times,
        positions=pos,
        velocities=vel,
        kinetic_energy=ke,
        potential_energy=pe,
        total_energy=ke + pe,
    )


# ==============================================================================
# Part B: Tully's Model I (Single Avoided Crossing) & FSSH Dynamics
# ==============================================================================
def tully_model_1_adiabatic(
    r: float,
    A: float = 0.01,
    B: float = 1.6,
    C: float = 0.005,
    D: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute adiabatic energies E_0, E_1, forces F_0, F_1, NAC d_01(R), and eigenvectors U(R)."""
    if r >= 0.0:
        v11 = A * (1.0 - np.exp(-B * r))
        dv11 = A * B * np.exp(-B * r)
    else:
        v11 = -A * (1.0 - np.exp(B * r))
        dv11 = A * B * np.exp(B * r)

    v22 = -v11
    dv22 = -dv11
    v12 = C * np.exp(-D * r**2)
    dv12 = -2.0 * C * D * r * v12 / C if C != 0 else 0.0

    # Adiabatic eigenvalues E_0 (lower) and E_1 (upper)
    delta = 0.5 * (v22 - v11)
    gap_half = np.hypot(delta, v12)

    e0 = -gap_half
    e1 = gap_half
    energies = np.array([e0, e1], dtype=np.float64)

    # Mixing angle theta(R) such that tan(2*theta) = v12 / (-delta)
    theta = 0.5 * np.arctan2(v12, -delta)
    c_th, s_th = np.cos(theta), np.sin(theta)
    U = np.array([[c_th, -s_th], [s_th, c_th]], dtype=np.float64)

    # Analytical adiabatic forces F_k = -dE_k/dR via Hellmann-Feynman in 2x2 basis
    dH_diab = np.array([[dv11, dv12], [dv12, dv22]], dtype=np.float64)
    dH_adiab = U.T @ dH_diab @ U
    forces = -np.diag(dH_adiab)

    # Analytical nonadiabatic coupling d_01(R) = <Phi_0 | d/dR Phi_1> = dH_adiab[0,1] / (E_1 - E_0)
    d01 = float(dH_adiab[0, 1] / (e1 - e0))

    return energies, forces, d01, U


def rk4_electronic_step(
    c: np.ndarray,
    energies: np.ndarray,
    v_dot_d01: float,
    dt: float,
    n_substeps: int = 10,
) -> np.ndarray:
    """Integrate i dc/dt = H_eff * c over dt using RK4 sub-stepping."""
    sub_dt = dt / n_substeps

    def deriv(c_vec: np.ndarray) -> np.ndarray:
        dc0 = -1j * energies[0] * c_vec[0] - v_dot_d01 * c_vec[1]
        dc1 = -1j * energies[1] * c_vec[1] + v_dot_d01 * c_vec[0]
        return np.array([dc0, dc1], dtype=np.complex128)

    c_curr = c.copy()
    for _ in range(n_substeps):
        k1 = deriv(c_curr)
        k2 = deriv(c_curr + 0.5 * sub_dt * k1)
        k3 = deriv(c_curr + 0.5 * sub_dt * k2)
        k4 = deriv(c_curr + sub_dt * k3)
        c_curr = c_curr + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return c_curr


def run_single_fssh_trajectory(
    k0: float = 20.0,
    r0: float = -8.0,
    mass: float = 2000.0,
    dt: float = 1.0,
    max_steps: int = 8000,
    seed: int = 42,
    apply_decoherence: bool = True,
) -> FSSHTrajectoryResult:
    """Propagate a single Tully FSSH trajectory on Tully Model I starting on state 0."""
    rng = np.random.default_rng(seed)
    r = r0
    v = k0 / mass
    active_state = 0
    c = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    energies, forces, d01, _ = tully_model_1_adiabatic(r)
    e_tot_init = 0.5 * mass * v**2 + energies[active_state]
    max_e_drift = 0.0
    max_norm_drift = 0.0
    n_hops = 0
    n_frustrated = 0

    for _ in range(max_steps):
        if abs(r) > 10.0:
            break

        # 1. Velocity Verlet nuclear step on active_state
        v_half = v + 0.5 * (forces[active_state] / mass) * dt
        r_new = r + v_half * dt
        energies_new, forces_new, d01_new, _ = tully_model_1_adiabatic(r_new)
        v_new = v_half + 0.5 * (forces_new[active_state] / mass) * dt

        # 2. Propagate electronic amplitudes c(t) using midpoint quantities
        e_mid = 0.5 * (energies + energies_new)
        v_dot_d01_mid = 0.5 * (v * d01 + v_new * d01_new)
        c = rk4_electronic_step(c, e_mid, v_dot_d01_mid, dt, n_substeps=10)

        # 3. Compute Tully's fewest-switches hopping probability g_{k -> j}
        target_state = 1 - active_state
        d_kl = d01_new if (active_state == 0 and target_state == 1) else -d01_new
        pop_active = float(np.abs(c[active_state]) ** 2)
        if pop_active > 1e-12:
            b_lk = -2.0 * float(
                np.real(np.conj(c[target_state]) * c[active_state] * v_new * d_kl)
            )
            g_hop = max(0.0, (dt * b_lk) / pop_active)
        else:
            g_hop = 0.0

        # 4. Stochastic hop check & momentum rescaling
        if rng.random() < g_hop:
            delta_v_pot = energies_new[target_state] - energies_new[active_state]
            ke_avail = 0.5 * mass * v_new**2
            if ke_avail >= delta_v_pot:
                sign_v = 1.0 if v_new >= 0.0 else -1.0
                v_new = sign_v * float(np.sqrt(2.0 * (ke_avail - delta_v_pot) / mass))
                active_state = target_state
                n_hops += 1
            else:
                n_frustrated += 1

        # 5. Optional Granucci-Persico energy-based decoherence correction
        if apply_decoherence:
            ke_curr = 0.5 * mass * v_new**2
            de_gap = abs(energies_new[1] - energies_new[0])
            if ke_curr > 1e-10 and de_gap > 1e-12:
                tau = (1.0 + 0.1 / ke_curr) / de_gap
                c[1 - active_state] *= np.exp(-dt / tau)
                pop_other = float(np.abs(c[1 - active_state]) ** 2)
                if abs(c[active_state]) > 1e-14 and pop_other < 1.0:
                    c[active_state] *= np.sqrt(
                        (1.0 - pop_other) / (np.abs(c[active_state]) ** 2)
                    )

        r, v, energies, forces, d01 = r_new, v_new, energies_new, forces_new, d01_new
        e_tot_curr = 0.5 * mass * v**2 + energies[active_state]
        max_e_drift = max(max_e_drift, abs(e_tot_curr - e_tot_init))
        max_norm_drift = max(max_norm_drift, abs(float(np.sum(np.abs(c) ** 2)) - 1.0))

    return FSSHTrajectoryResult(
        final_position=r,
        final_momentum=mass * v,
        active_state=active_state,
        transmitted=(r > 0.0),
        n_hops=n_hops,
        n_frustrated_hops=n_frustrated,
        max_energy_drift=max_e_drift,
        norm_drift=max_norm_drift,
    )


def run_fssh_ensemble(
    k0: float = 20.0, n_traj: int = 50, base_seed: int = 100
) -> dict[str, float]:
    """Run an ensemble of N_traj FSSH trajectories and compute branching ratios."""
    results = [
        run_single_fssh_trajectory(k0=k0, seed=base_seed + i) for i in range(n_traj)
    ]
    trans_lower = (
        sum(1 for r in results if r.transmitted and r.active_state == 0) / n_traj
    )
    trans_upper = (
        sum(1 for r in results if r.transmitted and r.active_state == 1) / n_traj
    )
    refl_lower = (
        sum(1 for r in results if (not r.transmitted) and r.active_state == 0) / n_traj
    )
    max_drift = max(r.max_energy_drift for r in results)
    return {
        "k0_au": k0,
        "transmission_state_0": trans_lower,
        "transmission_state_1": trans_upper,
        "reflection_state_0": refl_lower,
        "max_energy_drift_eh": max_drift,
    }


if __name__ == "__main__":
    md_traj = run_velocity_verlet_morse()
    e_drift = np.max(np.abs(md_traj.total_energy - md_traj.total_energy[0]))
    print(f"[Part A] Morse NVE Velocity Verlet max energy drift: {e_drift:.2e} Eh")

    for k_init in (10.0, 25.0):
        stats = run_fssh_ensemble(k0=k_init, n_traj=40)
        print(
            f"[Part B] Tully Model I (k0={k_init:4.1f} a.u.): "
            f"P_trans(state 0)={stats['transmission_state_0']:.2f}, "
            f"P_trans(state 1)={stats['transmission_state_1']:.2f}, "
            f"max_dE={stats['max_energy_drift_eh']:.2e} Eh"
        )